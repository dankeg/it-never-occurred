"""Unit tests for control framing, routine sampling, APFS clones, and JVM lifecycles."""

import contextlib
import io
import json
from pathlib import Path
import plistlib
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import harness


def components(game_time):
    """Return valid fake component digests at the requested game time."""
    return {"gameTime": game_time, **{key: "a" * 64 for key in harness.HASH_KEYS}}


def record(game_time):
    """Wrap fake component hashes in the sampled-record format."""
    return {"gameTime": game_time, "hash": components(game_time)}


class ClientTests(unittest.TestCase):
    """Exercise framing, deadlines, response validation, and connection cleanup."""
    def make_client(self, responses, timeout=1):
        """Return a client and mocked connection for the supplied response frames."""
        connection = Mock()
        connection.recv.side_effect = responses
        with patch.object(harness.socket, "create_connection", return_value=connection) as connect:
            client = harness.Client(25600, timeout=timeout)
        connect.assert_called_once_with(("127.0.0.1", 25600), timeout=timeout)
        self.addCleanup(client.close)
        return client, connection

    def test_partial_frames_ids_and_buffered_responses(self):
        """Verify partial frames ids and buffered responses."""
        response = json.dumps({"id": 1, "ok": True, "result": {"gameTime": 100}}).encode() + b"\n"
        later = b'{"id":2,"ok":true,"result":{}}\n'
        client, connection = self.make_client([response[:10], response[10:] + later])
        self.assertEqual(client.request("ping"), {"gameTime": 100})
        self.assertEqual(client.request("lockstep", {"enabled": True}), {})
        sent = [json.loads(call.args[0]) for call in connection.sendall.call_args_list]
        self.assertEqual(sent, [{"id": 1, "cmd": "ping", "args": {}},
                                {"id": 2, "cmd": "lockstep", "args": {"enabled": True}}])

    def test_command_error_is_not_retried_and_keeps_connection(self):
        """Verify command error is not retried and keeps connection."""
        client, connection = self.make_client([
            b'{"id":1,"ok":false,"error":"unknown bot"}\n',
            b'{"id":2,"ok":true,"result":{}}\n',
        ])
        with self.assertRaisesRegex(harness.CommandError, "unknown bot"):
            client.request("bot_state")
        self.assertEqual(client.request("ping"), {})
        self.assertEqual(connection.sendall.call_count, 2)
        connection.close.assert_not_called()

    def test_communication_methods_send_only_public_start_arguments(self):
        """Keep timed speech, talk alias, and ask framing free of local ticking."""
        replies = [json.dumps({"id": index, "ok": True, "result": {"accepted": True}}).encode() + b"\n"
                   for index in range(1, 4)]
        client, connection = self.make_client(replies)
        client.speak("alice", "hello", "shout", to="bob")
        client.say("alice", "hello")
        client.ask("alice", "npc:adult_00")
        sent = [json.loads(item.args[0]) for item in connection.sendall.call_args_list]
        self.assertEqual([(item["cmd"], item["args"]) for item in sent], [
            ("speak", {"name": "alice", "text": "hello", "volume": "shout", "to": "bob"}),
            ("speak", {"name": "alice", "text": "hello", "volume": "talk"}),
            ("ask", {"name": "alice", "npc": "npc:adult_00"})])

    def test_bad_envelopes_and_eof_close_connection(self):
        """Verify bad envelopes and eof close connection."""
        for response in (
            b"", b"{}", b"not json\n", b"[]\n",
            b'{"id":true,"ok":true,"result":{}}\n',
            b'{"id":9,"ok":true,"result":{}}\n',
            b'{"id":1,"ok":1,"result":{}}\n',
            b'{"id":1,"ok":true,"result":[]}\n',
        ):
            with self.subTest(response=response):
                client, connection = self.make_client([response, b""])
                with self.assertRaises(harness.ProtocolError):
                    client.request("ping")
                connection.close.assert_called_once()
                with self.assertRaises(harness.ProtocolError):
                    client.request("ping")

    def test_socket_timeout_closes_without_retry(self):
        """Verify socket timeout closes without retry."""
        client, connection = self.make_client([socket.timeout("late")])
        with self.assertRaises(TimeoutError):
            client.request("step", {"n": 20})
        connection.sendall.assert_called_once()
        connection.close.assert_called_once()

    def test_total_deadline_bounds_slow_partial_response(self):
        """Verify total deadline bounds slow partial response."""
        client, connection = self.make_client([b"{"], timeout=1)
        with patch.object(harness.time, "monotonic", side_effect=[10, 10.5, 11.1]):
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                client.request("ping")
        connection.recv.assert_called_once()
        connection.close.assert_called_once()

    def test_response_size_bound(self):
        """Verify response size bound."""
        client, connection = self.make_client([b"x" * (8 * 1024 * 1024 + 1)])
        with self.assertRaisesRegex(harness.ProtocolError, "oversized"):
            client.request("ping")
        connection.close.assert_called_once()


class DriverTests(unittest.TestCase):
    """Exercise routine movement and ordered hash sample validation."""
    def test_nested_step_hashes_and_direct_hash_components(self):
        """Verify nested step hashes and direct hash components."""
        result = {"gameTime": 142, "hashes": [record(120), record(140)]}
        self.assertEqual(harness.step_hashes(result, 103, 142), result["hashes"])
        self.assertEqual(harness.hash_components(components(120), 120), components(120))
        self.assertEqual(harness.step_hashes({"gameTime": 119, "hashes": []}, 103, 119), [])

    def test_malformed_hashes_and_missing_samples_fail(self):
        """Verify malformed hashes and missing samples fail."""
        for result in (
            {"gameTime": 141, "hashes": [record(120), record(140)]},
            {"gameTime": 142, "hashes": [record(120)]},
            {"gameTime": 142, "hashes": [record(140), record(120)]},
            {"gameTime": 142, "hashes": [{"gameTime": 120, "hash": components(121)}, record(140)]},
            {"gameTime": 142, "hashes": [components(120), components(140)]},
        ):
            with self.subTest(result=result), self.assertRaises(harness.ProtocolError):
                harness.step_hashes(result, 103, 142)
        for digest in ("x" * 64, "a" * 63, None):
            with self.assertRaises(harness.ProtocolError):
                harness.hash_components({**components(120), "entities": digest}, 120)

    def test_break_place_resolve_same_negative_coordinates_once(self):
        """Verify break place resolve same negative coordinates once."""
        client = Mock()
        client.request.return_value = {"pos": [-0.1, -60.0, -1.1], "yRot": 90}
        commands = harness.routine.actions_for(300)
        harness.apply_commands(client, commands)
        calls = client.request.call_args_list
        self.assertEqual([call.args[0] for call in calls],
                         ["bot_input", "bot_state", "bot_break", "bot_select", "bot_place"])
        for index in (2, 4):
            self.assertEqual({key: calls[index].args[1][key] for key in ("x", "y", "z")},
                             {"x": -2, "y": -61, "z": -2})
        self.assertNotIn("x", commands[1]["args"])

    def test_stationary_bot_fails_even_when_hashes_match(self):
        """Verify stationary bot fails even when hashes match."""
        client = Mock()
        client.request.side_effect = [
            {"gameTime": 100}, {"pos": [0.5, -60, 0.5]}, {},
            {"gameTime": 120, "hashes": [record(120)]}, {"pos": [0.5, -59, 0.5]},
        ]
        with self.assertRaisesRegex(harness.HarnessError, "never moved horizontally"):
            harness.run_routine(client, 120)

    def test_first_mismatch_and_missing_tail_print_components(self):
        """Verify first mismatch and missing tail print components."""
        for actual, expected_time in (([record(20), {**record(40), "hash": components(41)}], 40),
                                      ([record(20)], 40), ([record(40)], 20)):
            with self.subTest(actual=actual), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertFalse(harness.compare_hashes([record(20), record(40)], actual, "E2"))
                self.assertIn(f"game time {expected_time}", output.getvalue())
                self.assertIn('"blockEntities"', output.getvalue())


class CloneTests(unittest.TestCase):
    """Exercise same-volume APFS cloning and staging cleanup on failure."""
    def setUp(self):
        """Reset the fixtures used by each test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.destination = self.root / "destination"

    def test_verified_same_apfs_volume_uses_only_cp_clone(self):
        """Verify verified same apfs volume uses only cp clone."""
        def clone(command, **kwargs):
            """Simulate the clone operation expected by the enclosing test."""
            self.assertEqual(command[:3], ["/bin/cp", "-c", "-R"])
            self.assertEqual(command[3], str(self.source.resolve()))
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 120)
            Path(command[4]).mkdir()
            (Path(command[4]) / "level.dat").write_text("snapshot")

        with patch.object(harness.platform, "system", return_value="Darwin"), \
                patch.object(harness, "_apfs_volume", return_value="volume") as volume, \
                patch.object(harness.subprocess, "run", side_effect=clone) as run:
            harness.clone_apfs(self.source, self.destination)
        self.assertEqual(volume.call_count, 2)
        run.assert_called_once()
        self.assertEqual((self.destination / "level.dat").read_text(), "snapshot")

    def test_failed_clone_cleans_only_staging_and_never_falls_back(self):
        """Verify failed clone cleans only staging and never falls back."""
        def fail(command, **kwargs):
            """Raise the scripted clone failure after creating partial staging data."""
            Path(command[-1]).mkdir()
            raise subprocess.CalledProcessError(1, command, stderr="clonefile failed")

        (self.root / "unrelated").write_text("keep")
        with patch.object(harness.platform, "system", return_value="Darwin"), \
                patch.object(harness, "_apfs_volume", return_value="volume"), \
                patch.object(harness.subprocess, "run", side_effect=fail) as run:
            with self.assertRaisesRegex(harness.HarnessError, "clonefile failed"):
                harness.clone_apfs(self.source, self.destination)
        run.assert_called_once()
        self.assertFalse(self.destination.exists())
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["source", "unrelated"])

    def test_unsupported_cross_volume_or_existing_destinations_fail_closed(self):
        """Verify unsupported cross volume or existing destinations fail closed."""
        with patch.object(harness.platform, "system", return_value="Linux"), \
                patch.object(harness.subprocess, "run") as run:
            with self.assertRaisesRegex(harness.HarnessError, "no non-APFS"):
                harness.clone_apfs(self.source, self.destination)
            run.assert_not_called()
        with patch.object(harness.platform, "system", return_value="Darwin"), \
                patch.object(harness, "_apfs_volume", side_effect=["one", "two"]), \
                patch.object(harness.subprocess, "run") as run:
            with self.assertRaisesRegex(harness.HarnessError, "same APFS"):
                harness.clone_apfs(self.source, self.destination)
            run.assert_not_called()
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            harness.clone_apfs(self.source, self.destination)
        self.assertTrue(self.destination.exists())

    def test_clone_timeout_does_not_publish_destination(self):
        """Verify clone timeout does not publish destination."""
        with patch.object(harness.platform, "system", return_value="Darwin"), \
                patch.object(harness, "_apfs_volume", return_value="volume"), \
                patch.object(harness.subprocess, "run", side_effect=subprocess.TimeoutExpired("cp", 120)) as run:
            with self.assertRaisesRegex(harness.HarnessError, "APFS clone failed"):
                harness.clone_apfs(self.source, self.destination)
        run.assert_called_once()
        self.assertFalse(self.destination.exists())

    def test_volume_probe_failure_fails_closed(self):
        """Verify volume probe failure fails closed."""
        with patch.object(harness.subprocess, "run", side_effect=OSError("diskutil unavailable")) as run:
            with self.assertRaisesRegex(harness.HarnessError, "Cannot verify APFS volume"):
                harness._apfs_volume(self.source)
        run.assert_called_once()

    def test_diskutil_checks_device_from_df_not_arbitrary_directory(self):
        """Verify diskutil checks device from df not arbitrary directory."""
        for filesystem in ("apfs", "hfs"):
            info = plistlib.dumps({"FilesystemType": filesystem, "VolumeUUID": "volume"})
            results = [Mock(stdout="Filesystem 512-blocks Used Available Capacity Mounted on\n"
                                   "/dev/disk3s5 100 20 80 20% /System/Volumes/Data\n"), Mock(stdout=info)]
            with patch.object(harness.subprocess, "run", side_effect=results) as run:
                if filesystem == "apfs":
                    self.assertEqual(harness._apfs_volume(self.source), "volume")
                else:
                    with self.assertRaises(harness.HarnessError):
                        harness._apfs_volume(self.source)
                self.assertEqual(run.call_args_list[1].args[0],
                                 ["diskutil", "info", "-plist", "/dev/disk3s5"])


class ServerTests(unittest.TestCase):
    """Exercise bounded JVM startup and shutdown with process fixtures."""
    def setUp(self):
        """Reset the fixtures used by each test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.original_java_home = harness.JAVA_HOME
        (self.directory / "fabric-server-launch.jar").touch()
        java_home = self.directory / "jdk"
        (java_home / "bin").mkdir(parents=True)
        (java_home / "bin/java").touch()
        self.java_patch = patch.object(harness, "JAVA_HOME", java_home)
        self.java_patch.start()
        self.addCleanup(self.java_patch.stop)
        (self.directory / "mods").mkdir()
        built_mod = self.directory / "build/simharness-0.1.0.jar"
        built_mod.parent.mkdir()
        built_mod.write_bytes(b"current harness fixture")
        mod_patch = patch.object(harness, "MOD_JAR", built_mod)
        mod_patch.start()
        self.addCleanup(mod_patch.stop)
        provenance_patch = patch.object(harness, "record_provenance")  # git is not under test here
        provenance_patch.start()
        self.addCleanup(provenance_patch.stop)

    def test_exact_java_arguments_logging_ready_and_clean_stop(self):
        """Verify exact java arguments logging ready and clean stop."""
        process = Mock(returncode=None)
        process.poll.side_effect = lambda: process.returncode
        client = Mock()

        def request(command):
            """Handle a scripted control request using the enclosing test fixture."""
            if command == "stop":
                process.returncode = 0
                return {}
            return {"gameTime": 100}

        client.request.side_effect = request
        with patch.object(harness.subprocess, "Popen", return_value=process) as popen, \
                patch.object(harness, "Client", return_value=client), \
                patch.object(harness.socket, "socket"):
            server = harness.Server(self.directory, port=25601, base_seed=42)
            with server:
                self.assertIs(server.client, client)
                self.assertEqual(client.timeout, 60)
                self.assertEqual(self.original_java_home,
                                 Path.home() / ".local/jdks/jdk-25.0.4.1+1/Contents/Home")
            self.assertEqual(popen.call_args.args[0], [str(harness.JAVA_HOME / "bin/java"),
                "-Dsimharness.port=25601", "-Dsimharness.reseed_every=20", "-Dsimharness.base_seed=42",
                "-jar", "fabric-server-launch.jar", "nogui"])
            self.assertEqual(popen.call_args.kwargs["cwd"], self.directory.resolve())
            self.assertIs(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
            self.assertTrue(popen.call_args.kwargs["stdout"].closed)
            process.wait.assert_called_once_with(timeout=60)
            process.terminate.assert_not_called()
            client.close.assert_called_once()
            server.stop()

    def test_readiness_timeout_reaps_process_and_closes_log(self):
        """Verify readiness timeout reaps process and closes log."""
        process = Mock(returncode=None)
        process.poll.return_value = None
        with patch.object(harness.subprocess, "Popen", return_value=process), \
                patch.object(harness.socket, "socket"), \
                patch.object(harness.time, "monotonic", side_effect=[0, 1]), \
                patch.object(harness, "Client") as client:
            server = harness.Server(self.directory, startup_timeout=0.5)
            with self.assertRaisesRegex(harness.HarnessError, "readiness timed out"):
                server.start()
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)
            self.assertIsNone(server._log)
            client.assert_not_called()

    def test_failed_process_creation_closes_log(self):
        """Verify failed process creation closes log."""
        with patch.object(harness.subprocess, "Popen", side_effect=OSError("exec failed")), \
                patch.object(harness.socket, "socket"):
            server = harness.Server(self.directory)
            with self.assertRaisesRegex(OSError, "exec failed"):
                server.start()
            self.assertIsNone(server.process)
            self.assertIsNone(server._log)

    def test_early_exit_and_invalid_readiness_reap(self):
        """Verify early exit and invalid readiness reap."""
        for exited in (True, False):
            process = Mock(returncode=7 if exited else None)
            process.poll.return_value = process.returncode
            client = Mock()
            client.request.return_value = {"gameTime": "bad"}
            with patch.object(harness.subprocess, "Popen", return_value=process), \
                    patch.object(harness.socket, "socket"), \
                    patch.object(harness, "Client", return_value=client):
                server = harness.Server(self.directory)
                with self.assertRaises(harness.HarnessError):
                    server.start()
                self.assertIsNone(server._log)
                if not exited:
                    process.terminate.assert_called_once()
                    client.close.assert_called_once()

    def test_failed_stop_escalates_terminate_then_kill_with_bounded_waits(self):
        """Verify failed stop escalates terminate then kill with bounded waits."""
        server = harness.Server(self.directory, shutdown_timeout=2)
        process = Mock(returncode=None)
        process.poll.side_effect = lambda: process.returncode
        process.wait.side_effect = [subprocess.TimeoutExpired("java", 2),
                                    subprocess.TimeoutExpired("java", 5), -9]
        process.kill.side_effect = lambda: setattr(process, "returncode", -9)
        server.process = process
        server.client = Mock()
        server.client.request.side_effect = TimeoutError("stop hung")
        server._log = (self.directory / "log").open("ab")
        with self.assertRaisesRegex(harness.HarnessError, "Forced shutdown"):
            server.stop()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual([call.kwargs["timeout"] for call in process.wait.call_args_list], [2, 5, 5])
        self.assertIsNone(server.client)
        self.assertIsNone(server._log)


if __name__ == "__main__":
    unittest.main()
