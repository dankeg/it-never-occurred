"""Unit tests for isolated server assembly and launcher validation."""

import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import setup_server as setup


def jar_bytes():
    """Return an in-memory JAR fixture for launcher validation."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")
    return buffer.getvalue()


class SetupTests(unittest.TestCase):
    """Exercise dependency validation and atomic run-directory assembly."""
    def setUp(self):
        """Reset the fixtures used by each test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        self.gradle = self.root / "gradle"
        self.mod_jar = self.root / "simharness-0.1.0.jar"
        self.mod_jar.write_bytes(jar_bytes())
        version = (self.gradle / "caches/modules-2/files-2.1/net.fabricmc.fabric-api/fabric-api" /
                   setup.FABRIC_API_VERSION)
        (version / "digest").mkdir(parents=True)
        self.api_jar = version / "digest" / f"fabric-api-{setup.FABRIC_API_VERSION}.jar"
        self.api_jar.write_bytes(jar_bytes())

    def launcher_responses(self):
        """Return the mocked installer metadata and launcher downloads."""
        return [io.BytesIO(json.dumps([{"version": "1.2.3"}, {"version": "1.2.2"}]).encode()),
                io.BytesIO(jar_bytes())]

    def game_files(self, base):
        """Populate the pinned server and library JARs that a booted base supplies."""
        relative_jars = (Path(f".fabric/server/{setup.MINECRAFT_VERSION}-server.jar"),
                         Path("libraries/example/library.jar"),
                         Path(f"versions/{setup.MINECRAFT_VERSION}/server-{setup.MINECRAFT_VERSION}.jar"))
        for relative in relative_jars:
            target = base / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(jar_bytes())
        return relative_jars

    def clone_directory(self, command, *, check):
        """Simulate an APFS directory clone while checking the production command."""
        self.assertEqual(command[:3], ["cp", "-c", "-R"])
        self.assertTrue(check)
        shutil.copytree(command[3], command[4])

    def test_setup_resolves_latest_download_caches_and_writes_required_files(self):
        """Verify setup resolves latest download caches and writes required files."""
        with patch.object(setup, "urlopen", side_effect=self.launcher_responses()) as fetch:
            directory = setup.setup_server("base", self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle,
                                           bootstrap_base=True)
        self.assertEqual(directory, self.runs.resolve() / "base")
        self.assertEqual([call.args[0] for call in fetch.call_args_list],
                         [setup.INSTALLERS_URL,
                          "https://meta.fabricmc.net/v2/versions/loader/26.3/0.19.5/1.2.3/server/jar"])
        self.assertTrue(all(call.kwargs["timeout"] == 30 for call in fetch.call_args_list))
        self.assertEqual((directory / "eula.txt").read_text(), "eula=true\n")
        properties = (directory / "server.properties").read_text()
        for line in ("online-mode=false", "level-seed=20261016", r"level-type=minecraft\:flat",
                     "spawn-monsters=false", "spawn-animals=false", "spawn-npcs=false",
                     "pause-when-empty-seconds=0", "max-tick-time=-1", "view-distance=4",
                     "simulation-distance=4", "sync-chunk-writes=true", "level-name=world"):
            self.assertIn(line + "\n", properties)
        self.assertFalse((directory / "world").exists())
        self.assertEqual(sorted(path.name for path in (directory / "mods").iterdir()),
                         sorted([self.mod_jar.name, self.api_jar.name]))
        relative_jars = self.game_files(directory)
        (directory / "world").mkdir()
        (directory / "world/level.dat").write_bytes(b"world must not be cloned")
        with patch.object(setup, "urlopen") as fetch, \
                patch.object(setup.subprocess, "run", side_effect=self.clone_directory) as clone:
            second = setup.setup_server("second", self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle)
            fetch.assert_not_called()
        self.assertEqual(clone.call_count, 3)
        for relative in relative_jars:
            self.assertEqual((second / relative).read_bytes(), (directory / relative).read_bytes())
        self.assertFalse((second / "world").exists())

    def test_missing_base_requires_explicit_bootstrap_before_any_download(self):
        """A missing base fails for both base and fork names unless bootstrap is explicit."""
        for name in ("base", "fork"):
            with self.subTest(name=name), patch.object(setup, "urlopen") as fetch:
                with self.assertRaisesRegex(FileNotFoundError, "Missing prepared base server") as caught:
                    setup.setup_server(name, self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle)
                self.assertIn(str(self.runs / "base"), str(caught.exception))
                self.assertIn("scripts/prepare_world.py", str(caught.exception))
                fetch.assert_not_called()
                self.assertEqual(list(self.runs.iterdir()), [])

    def test_bootstrap_cannot_bypass_game_file_cloning_for_forks(self):
        """The prepare_world bootstrap option is restricted to the base run name."""
        with patch.object(setup, "urlopen") as fetch:
            with self.assertRaisesRegex(ValueError, "Only the base run may bootstrap"):
                setup.setup_server("fork", self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle,
                                   bootstrap_base=True)
            fetch.assert_not_called()
        self.assertFalse(self.runs.exists())

    def test_incomplete_game_files_fail_before_any_clone_even_with_bootstrap(self):
        """Missing directories, pinned server JARs and empty libraries never silently fall back to downloads."""
        required = (Path(".fabric/server"), Path("libraries"), Path("versions"),
                    Path(f".fabric/server/{setup.MINECRAFT_VERSION}-server.jar"),
                    Path(f"versions/{setup.MINECRAFT_VERSION}/server-{setup.MINECRAFT_VERSION}.jar"),
                    Path("libraries/example/library.jar"))
        staging = self.root / "staging"
        for index, relative in enumerate(required):
            base = self.root / f"base_{index}"
            self.game_files(base)
            missing = base / relative
            if missing.is_dir():
                shutil.rmtree(missing)
            else:
                missing.unlink()
            for bootstrap in (False, True):
                with self.subTest(missing=relative, bootstrap=bootstrap), \
                        patch.object(setup.subprocess, "run") as clone:
                    with self.assertRaisesRegex(FileNotFoundError, "Incomplete prepared base server") as caught:
                        setup._clone_game_files(base, staging, bootstrap_base=bootstrap)
                    expected = "libraries/**/*.jar" if relative == required[-1] else str(relative)
                    self.assertIn(expected, str(caught.exception))
                    clone.assert_not_called()
                    self.assertFalse(staging.exists())

    def test_incomplete_base_fails_before_downloading_or_publishing(self):
        """An existing but unbooted base cannot publish a fork or trigger launcher downloads."""
        (self.runs / "base").mkdir(parents=True)
        with patch.object(setup, "urlopen") as fetch:
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete prepared base server"):
                setup.setup_server("fork", self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle)
            fetch.assert_not_called()
        self.assertEqual(list(self.runs.iterdir()), [self.runs / "base"])

    def test_failed_clone_keeps_destination_absent_and_does_not_download(self):
        """Clone failures leave the base intact without publishing a partial fork."""
        relative_jars = self.game_files(self.runs / "base")
        with patch.object(setup, "urlopen") as fetch, \
                patch.object(setup.subprocess, "run", side_effect=OSError("clone failed")):
            with self.assertRaisesRegex(OSError, "clone failed"):
                setup.setup_server("fork", self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle)
            fetch.assert_not_called()
        self.assertEqual(list(self.runs.iterdir()), [self.runs / "base"])
        self.assertTrue(all((self.runs / "base" / relative).is_file() for relative in relative_jars))

    def test_existing_run_and_unsafe_names_are_never_overwritten(self):
        """Verify existing run and unsafe names are never overwritten."""
        self.runs.mkdir()
        existing = self.runs / "base"
        existing.mkdir()
        (existing / "important").write_text("keep")
        for name, exception in (("base", FileExistsError), ("../bad", ValueError),
                                ("_cache", ValueError), ("a/b", ValueError), ("", ValueError)):
            with self.subTest(name=name), patch.object(setup, "urlopen") as fetch:
                with self.assertRaises(exception):
                    setup.setup_server(name, self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle)
                fetch.assert_not_called()
        self.assertEqual((existing / "important").read_text(), "keep")

    def test_missing_mod_or_ambiguous_api_fail_before_downloading(self):
        """Verify missing mod or ambiguous api fail before downloading."""
        with patch.object(setup, "urlopen") as fetch:
            with self.assertRaises(FileNotFoundError):
                setup.setup_server("base", self.runs, mod_jar=self.root / "absent", gradle_cache=self.gradle)
            other = self.api_jar.parent.parent / "other" / self.api_jar.name
            other.parent.mkdir()
            other.write_bytes(jar_bytes())
            with self.assertRaisesRegex(FileNotFoundError, "exactly one Fabric API"):
                setup.setup_server("base", self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle)
            fetch.assert_not_called()

    def test_bad_or_failed_download_does_not_publish_partial_cache(self):
        """Verify bad or failed download does not publish partial cache."""
        for response in (io.BytesIO(b"not a jar"), OSError("network unavailable")):
            cache = self.runs / "_cache"
            with patch.object(setup, "urlopen", side_effect=[self.launcher_responses()[0], response]):
                with self.assertRaises((ValueError, OSError)):
                    setup.cached_launcher(cache)
            self.assertEqual(list(cache.iterdir()), [])
        (cache / "fabric-server-launch-26.3-0.19.5.jar").write_text("bad cache")
        with patch.object(setup, "urlopen") as fetch:
            with self.assertRaisesRegex(ValueError, "Invalid cached launcher"):
                setup.cached_launcher(cache)
            fetch.assert_not_called()

    def test_invalid_installer_metadata_is_not_used_in_url(self):
        """Verify invalid installer metadata is not used in url."""
        for metadata in ([], {}, [None], [{"version": "../../evil"}], [{"version": None}]):
            with self.subTest(metadata=metadata), patch.object(setup, "urlopen",
                                                             return_value=io.BytesIO(
                                                                 json.dumps(metadata).encode())) as fetch:
                with self.assertRaises(ValueError):
                    setup.cached_launcher(self.runs / "_cache")
                fetch.assert_called_once()

    def test_failed_assembly_keeps_destination_absent(self):
        """Verify failed assembly keeps destination absent."""
        with patch.object(setup, "urlopen", side_effect=self.launcher_responses()), \
                patch.object(setup.shutil, "copy2", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                setup.setup_server("base", self.runs, mod_jar=self.mod_jar, gradle_cache=self.gradle,
                                   bootstrap_base=True)
        self.assertFalse((self.runs / "base").exists())
        self.assertEqual([path.name for path in self.runs.iterdir()], ["_cache"])


if __name__ == "__main__":
    unittest.main()
