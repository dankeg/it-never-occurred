"""Unit tests for the testimony audit: grounded news passes; fabricated origins and unheard relays are caught."""

import unittest

from scripts.audit_testimony import TestimonyAudit

CHANGE = {"type": "block_change", "id": 7, "pos": [1, 2, 3], "gameTime": 100}
SEEN = {"observer": "npc:a", "gameTime": 110, "changes": [{"id": 7}]}


def utterance(utt_id, speaker, key, chain, start, listeners, origin="change:7"):
    """A villager utterance carrying one news item."""
    return {"type": "utterance", "uttId": utt_id, "speaker": speaker, "speakerKey": key, "startedAt": start,
            "gameTime": start + 10, "listeners": listeners,
            "news": [{"newsId": "change:7", "originEventId": origin, "chain": chain, "subject": [1, 2, 3]}]}


class TestimonyAuditTest(unittest.TestCase):
    """Grounded first-hand and relayed news pass; fabricated or unheard news is flagged."""

    def test_grounded_first_hand_and_relay_pass(self):
        events = [CHANGE, utterance(1, "A", "npc:a", ["A"], 200, ["npc:b"]),
                  utterance(2, "B", "npc:b", ["A"], 300, [])]
        audit = TestimonyAudit(events, [SEEN]).run()
        self.assertEqual((audit.checked, audit.violations), (2, []))

    def test_fabricated_origin_is_caught(self):
        audit = TestimonyAudit([CHANGE, utterance(1, "A", "npc:a", ["A"], 200, [], origin="change:99")], [SEEN]).run()
        self.assertEqual(len(audit.violations), 1)

    def test_unheard_relay_and_unseen_first_hand_are_caught(self):
        events = [CHANGE, utterance(1, "B", "npc:b", ["A"], 300, []), utterance(2, "C", "npc:c", ["C"], 300, [])]
        audit = TestimonyAudit(events, [SEEN]).run()
        self.assertEqual(len(audit.violations), 2)


if __name__ == "__main__":
    unittest.main()
