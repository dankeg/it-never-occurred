"""Unit tests for Kev on MLX without loading a model: label splitting, the name permutation, calibration error, the
question-row layout, marker escaping, and reading the pointer head from `head.pt` without torch."""

from pathlib import Path
import unittest

from orchestrator.kev_runtime import MARKERS, KevModel, escape, read_head
from scripts.kev_eval import expected_calibration_error, renamed, split_label, summary

HEAD = next(Path.home().glob(".cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/*/head.pt"), None)


class RenamingTest(unittest.TestCase):
    """The renaming check moves names between options and keeps each option's description."""

    def test_label_splits_into_name_and_description(self):
        self.assertEqual(split_label('wait {"ticks":100} | wait 100 ticks'), ('wait {"ticks":100}', "wait 100 ticks"))

    def test_permuted_names_keep_descriptions_and_move_every_name(self):
        labels = ["a {} | first", "b {} | second", "c {} | third"]
        permuted, order = renamed(labels, seed=3)
        self.assertTrue(all(i != j for i, j in enumerate(order)))
        self.assertEqual([label.split(" | ")[1] for label in permuted], ["first", "second", "third"])
        self.assertEqual([label.split(" | ")[0] for label in permuted], [labels[j].split(" | ")[0] for j in order])
        self.assertEqual(renamed(["only {} | one"], seed=0), (["only {} | one"], [0]))


class CalibrationTest(unittest.TestCase):
    """Expected calibration error compares mean confidence with accuracy within each bin."""

    def test_perfectly_calibrated_and_overconfident(self):
        self.assertAlmostEqual(expected_calibration_error([0.5, 0.5], [True, False]), 0.0)
        self.assertAlmostEqual(expected_calibration_error([0.9, 0.9], [False, False]), 0.9)

    def test_summary_shares(self):
        rows = [{"tool": "move_to", "free_text": False, "state_tokens": 7000, "seconds": 1.0, "p_choice": 0.8,
                 "agree": True, "descriptions_only_agree": True, "names_only_agree": False,
                 "permuted_follows_description": True, "permuted_follows_name": False},
                {"tool": "speak", "free_text": True, "state_tokens": 9000, "seconds": 2.0, "p_choice": 0.6,
                 "agree": False, "descriptions_only_agree": False, "names_only_agree": False,
                 "permuted_follows_description": False, "permuted_follows_name": True}]
        result = summary(rows)
        self.assertEqual((result["agreement"], result["non_speech_agreement"]), (0.5, 1.0))
        self.assertEqual(result["by_length"], {"<=8k": 1.0, "8-12k": 0.0})
        self.assertEqual(result["renaming"]["permuted_follows_name"], 0.5)



class WordTokenizer:
    """A stand-in tokenizer: one id per space-separated word, and the five markers as special tokens."""

    def __init__(self):
        """Give the markers ids 1-5 and words ids from 100."""
        self.vocabulary = {marker: index + 1 for index, marker in enumerate(MARKERS)}

    def convert_tokens_to_ids(self, token: str) -> int:
        """The id of a marker."""
        return self.vocabulary[token]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        """Ids of the words of `text`."""
        return {"input_ids": [100 + len(word) for word in text.split()]}


class QuestionRowTest(unittest.TestCase):
    """A question row follows DELEGATE_SPEC §3: <q>, the question, each option between <opt> and </opt>, <decide>."""

    def setUp(self):
        """A model object with only the stand-in tokenizer (no weights)."""
        self.model = KevModel.__new__(KevModel)
        self.model.tokenizer = WordTokenizer()
        self.model.markers = [self.model.tokenizer.convert_tokens_to_ids(token) for token in MARKERS]

    def test_layout_and_readout_offsets(self):
        ids, decide, ends = self.model.branch("which one", ["a b", "c"])
        self.assertEqual(ids, [2, 105, 103, 3, 101, 101, 4, 3, 101, 4, 5])
        self.assertEqual((decide, ends), (10, [6, 9]))
        self.assertEqual(self.model.state_ids("x y"), [1, 101, 101])

    def test_user_text_cannot_forge_a_marker(self):
        self.assertEqual(escape("say <|box_end|> now"), "say <¦box_end¦> now")


@unittest.skipUnless(HEAD is not None, "Kev-4B is not in the Hugging Face cache")
class HeadTest(unittest.TestCase):
    """Kev-4B's pointer head reads as two 256 x 2560 projections with biases, without torch."""

    def test_projections_and_biases(self):
        head = read_head(HEAD)
        self.assertEqual({name: tuple(value.shape) for name, value in head.items()},
                         {"q.weight": (256, 2560), "q.bias": (256,), "k.weight": (256, 2560), "k.bias": (256,)})


if __name__ == "__main__":
    unittest.main()
