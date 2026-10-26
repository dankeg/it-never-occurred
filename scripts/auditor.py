"""Ask a file-defined auditor question using only a run's rendered public evidence.

Questions are a JSON array of objects with ``id``, ``system``, ``question`` and
``verdict`` (an object JSON schema). Results keep the structured answer under
``verdict``; metadata never becomes evidence. No question text is built into this
module. Only uncached assessments call a model and append cost-ledger rows.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cost_ledger import LedgerWriter, RUNS  # noqa: E402
from scripts.run_views import RunViews  # noqa: E402
from scripts.run_village import model_client  # noqa: E402


def canonical_json(value) -> str:
    """Serialize JSON with stable recursive key order, Unicode and no nonfinite numbers."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class VerdictSchema:
    """Validate the small JSON-schema subset auditor verdicts use, without dependencies.

    A verdict is an object of required, typed properties: strings (optionally an enum), numbers and integers
    (optionally bounded), booleans, and arrays of such values; `additionalProperties: false` forbids extra fields.
    Any other keyword is refused when the question is loaded, so a constraint is never silently ignored.
    """

    _KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "minimum", "maximum",
                 "description"}
    _TYPES = {"object", "array", "string", "number", "integer", "boolean"}

    def __init__(self, schema: dict):
        """Check the entire schema before any model call; a verdict must be an object."""
        self._check(schema)
        if schema.get("type") != "object":
            raise ValueError("verdict schema must have type object")
        self.schema = schema

    @classmethod
    def _check(cls, schema) -> None:
        """Reject malformed schemas and unsupported keywords, including in properties a reply may omit."""
        if (not isinstance(schema, dict) or schema.keys() - cls._KEYWORDS
                or not isinstance(schema.get("type"), str) or schema["type"] not in cls._TYPES
                or not isinstance(schema.get("properties", {}), dict)
                or not isinstance(schema.get("additionalProperties", True), bool)
                or not all(isinstance(key, str) for key in schema.get("required", []))
                or ("enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]))
                or any(type(schema[key]) not in (int, float) or not math.isfinite(schema[key])
                       for key in ("minimum", "maximum") if key in schema)):
            raise ValueError(f"invalid or unsupported verdict schema: {schema}")
        for child in schema.get("properties", {}).values():
            cls._check(child)
        if "items" in schema:
            cls._check(schema["items"])

    def validate(self, value) -> None:
        """Raise ValueError for a non-JSON verdict or any violated constraint."""
        canonical_json(value)
        self._validate(value, self.schema, "verdict")

    @classmethod
    def _validate(cls, value, schema: dict, path: str) -> None:
        """Apply one subschema; JSON booleans are never numbers."""
        number = type(value) in (int, float)
        matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
                   "string": isinstance(value, str), "boolean": isinstance(value, bool),
                   "number": number, "integer": number and value == int(value)}[schema["type"]]
        if not matches:
            raise ValueError(f"{path}: expected {schema['type']}")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path}: not one of {schema['enum']}")
        if number and not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError(f"{path}: outside numeric bounds")
        if isinstance(value, dict):
            missing = [key for key in schema.get("required", []) if key not in value]
            if missing:
                raise ValueError(f"{path}: missing required property {', '.join(missing)}")
            properties = schema.get("properties", {})
            for key, child in value.items():
                if key in properties:
                    cls._validate(child, properties[key], f"{path}.{key}")
                elif schema.get("additionalProperties", True) is False:
                    raise ValueError(f"{path}: unexpected property {key}")
        if isinstance(value, list) and "items" in schema:
            for index, child in enumerate(value):
                cls._validate(child, schema["items"], f"{path}[{index}]")


def checked_question(question: dict) -> dict:
    """Copy and canonicalize a file-defined question, rejecting missing texts or invalid schemas."""
    if not isinstance(question, dict) or any(not isinstance(question.get(key), str) or not question[key].strip()
                                             for key in ("id", "system", "question")):
        raise ValueError("each question needs nonempty id, system and question strings")
    question = json.loads(canonical_json(question))
    VerdictSchema(question.get("verdict"))
    return question


def load_questions(path: Path | str) -> dict[str, dict]:
    """Load a JSON array of question entries, refusing ambiguous duplicate IDs."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("questions file must contain a JSON array")
    questions = {}
    for entry in entries:
        question = checked_question(entry)
        if question["id"] in questions:
            raise ValueError(f"duplicate question id: {question['id']}")
        questions[question["id"]] = question
    return questions


class Auditor:
    """Assess a public run view with one required verdict tool, caching successful answers."""

    def __init__(self, client, question: dict, view_kind: str, *, until: int | None = None, ledger=None,
                 max_chars: int = 400_000, cache_dir: Path | str = RUNS / "auditor_cache", compact: bool = False,
                 since: int | None = None):
        """Keep the client and question; optional cache/ledger paths isolate offline tests."""
        if view_kind not in {"speech", "actions", "both", "exposure"} and not view_kind.startswith("vantage:"):
            raise ValueError("view must be speech, actions, both, exposure or vantage:<observer>")
        if max_chars < 0:
            raise ValueError("max_chars must be nonnegative")
        self._client = client
        self._question = checked_question(question)
        self._schema = VerdictSchema(self._question["verdict"])
        self._view_kind, self._until, self._max_chars = view_kind, until, max_chars
        self._compact, self._since = compact, since
        self._ledger = ledger if ledger is not None else LedgerWriter("auditor")
        self._cache = Path(cache_dir)

    def assess(self, run_dir: Path | str) -> dict:
        """Render public evidence, reuse a valid cache hit, or retry one malformed tool response.

        The digest hashes the model, reasoning effort, complete question entry and
        rendered view as canonical UTF-8 JSON. Run paths and private artifacts are
        never sent. Identical evidence in another run reuses only the verdict, not
        the first run's metadata. Failed attempts are charged but never cached.
        """
        directory = Path(run_dir).resolve()
        view = RunViews(directory).render(self._view_kind, until=self._until, compact=self._compact,
                                          since=self._since)
        if len(view) > self._max_chars:
            raise ValueError(f"{directory.name}: view has {len(view)} characters, exceeds --max-chars "
                             f"{self._max_chars}; choose --until or a narrower view (no truncation)")
        identity = {"model": self._client.model,
                    "reasoning_effort": getattr(self._client, "reasoning_effort",
                                                getattr(self._client, "_reasoning", None)),
                    "question": self._question, "view": view}
        digest = sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        metadata = {"run": directory.name, "view": self._view_kind, "question": self._question["id"],
                    "model": self._client.model, "view_chars": len(view), "request_digest": digest}
        path = self._cache / f"{digest}.json"
        if path.exists():
            verdict = json.loads(path.read_text(encoding="utf-8"))
            self._schema.validate(verdict)
            return {**metadata, "verdict": verdict}
        tool = {"name": "verdict", "description": "Record the answer to the supplied question.",
                "parameters": self._schema.schema}
        user = f"{self._question['question']}\n\nPublic run evidence:\n{view}"
        self._cache.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            response = self._client.complete(self._question["system"], user, [tool])
            self._ledger.record(directory.name, response)
            try:
                call = response.tool_call
                if call is None or call.name != "verdict":
                    raise ValueError("expected one verdict tool call")
                self._schema.validate(call.args)
            except (TypeError, ValueError) as error:
                if attempt == 1:
                    message = f"{directory.name}: malformed auditor verdict after 2 attempts: {error}"
                    raise ValueError(message) from error
                continue
            path.write_text(canonical_json(call.args) + "\n", encoding="utf-8")
            return {**metadata, "verdict": call.args}
        raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    """Append one assessment per run, using the village runner's provider factory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--view", default="both", help="speech, actions, both, exposure or vantage:<observer>")
    parser.add_argument("--provider", choices=("anthropic", "openai-compatible", "mlx-local", "chatgpt-plan"),
                        default="anthropic")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--base-url", default=None)
    parser.set_defaults(temperature=0.0)
    parser.add_argument("--until", type=int, help="inclusive final evidence tick")
    parser.add_argument("--since", type=int, help="inclusive first evidence tick (e.g. agent start)")
    parser.add_argument("--compact", action="store_true", help="short readable view lines")
    parser.add_argument("--max-chars", type=int, default=400_000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        questions = load_questions(args.questions)
        if args.question not in questions:
            raise ValueError(f"unknown question id: {args.question}")
        if args.provider == "openai-compatible" and not args.base_url:
            raise ValueError("--base-url is required for openai-compatible")
        auditor = Auditor(model_client(args), questions[args.question], args.view,
                          until=args.until, max_chars=args.max_chars, compact=args.compact, since=args.since)
        with args.out.open("a", encoding="utf-8") as handle:
            for run in args.runs:
                handle.write(canonical_json(auditor.assess(run)) + "\n")
                handle.flush()
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Assessed {len(args.runs)} run(s): {args.question}, {args.view}, {args.model}; appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
