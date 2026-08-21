from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from slm.checks import MechanicalCheck, normalize_token, run_mechanical_check
from slm.generation import Cell, CodeShape, Pressure, SeedDomain
from slm.scenarios import Category, LifetimeConcept, Scenario, require_authored

# Jaccard similarity above this against any eval scenario is treated as the same
# program rewritten, not an independent example.
CONTAMINATION_THRESHOLD = 0.8
SHINGLE_SIZE = 4
MIN_CODE_CHARS = 20
MAX_CODE_CHARS = 900

# One short line, deliberately carrying none of the spec: the behaviour has to live in
# the weights, not the prompt. The base model is evaluated with its best engineered
# prompt, so a tuned model that only needs this line is the honest comparison.
TRAIN_SYSTEM_PROMPT = "You are a Python state-lifetime tutor."


class Author(StrEnum):
    """Which path produced a row.

    Recorded per row so the published dataset is honest about its origin: the v1 pool
    was authored in-session, while `generate.py --target` is the reproducible path a
    grader can rerun.
    """

    IN_SESSION = "in_session"
    TEACHER = "teacher"


class RejectReason(StrEnum):
    """Why a candidate did not make it into the pool."""

    UNPARSEABLE_CODE = "unparseable_code"
    BUG_REGION_NOT_IN_CODE = "bug_region_not_in_code"
    CONCEPT_NOT_IN_AST = "concept_not_in_ast"
    CODE_LENGTH = "code_length"
    FORBIDDEN_TOKEN_IN_BUG_REGION = "forbidden_token_in_bug_region"
    CONTAMINATES_EVAL_SET = "contaminates_eval_set"
    DUPLICATE_CODE = "duplicate_code"
    DUPLICATE_STUDENT_MESSAGE = "duplicate_student_message"
    RESPONSE_FAILS_SPEC = "response_fails_spec"
    NEAR_MISS_PASSES_SPEC = "near_miss_passes_spec"


class GenerationProvenance(BaseModel):
    """Where one accepted example came from."""

    author: Author
    model_id: str | None = None
    batch: str


class Candidate(BaseModel):
    """One authored example before it has passed the gate.

    This is the wire format for both paths: what the teacher returns and what an
    in-session batch file contains. The cell fields are assigned by construction - they
    are the bucket that was requested, not something the author labelled after the fact.
    """

    lifetime_concept: LifetimeConcept
    code_shape: CodeShape
    category: Category
    seed_domain: SeedDomain
    pressure: Pressure | None = None

    code: str
    student_message: str
    bug: str
    bug_region: str
    expected_question_focus: str
    forbidden_fix_tokens: list[str]
    response: str
    near_miss: str | None = None

    @property
    def cell(self) -> Cell:
        """The grid bucket this candidate belongs to."""
        return Cell(
            lifetime_concept=self.lifetime_concept,
            code_shape=self.code_shape,
            category=self.category,
        )


class TrainingExample(BaseModel):
    """One accepted training row.

    Composes `Scenario` rather than redefining its fields, so `run_mechanical_check`,
    `judge_response`, and `slm/reporting.py` all operate on training rows unchanged.
    """

    scenario: Scenario
    rank: int
    code_shape: CodeShape
    seed_domain: SeedDomain
    pressure: Pressure | None
    response: str
    near_miss: str | None
    provenance: GenerationProvenance

    @property
    def cell(self) -> Cell:
        """The grid bucket this example fills."""
        return Cell(
            lifetime_concept=require_authored(
                self.scenario.lifetime_concept, "lifetime_concept", self.scenario.id
            ),
            code_shape=self.code_shape,
            category=self.scenario.category,
        )


class Rejection(BaseModel):
    """A candidate that failed the gate, kept for the failure-mode histogram and DPO pairs."""

    candidate: Candidate
    reason: RejectReason
    detail: str


# --- Normalisation ------------------------------------------------------------


def normalize_code(code: str) -> str:
    """Collapse whitespace so formatting differences do not hide a duplicate."""
    return " ".join(code.split())


def _shingles(text: str) -> set[str]:
    """Return overlapping token n-grams used for near-duplicate detection."""
    tokens = normalize_code(text).lower().split()
    if len(tokens) < SHINGLE_SIZE:
        return {" ".join(tokens)} if tokens else set()
    return {
        " ".join(tokens[i : i + SHINGLE_SIZE])
        for i in range(len(tokens) - SHINGLE_SIZE + 1)
    }


def jaccard(left: str, right: str) -> float:
    """Return token-shingle Jaccard similarity between two programs, 0.0 to 1.0."""
    a, b = _shingles(left), _shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- AST validation -----------------------------------------------------------
#
# Validation is at the concept level rather than per code shape. Twenty exact matchers
# would be brittle enough to reject good examples, and the failure worth catching is the
# author drifting to a different concept entirely - which a concept-level check does
# catch. The eval set doubles as the fixture: every scenario in data/scenarios.jsonl
# must pass its own concept's validator (see tests).

_MUTABLE_FACTORIES = frozenset({"list", "dict", "set", "defaultdict", "Counter", "deque"})


def _is_mutable_value(node: ast.expr | None) -> bool:
    """Return whether an expression evaluates to a fresh mutable container."""
    if node is None:
        return False
    if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in _MUTABLE_FACTORIES:
            return True
        if isinstance(func, ast.Attribute) and func.attr in {"copy", "field"}:
            return True
    # `[[]] * 3` and friends - a repeated mutable is still a fresh container.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _is_mutable_value(node.left) or _is_mutable_value(node.right)
    return False


def _is_definition_time_default(node: ast.expr | None) -> bool:
    """Return whether a parameter default is built once when the `def` is executed.

    Looser than `_is_mutable_value` on purpose: in a default position *any* call is
    evaluated once at definition time, which is exactly the `default_from_call` bug
    (`def f(buf=make_buffer())`). Outside a default position a bare call says nothing
    about lifetime, so the narrow predicate is still the right one there.
    """
    return isinstance(node, ast.Call) or _is_mutable_value(node)


def _has_mutable_default(tree: ast.AST) -> bool:
    """Return whether any function parameter default is evaluated once at definition time."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults: list[ast.expr | None] = [*node.args.defaults, *node.args.kw_defaults]
            if any(_is_definition_time_default(d) for d in defaults):
                return True
        # A dataclass field default is the class-body equivalent of a parameter default.
        if isinstance(node, ast.AnnAssign) and _is_mutable_value(node.value):
            return True
    return False


def _has_shared_container(tree: ast.AST) -> bool:
    """Return whether a mutable object is owned more broadly than one instance."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and _is_mutable_value(stmt.value):
                    return True
                if isinstance(stmt, ast.AnnAssign) and _is_mutable_value(stmt.value):
                    return True
        # A module-level mutable mutated from inside a function is the same failure.
        if isinstance(node, ast.Global):
            return True
    if isinstance(tree, ast.Module):
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and _is_mutable_value(stmt.value):
                return True
    return False


def _has_assignment_in_loop(tree: ast.AST) -> bool:
    """Return whether a name is rebound inside a loop body."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            for inner in ast.walk(node):
                if inner is node:
                    continue
                if isinstance(inner, (ast.Assign, ast.AnnAssign)):
                    return True
    return False


def _has_alias(tree: ast.AST) -> bool:
    """Return whether two names are bound to one object without a deep copy.

    The loosest of the four validators: binding a name to another name, an attribute, or
    a subscript is what aliasing *is*, and those shapes also appear incidentally in other
    concepts' programs. Measured against the eval set it accepts all 9 aliasing scenarios
    and leaks 4 of 27 from other concepts. That asymmetry is deliberate - a false
    rejection discards a good example, a false acceptance only means a row landed in a
    neighbouring in-scope cell.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
            return True
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
            return True
        if isinstance(value, ast.Call):
            func = value.func
            if isinstance(func, ast.Attribute) and func.attr == "copy":
                return True
            if isinstance(func, ast.Name) and func.id in {"list", "dict", "set"}:
                return True
    return False


_CONCEPT_VALIDATORS = {
    LifetimeConcept.CREATION: _has_mutable_default,
    LifetimeConcept.OWNERSHIP: _has_shared_container,
    LifetimeConcept.RESET: _has_assignment_in_loop,
    LifetimeConcept.ALIASING: _has_alias,
}


def validate_code(code: str, concept: LifetimeConcept) -> str | None:
    """Check that a program parses and structurally exhibits its declared concept.

    Parses only - the code is never executed, so an unreviewed generated program cannot
    do anything.

    Args:
        code: The candidate's Python source.
        concept: The lifetime concept the author claims this program demonstrates.

    Returns:
        None if the code is valid, otherwise a human-readable failure detail.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"does not parse: {exc.msg}"
    if not _CONCEPT_VALIDATORS[concept](tree):
        return f"no AST pattern matching concept '{concept}'"
    return None


# --- The gate -----------------------------------------------------------------


def find_contamination(candidate: Candidate, eval_set: Sequence[Scenario]) -> str | None:
    """Check a candidate against the frozen eval set.

    The eval set is the primary check against overfitting, so any overlap has to be
    caught here rather than discovered as an inflated score later.

    Args:
        candidate: The authored example.
        eval_set: Scenarios loaded from `data/scenarios.jsonl`.

    Returns:
        None if clean, otherwise which eval scenario it collided with and how.
    """
    code = normalize_code(candidate.code)
    message = candidate.student_message.strip().lower()
    for scenario in eval_set:
        if normalize_code(scenario.code) == code:
            return f"identical code to {scenario.id}"
        if scenario.student_message.strip().lower() == message:
            return f"identical student_message to {scenario.id}"
        similarity = jaccard(candidate.code, scenario.code)
        if similarity > CONTAMINATION_THRESHOLD:
            return f"{similarity:.0%} shingle overlap with {scenario.id}"
    return None


def _as_scenario(candidate: Candidate, example_id: str) -> Scenario:
    """Project a candidate onto the Scenario model the eval harness already speaks."""
    return Scenario(
        id=example_id,
        category=candidate.category,
        language="python",
        code=candidate.code,
        student_message=candidate.student_message,
        bug=candidate.bug,
        bug_region=candidate.bug_region,
        lifetime_concept=candidate.lifetime_concept,
        expected_question_focus=candidate.expected_question_focus,
        forbidden_fix_tokens=candidate.forbidden_fix_tokens,
    )


def screen(
    candidate: Candidate,
    eval_set: Sequence[Scenario],
    seen_code: set[str],
    seen_messages: set[str],
    require_near_miss: bool = True,
) -> tuple[MechanicalCheck, RejectReason | None, str]:
    """Run every free gate against one candidate, cheapest first.

    Args:
        candidate: The authored example.
        eval_set: The frozen eval scenarios to check contamination against.
        seen_code: Normalised code of already-accepted examples; mutated on acceptance.
        seen_messages: Lowercased student messages already accepted; mutated on acceptance.
        require_near_miss: Whether a near-miss must be present and must fail the spec.

    Returns:
        The response's mechanical check, the reject reason (None if accepted), and a
        detail string explaining the rejection.
    """
    if not MIN_CODE_CHARS <= len(candidate.code) <= MAX_CODE_CHARS:
        return (
            run_mechanical_check(candidate.response, _as_scenario(candidate, "pending")),
            RejectReason.CODE_LENGTH,
            f"{len(candidate.code)} chars outside [{MIN_CODE_CHARS}, {MAX_CODE_CHARS}]",
        )

    scenario = _as_scenario(candidate, "pending")
    check = run_mechanical_check(candidate.response, scenario)

    if normalize_code(candidate.bug_region) not in normalize_code(candidate.code):
        return check, RejectReason.BUG_REGION_NOT_IN_CODE, repr(candidate.bug_region)

    # Forbidden tokens describe the *correction*. If one is contained in the region the
    # tutor is required to quote, then quoting the bug - which the spec explicitly allows
    # - trips `stated_fix`, and a correct response is unfilterable. Caught here so it
    # reads as the authoring error it is rather than as a mysterious spec failure.
    region = normalize_token(candidate.bug_region)
    trapped = [t for t in candidate.forbidden_fix_tokens if normalize_token(t) in region]
    if trapped:
        return check, RejectReason.FORBIDDEN_TOKEN_IN_BUG_REGION, f"{trapped} inside bug_region"

    code_error = validate_code(candidate.code, candidate.lifetime_concept)
    if code_error is not None:
        reason = (
            RejectReason.UNPARSEABLE_CODE
            if code_error.startswith("does not parse")
            else RejectReason.CONCEPT_NOT_IN_AST
        )
        return check, reason, code_error

    collision = find_contamination(candidate, eval_set)
    if collision is not None:
        return check, RejectReason.CONTAMINATES_EVAL_SET, collision

    normalized = normalize_code(candidate.code)
    if normalized in seen_code:
        return check, RejectReason.DUPLICATE_CODE, "already in pool"
    message = candidate.student_message.strip().lower()
    if message in seen_messages:
        return check, RejectReason.DUPLICATE_STUDENT_MESSAGE, "already in pool"

    if not check.passed:
        return check, RejectReason.RESPONSE_FAILS_SPEC, _describe_check(check)

    if require_near_miss:
        if candidate.near_miss is None:
            return check, RejectReason.NEAR_MISS_PASSES_SPEC, "no near_miss supplied"
        # The negative has to be a genuine violation, otherwise it is not a usable
        # preference pair and it means the gate has stopped discriminating.
        near = run_mechanical_check(candidate.near_miss, scenario)
        if near.passed:
            return check, RejectReason.NEAR_MISS_PASSES_SPEC, "near_miss satisfies the spec"

    seen_code.add(normalized)
    seen_messages.add(message)
    return check, None, ""


def _describe_check(check: MechanicalCheck) -> str:
    """Summarise which mechanical clauses a response broke."""
    broken: list[str] = []
    if check.emitted_code:
        broken.append("emitted_code")
    # `is True` / `is False` because both clauses are tri-state: None means the scenario
    # supplied nothing to check against. Authoring candidates always supply both, so None
    # should not occur here - but reading it as a violation is the exact bug the tri-state
    # change introduces everywhere it is read with plain truthiness.
    if check.stated_fix is True:
        broken.append("stated_fix")
    if check.question_count != 1:
        broken.append(f"question_count={check.question_count}")
    if check.has_localization is False:
        broken.append("no_localization")
    if check.possible_compound_question:
        broken.append("possible_compound_question")
    return ", ".join(broken) or "unknown"


# --- Rank assignment ----------------------------------------------------------


def _cell_jitter(cell: Cell) -> str:
    """Return a stable, category-mixing tiebreaker for cells at the same position.

    Ordering ties by `cell.slug` looks harmless but sorts every `adversarial-*` cell
    ahead of every `clean-*` one, so when cells hold equal counts the first half of the
    pool is 100% adversarial and the small curve points test a distribution the large
    ones never see. Hashing the slug interleaves categories and concepts instead.
    `hashlib` rather than `hash()` because the built-in is salted per process, which
    would make rank assignment - and therefore the curve manifests - irreproducible.
    """
    return hashlib.blake2b(cell.slug.encode(), digest_size=8).hexdigest()


def assign_ranks(
    accepted: Sequence[tuple[Candidate, GenerationProvenance]],
) -> list[TrainingExample]:
    """Order accepted candidates so that every prefix of the pool stays balanced.

    Each example is keyed by its fractional position within its own cell, so a prefix of
    length K contains roughly K/N of every cell. That keeps concept, code-shape, and the
    2:1 clean:adversarial ratio balanced at *every* curve point, and makes the curve
    subsets automatically nested (62 subset of 125 subset of 250 subset of 500). Without
    this, the curve would confound dataset size with which examples happened to be drawn.

    Args:
        accepted: Candidates that passed the gate, paired with their provenance.

    Returns:
        Training examples with `rank` assigned and ids stamped, ordered by rank.
    """
    by_cell: dict[Cell, list[tuple[Candidate, GenerationProvenance]]] = {}
    for candidate, provenance in accepted:
        by_cell.setdefault(candidate.cell, []).append((candidate, provenance))

    # (position, jitter, slug, index): jitter only breaks ties in the ordering, while
    # slug is what names the example - keeping them separate so ids stay human-readable.
    keyed: list[tuple[float, str, str, int, Candidate, GenerationProvenance]] = []
    for cell, rows in by_cell.items():
        for index, (candidate, provenance) in enumerate(rows):
            # Midpoint of this example's slice of its cell, so cells of different sizes
            # interleave proportionally rather than the largest cell dominating the head.
            position = (index + 0.5) / len(rows)
            keyed.append((position, _cell_jitter(cell), cell.slug, index, candidate, provenance))
    keyed.sort(key=lambda row: (row[0], row[1], row[3]))

    examples: list[TrainingExample] = []
    for rank, (_, _jitter, slug, index, candidate, provenance) in enumerate(keyed):
        examples.append(
            TrainingExample(
                scenario=_as_scenario(candidate, f"{slug}-{index:03d}"),
                rank=rank,
                code_shape=candidate.code_shape,
                seed_domain=candidate.seed_domain,
                pressure=candidate.pressure,
                response=candidate.response,
                near_miss=candidate.near_miss,
                provenance=provenance,
            )
        )
    return examples


# --- Curve manifests ----------------------------------------------------------


def curve_points(total: int, halvings: int = 3) -> list[int]:
    """Return log-spaced curve sizes N, N/2, N/4, N/8 for a pool of `total`.

    Args:
        total: Size of the full pool.
        halvings: How many times to halve below the full pool.

    Returns:
        Sizes in ascending order, smallest first, with any zero sizes dropped.
    """
    sizes = {total // (2**step) for step in range(halvings + 1)}
    return sorted(size for size in sizes if size > 0)


def write_curve_manifests(
    examples: Sequence[TrainingExample], out_dir: Path, halvings: int = 3
) -> dict[int, Path]:
    """Write one id manifest per curve point.

    Manifests hold ids rather than copies of the data, so the points cannot drift apart
    from the pool and adding a point costs nothing.

    Args:
        examples: The ranked pool.
        out_dir: Directory to write `curve/n-<size>.txt` into.
        halvings: How many halvings below the full pool to emit.

    Returns:
        Manifest path per curve size.
    """
    curve_dir = out_dir / "curve"
    curve_dir.mkdir(parents=True, exist_ok=True)
    # Clear first: the pool grows between ingests, so manifests written at an earlier
    # size would otherwise linger and a training run could silently pick up a stale
    # curve point that no longer matches the pool it claims a prefix of.
    for stale in curve_dir.glob("n-*.txt"):
        stale.unlink()
    ordered = sorted(examples, key=lambda e: e.rank)
    written: dict[int, Path] = {}
    for size in curve_points(len(ordered), halvings):
        path = curve_dir / f"n-{size}.txt"
        path.write_text("\n".join(e.scenario.id for e in ordered[:size]) + "\n")
        written[size] = path
    return written


# --- Export -------------------------------------------------------------------


def to_chat_messages(
    example: TrainingExample, system_prompt: str = TRAIN_SYSTEM_PROMPT
) -> list[dict[str, str]]:
    """Render one example in the chat format TRL's SFT path consumes.

    Derived at export time rather than stored, so changing the training prompt is a
    re-export instead of a regeneration.

    Args:
        example: The accepted training row.
        system_prompt: The one-line system prompt to train against.

    Returns:
        A system/user/assistant message list.
    """
    scenario = example.scenario
    user = f"```{scenario.language}\n{scenario.code}\n```\n{scenario.student_message}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": example.response},
    ]


# --- IO -----------------------------------------------------------------------


def load_candidates(path: Path) -> list[Candidate]:
    """Load authored candidates from a JSONL batch file.

    Args:
        path: Path to the batch file.

    Returns:
        Every candidate in the file, in file order.

    Raises:
        ValueError: If the file contains no candidates.
    """
    candidates = [
        Candidate.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not candidates:
        raise ValueError(f"no candidates in {path}")
    return candidates


def load_pool(path: Path) -> list[TrainingExample]:
    """Load an existing pool, or return an empty list if it does not exist yet."""
    if not path.exists():
        return []
    return [
        TrainingExample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_jsonl(rows: Iterable[BaseModel], path: Path) -> None:
    """Write pydantic rows as JSONL, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row.model_dump_json()}\n" for row in rows))


def append_jsonl(rows: Iterable[BaseModel], path: Path) -> None:
    """Append pydantic rows to a JSONL file, creating it if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(f"{row.model_dump_json()}\n")


def export_sft(examples: Sequence[TrainingExample], path: Path) -> None:
    """Write the TRL-ready chat JSONL for a set of examples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"messages": to_chat_messages(e)}) + "\n"
            for e in sorted(examples, key=lambda e: e.rank)
        )
    )
