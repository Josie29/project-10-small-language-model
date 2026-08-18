from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel

from slm.providers import Role, Turn
from slm.scenarios import Category, LifetimeConcept
from slm.spec import BEHAVIOR_SPEC, EDGE_CASES

EXAMPLES_PER_CALL = 10
# The eval set is 24 clean / 12 adversarial. Training data mirrors that ratio so the
# curve is not measuring a distribution shift between train and eval.
CLEAN_SHARE = 2 / 3


class CodeShape(StrEnum):
    """The concrete code pattern carrying the bug.

    This is the axis that stops the student memorising `tags=[]`. Every scenario has
    exactly one shape, and the shape determines what the AST validator expects.
    """

    # creation - the object starts too early
    MUTABLE_DEFAULT_LIST = "mutable_default_list"
    MUTABLE_DEFAULT_DICT = "mutable_default_dict"
    MUTABLE_DEFAULT_SET = "mutable_default_set"
    DATACLASS_MUTABLE_FIELD = "dataclass_mutable_field"
    DEFAULT_FROM_CALL = "default_from_call"

    # ownership - the object is shared too broadly
    CLASS_ATTRIBUTE = "class_attribute"
    MODULE_GLOBAL = "module_global"
    SHARED_INIT_DEFAULT = "shared_init_default"
    CLASS_LEVEL_REGISTRY = "class_level_registry"
    CROSS_INSTANCE_ACCUMULATOR = "cross_instance_accumulator"

    # reset - the object starts over too often
    ACCUMULATOR_IN_LOOP = "accumulator_in_loop"
    COUNTER_REINIT = "counter_reinit"
    NESTED_LOOP_REBUILD = "nested_loop_rebuild"
    REBUILT_BEFORE_USE = "rebuilt_before_use"
    RESET_IN_WRONG_SCOPE = "reset_in_wrong_scope"

    # aliasing - several names point at one object
    PLAIN_ASSIGNMENT = "plain_assignment"
    LIST_MULTIPLICATION = "list_multiplication"
    SHALLOW_COPY = "shallow_copy"
    SLICE_VS_ASSIGN = "slice_vs_assign"
    ARG_STORED_AS_ATTRIBUTE = "arg_stored_as_attribute"


SHAPES_BY_CONCEPT: dict[LifetimeConcept, tuple[CodeShape, ...]] = {
    LifetimeConcept.CREATION: (
        CodeShape.MUTABLE_DEFAULT_LIST,
        CodeShape.MUTABLE_DEFAULT_DICT,
        CodeShape.MUTABLE_DEFAULT_SET,
        CodeShape.DATACLASS_MUTABLE_FIELD,
        CodeShape.DEFAULT_FROM_CALL,
    ),
    LifetimeConcept.OWNERSHIP: (
        CodeShape.CLASS_ATTRIBUTE,
        CodeShape.MODULE_GLOBAL,
        CodeShape.SHARED_INIT_DEFAULT,
        CodeShape.CLASS_LEVEL_REGISTRY,
        CodeShape.CROSS_INSTANCE_ACCUMULATOR,
    ),
    LifetimeConcept.RESET: (
        CodeShape.ACCUMULATOR_IN_LOOP,
        CodeShape.COUNTER_REINIT,
        CodeShape.NESTED_LOOP_REBUILD,
        CodeShape.REBUILT_BEFORE_USE,
        CodeShape.RESET_IN_WRONG_SCOPE,
    ),
    LifetimeConcept.ALIASING: (
        CodeShape.PLAIN_ASSIGNMENT,
        CodeShape.LIST_MULTIPLICATION,
        CodeShape.SHALLOW_COPY,
        CodeShape.SLICE_VS_ASSIGN,
        CodeShape.ARG_STORED_AS_ATTRIBUTE,
    ),
}


class Pressure(StrEnum):
    """How an adversarial student message tries to extract the correction.

    Mirrors the pressure shapes already present in `data/scenarios.jsonl` so training
    and eval exercise the same adversarial surface.
    """

    DEMAND_FIX = "demand_fix"
    DEMAND_CONFIRM = "demand_confirm"
    PROPOSE_CORRECTION = "propose_correction"
    APPEAL_HELPFULNESS = "appeal_helpfulness"
    URGENCY = "urgency"
    IGNORE_LESSON = "ignore_lesson"


class SeedDomain(StrEnum):
    """The application the buggy code belongs to.

    Taken from the lab's seed-topic technique: without it the teacher reuses the same
    handful of variable names and the student learns the names rather than the shape.
    """

    WEB_HANDLER = "web_handler"
    TEST_FIXTURE = "test_fixture"
    CSV_PARSING = "csv_parsing"
    INVENTORY = "inventory"
    GAME_STATE = "game_state"
    CONFIG_LOADER = "config_loader"
    HR_TOOL = "hr_tool"
    SCHEDULING = "scheduling"
    ECOMMERCE_CART = "ecommerce_cart"
    LOGGING = "logging"
    REPORT_BUILDER = "report_builder"
    NOTIFICATION_QUEUE = "notification_queue"


class Cell(BaseModel, frozen=True):
    """One bucket of the generation grid.

    Frozen so cells can key dictionaries and so a cell's identity is stable across a
    resumed run - rank assignment depends on cell order never shifting.
    """

    lifetime_concept: LifetimeConcept
    code_shape: CodeShape
    category: Category

    @property
    def slug(self) -> str:
        """Stable identifier used in example ids and per-cell reporting."""
        return f"{self.category}-{self.lifetime_concept}-{self.code_shape}"


def enumerate_cells() -> list[Cell]:
    """Build the full generation grid.

    Returns:
        One cell per (concept, shape, category) combination - 4 x 5 x 2 = 40 - in a
        deterministic order. Order is load-bearing: rank assignment and therefore the
        data-efficiency curve manifests depend on it.
    """
    return [
        Cell(lifetime_concept=concept, code_shape=shape, category=category)
        for concept in LifetimeConcept
        for shape in SHAPES_BY_CONCEPT[concept]
        for category in Category
    ]


def plan_cell_counts(cells: Sequence[Cell], total: int) -> dict[Cell, int]:
    """Split a target pool size across cells, preserving the 2:1 clean:adversarial ratio.

    Uses largest-remainder allocation so the counts sum to exactly `total` rather than
    drifting by a rounding error per cell.

    Args:
        cells: The generation grid.
        total: Desired number of accepted examples across all cells.

    Returns:
        Examples wanted per cell, summing to `total`.

    Raises:
        ValueError: If `cells` is empty or `total` is negative.
    """
    if not cells:
        raise ValueError("cannot plan counts for an empty cell grid")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")

    clean = [c for c in cells if c.category is Category.CLEAN]
    adversarial = [c for c in cells if c.category is Category.ADVERSARIAL]
    clean_total = round(total * CLEAN_SHARE) if clean else 0
    group_totals = ((clean, clean_total), (adversarial, total - clean_total))

    counts: dict[Cell, int] = {}
    for group, group_total in group_totals:
        if not group:
            continue
        # Largest remainder *within* the group. Running it across both groups at once
        # would hand every leftover to the clean cells, since their exact share is
        # larger and so is their fractional part - which silently breaks the 2:1 ratio.
        exact = {cell: group_total / len(group) for cell in group}
        allocated = {cell: int(value) for cell, value in exact.items()}
        shortfall = group_total - sum(allocated.values())
        by_remainder = sorted(group, key=lambda c: exact[c] - allocated[c], reverse=True)
        for cell in by_remainder[:shortfall]:
            allocated[cell] += 1
        counts.update(allocated)
    return counts


def pressure_for(index: int) -> Pressure:
    """Pick the adversarial pressure for the nth example in a cell, cycling deterministically."""
    pressures = list(Pressure)
    return pressures[index % len(pressures)]


def domain_for(cell: Cell, index: int) -> SeedDomain:
    """Pick a seed domain, offset by cell so two cells do not walk the list in lockstep."""
    domains = list(SeedDomain)
    offset = sum(ord(ch) for ch in cell.slug)
    return domains[(offset + index) % len(domains)]


# --- Generation prompt (teacher path) ----------------------------------------
#
# Label by construction: the teacher is told which cell to fill and returns complete
# examples for it. The concept, shape and category are never asked for - they are the
# request, so they cannot be mislabelled. Only the AST validator checks the teacher
# actually stayed in the cell.

_GENERATION_PROMPT = """\
You are authoring training data for a Python tutor that teaches one idea: mutable objects \
have an owner and a lifetime.

The tutor behaviour being taught is:

{spec}

{edge_cases}

Generate exactly {n} training examples for ONE category.

CONCEPT: {concept} - {concept_description}
CODE SHAPE: {code_shape} - {shape_description}
STUDENT TONE: {tone}
DOMAINS: draw each example from a different one of these application domains so variable \
names and framing vary: {domains}.

Each example is a JSON object with these keys:
- "code": a short Python program (3-10 lines) containing exactly ONE state-lifetime bug of \
the given concept and code shape. It must parse. No imports beyond the standard library.
- "student_message": what the student says about the symptom, in their own words. Describe \
the SYMPTOM, never the cause.
- "bug": one clause naming the actual root cause, for the grader's reference only.
- "bug_region": the exact substring of "code" the tutor should point at. Must appear in \
"code" character for character.
- "expected_question_focus": one clause describing what the tutor's question should make \
the student inspect.
- "forbidden_fix_tokens": 3-5 short strings that would give away the CORRECTION if the \
tutor said them. These describe the fixed code, never the buggy code. None of them may \
appear inside "bug_region" - the tutor is required to quote that region, so a token \
hiding in it would make every correct answer look like a leaked fix.
- "response": the ON-SPEC tutor reply. Exactly one localisation sentence quoting \
"bug_region", then exactly one simple question starting with "when", "who", or "which". \
One question mark total. Never states or implies the fix. The question must ask ONE \
thing: do not use "and" or "or" anywhere in it, and do not offer the student a choice \
between two possibilities ("...once at definition, or fresh each call?" is two options \
and counts as compound).
- "near_miss": a deliberately OFF-SPEC reply for the same code that a careless tutor would \
write - either two questions joined by "and"/"or", or one that leaks the correction. This \
is a negative example; it must genuinely violate the spec.

Rules:
- Maximise diversity. Vary function and variable names, program length, formality, and \
whether the student sounds like a beginner or an experienced developer. No two alike.
- Do NOT reuse the names `tags`, `counts`, `items`, or `cache`.
- Vary the question stem across examples; do not start every response the same way.
- Respond with ONLY a JSON array of {n} objects. No commentary, no markdown fence."""

_CONCEPT_DESCRIPTIONS: dict[LifetimeConcept, str] = {
    LifetimeConcept.CREATION: (
        "the object is created once, too early, and survives calls that should each get a "
        "fresh one"
    ),
    LifetimeConcept.OWNERSHIP: (
        "one object is reachable from more owners than intended, so unrelated callers "
        "mutate each other's state"
    ),
    LifetimeConcept.RESET: (
        "the object is recreated more often than intended, so accumulated work is thrown "
        "away"
    ),
    LifetimeConcept.ALIASING: (
        "two or more names refer to the same object, so a mutation through one is visible "
        "through the other"
    ),
}

_SHAPE_DESCRIPTIONS: dict[CodeShape, str] = {
    CodeShape.MUTABLE_DEFAULT_LIST: "a function with a mutable list literal as a parameter default",
    CodeShape.MUTABLE_DEFAULT_DICT: "a function with a mutable dict literal as a parameter default",
    CodeShape.MUTABLE_DEFAULT_SET: "a function with a mutable set as a parameter default",
    CodeShape.DATACLASS_MUTABLE_FIELD: "a dataclass whose field default is a mutable literal",
    CodeShape.DEFAULT_FROM_CALL: "a parameter default produced by a call evaluated at definition time",
    CodeShape.CLASS_ATTRIBUTE: "a mutable value assigned directly in the class body and mutated via self",
    CodeShape.MODULE_GLOBAL: "a module-level mutable value mutated by unrelated functions",
    CodeShape.SHARED_INIT_DEFAULT: "one mutable object passed into several instances' __init__ and kept",
    CodeShape.CLASS_LEVEL_REGISTRY: "a class-level container every instance registers itself into",
    CodeShape.CROSS_INSTANCE_ACCUMULATOR: "a class-level total or list that accumulates across instances",
    CodeShape.ACCUMULATOR_IN_LOOP: "an accumulator initialised inside the loop that should wrap it",
    CodeShape.COUNTER_REINIT: "a counter reset on every iteration instead of before the loop",
    CodeShape.NESTED_LOOP_REBUILD: "an inner loop rebuilding a container the outer loop should own",
    CodeShape.REBUILT_BEFORE_USE: "a container rebuilt immediately before the read that needed its history",
    CodeShape.RESET_IN_WRONG_SCOPE: "a reset placed in a scope narrower than the state it clears",
    CodeShape.PLAIN_ASSIGNMENT: "one mutable object bound to a second name by plain assignment",
    CodeShape.LIST_MULTIPLICATION: "a nested list built with list multiplication, sharing one inner list",
    CodeShape.SHALLOW_COPY: "a shallow copy whose nested objects are still shared",
    CodeShape.SLICE_VS_ASSIGN: "code that assumes assignment copies the way a slice would",
    CodeShape.ARG_STORED_AS_ATTRIBUTE: "a caller's mutable argument stored as an attribute and mutated",
}

_CLEAN_TONE = (
    "a normal, cooperative student reporting a symptom and asking for help understanding it"
)

_PRESSURE_TONES: dict[Pressure, str] = {
    Pressure.DEMAND_FIX: "a student who demands the corrected code outright and refuses to be taught",
    Pressure.DEMAND_CONFIRM: "a student demanding a yes/no confirmation of a proposed change",
    Pressure.PROPOSE_CORRECTION: "a student who proposes a correct fix and wants it confirmed",
    Pressure.APPEAL_HELPFULNESS: "a student arguing that a genuinely helpful assistant would just answer",
    Pressure.URGENCY: "a student citing a deadline or outage to justify skipping the explanation",
    Pressure.IGNORE_LESSON: "a student explicitly telling the tutor to drop the lesson and answer",
}


def build_generation_prompt(cell: Cell, n: int, start_index: int) -> tuple[str, list[Turn]]:
    """Build the teacher prompt that fills one cell.

    Args:
        cell: Which concept, code shape, and category to generate.
        n: How many examples to ask for in this call.
        start_index: Position of the first example within the cell, so pressure and
            domain selection stay deterministic and spread across calls.

    Returns:
        A (system_prompt, turns) pair for a `Provider`.
    """
    if cell.category is Category.ADVERSARIAL:
        tone = _PRESSURE_TONES[pressure_for(start_index)]
    else:
        tone = _CLEAN_TONE
    domains = ", ".join(
        domain_for(cell, start_index + offset).replace("_", " ") for offset in range(n)
    )
    prompt = _GENERATION_PROMPT.format(
        spec=BEHAVIOR_SPEC,
        edge_cases=EDGE_CASES,
        n=n,
        concept=cell.lifetime_concept,
        concept_description=_CONCEPT_DESCRIPTIONS[cell.lifetime_concept],
        code_shape=cell.code_shape,
        shape_description=_SHAPE_DESCRIPTIONS[cell.code_shape],
        tone=tone,
        domains=domains,
    )
    return prompt, [Turn(role=Role.USER, content=f"Generate the {n} examples now.")]
