from __future__ import annotations

from slm.checks import MechanicalCheck, canned_response, run_mechanical_check
from slm.config import (
    DEFAULT_CONCURRENCY,
    JUDGE,
    MAX_TOKENS,
    Backend,
    Family,
    ModelSpec,
    load_env_file,
)
from slm.judge import JudgeVerdict, build_judge_prompt, judge_response
from slm.prompting import Strategy, build_prompt
from slm.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    Provider,
    Role,
    Turn,
    build_client,
)
from slm.reporting import (
    CellResult,
    Trial,
    aggregate,
    render_degraded_note,
    render_table,
    write_results,
)
from slm.scenarios import (
    Category,
    RubricCoverage,
    Scenario,
    load_scenarios,
    load_scenarios_with_coverage,
    require_authored,
    rubric_coverage,
    stratified_sample,
)
from slm.spec import BEHAVIOR_SPEC, EDGE_CASES, JUDGE_RUBRIC

__all__ = [
    "AnthropicProvider",
    "BEHAVIOR_SPEC",
    "Backend",
    "CellResult",
    "Category",
    "DEFAULT_CONCURRENCY",
    "EDGE_CASES",
    "Family",
    "JUDGE",
    "JUDGE_RUBRIC",
    "JudgeVerdict",
    "MAX_TOKENS",
    "MechanicalCheck",
    "ModelSpec",
    "OpenAICompatibleProvider",
    "Provider",
    "RubricCoverage",
    "Role",
    "Scenario",
    "Strategy",
    "Trial",
    "Turn",
    "aggregate",
    "build_client",
    "build_judge_prompt",
    "build_prompt",
    "canned_response",
    "judge_response",
    "load_env_file",
    "load_scenarios",
    "load_scenarios_with_coverage",
    "render_degraded_note",
    "render_table",
    "require_authored",
    "rubric_coverage",
    "run_mechanical_check",
    "stratified_sample",
    "write_results",
]
