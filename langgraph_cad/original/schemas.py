"""Shared CAD contract.

Each agent reads only the previous artifact it needs and writes one state key.
State stores model_dump() dicts. Names are the join keys:

- part name: CadSpec -> GeometryPlan
- parameter name: RequirementSpec -> CadSpec -> PlanStep.uses
- step id: GeometryPlan -> RepairDecision
- operation: GeometryPlan -> KnowledgePack
- object name: GeneratedScript -> RunResult
"""

from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator

MAX_REPAIR_ATTEMPTS = 3

FactStatus = Literal["given", "unknown"]
OperationName = Literal[
    "sketch",
    "pad",
    "pocket",
    "revolve",
    "sweep",
    "loft",
    "thickness",
    "fillet",
    "chamfer",
    "boolean_cut",
    "boolean_fuse",
    "boolean_common",
    "mirror",
    "linear_pattern",
    "polar_pattern",
    "placement",
]
MetricName = Literal[
    "document_saved",
    "shape_valid",
    "solid_count",
    "volume_positive",
    "bbox",
    "parameter",
    "feature_present",
    "interface",
]
RepairAction = Literal["accept", "revise_plan", "revise_code", "stop"]


class Parameter(BaseModel):
    name: str = Field(description="Dimension name, 1-3 words")
    value: float | None = Field(description="Stated number, otherwise null")
    unit: str | None = Field(description="Stated unit, otherwise null")
    status: FactStatus


class RequirementSpec(BaseModel):
    """normalize_request writes state['requirements']."""

    goal: str = Field(description="What to model, one sentence")
    unit_system: str | None = Field(description="Named unit system, otherwise null")
    parameters: List[Parameter] = Field(
        description="Every size needed to model the object. Missing numbers use null value, null unit, and status unknown. Never empty."
    )
    constraints: List[str] = Field(description="User-stated limits; empty if none")
    ambiguities: List[str] = Field(
        description="One item per unknown parameter. Empty only when every parameter is given."
    )


class NormalizeInput(BaseModel):
    request: str


class PartSpec(BaseModel):
    name: str = Field(description="Distinct solid name")
    quantity: int = Field(description="Stated count, otherwise 1")
    shape: str = Field(description="User's form in one sentence, not a parameter name")
    features: List[str] = Field(description="Identity features such as open rim or wall. No sizes or primitive names.")


class SpecParameter(Parameter):
    part: str = Field(description="Owning part name, or assembly")


class InterfaceSpec(BaseModel):
    a: str = Field(description="First part name")
    b: str = Field(description="Second part name")
    relation: str = Field(description="Mate in a few words")


class CadSpec(BaseModel):
    """specify_cad writes state['cad_spec']."""

    parts: List[PartSpec]
    parameters: List[SpecParameter]
    constraints: List[str] = Field(description="Copied limits; empty if none")
    interfaces: List[InterfaceSpec] = Field(description="Mates between parts; empty if one part")

    @model_validator(mode="after")
    def references_match_parts(self) -> "CadSpec":
        if not self.parts:
            raise ValueError("at least one part is required")
        names = {part.name for part in self.parts}
        allowed = names | {"assembly"}
        for parameter in self.parameters:
            if parameter.part not in allowed:
                raise ValueError(f"parameter {parameter.name} has unknown part {parameter.part}")
        for interface in self.interfaces:
            if interface.a not in names or interface.b not in names:
                raise ValueError("interface references an unknown part")
        return self


class SpecifyInput(BaseModel):
    requirements: RequirementSpec


class StepUse(BaseModel):
    source: Literal["parameter", "step"]
    ref: str = Field(description="Exact name from cad_spec.parameters, or an earlier step id")


class PlanStep(BaseModel):
    id: str = Field(description="Unique step id")
    part: str = Field(description="CadSpec part name")
    operation: OperationName
    depends_on: List[str] = Field(description="Earlier step ids")
    uses: List[StepUse] = Field(description="Parameters or earlier steps consumed")
    intent: str = Field(description="What this operation creates, one sentence")


class GeometryPlan(BaseModel):
    """plan_geometry writes state['geometry_plan']."""

    steps: List[PlanStep]

    @model_validator(mode="after")
    def steps_are_ordered(self) -> "GeometryPlan":
        if not self.steps:
            raise ValueError("plan needs at least one step")
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"duplicate step id {step.id}")
            if any(dep not in seen for dep in step.depends_on):
                raise ValueError(f"{step.id} depends on a later or unknown step")
            for use in step.uses:
                if use.source == "step" and use.ref not in seen:
                    raise ValueError(f"{step.id} uses unknown step {use.ref}")
            seen.add(step.id)
        return self


class PlanInput(BaseModel):
    cad_spec: CadSpec


class OperationRequest(BaseModel):
    operation: OperationName
    intent: str = Field(description="One sentence copied from the plan steps")


class RetrieveInput(BaseModel):
    """Derived from GeometryPlan. One request per distinct operation."""

    requests: List[OperationRequest]


class KnowledgeSnippet(BaseModel):
    operation: OperationName
    source: str = Field(description="Documentation file name")
    summary: str = Field(description="What the API call does, one sentence")
    code: str = Field(description="Shortest relevant FreeCAD example")


class KnowledgePack(BaseModel):
    """retrieve_knowledge writes state['knowledge']."""

    snippets: List[KnowledgeSnippet]


class GenerateInput(BaseModel):
    plan: GeometryPlan
    knowledge: KnowledgePack


class GeneratedScript(BaseModel):
    """generate_code writes state['script']."""

    code: str
    object_names: List[str] = Field(description="FreeCAD object names the script creates")


class RunInput(BaseModel):
    script: GeneratedScript


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float


class ShapeMetrics(BaseModel):
    object_name: str
    is_null: bool
    is_valid: bool
    solid_count: int
    volume: float | None
    bbox: BoundingBox | None


class RunResult(BaseModel):
    """run_freecad writes state['run_result']."""

    ok: bool
    return_code: int
    fcstd_path: str | None
    error: str | None = Field(description="Traceback tail, empty on success")
    metrics: List[ShapeMetrics]
    screenshots: List[str] = Field(description="Saved view paths; empty if none")


class ValidateInput(BaseModel):
    cad_spec: CadSpec
    plan: GeometryPlan
    run: RunResult


class CheckResult(BaseModel):
    metric: MetricName
    target: str = Field(description="Object name, part name, parameter name, or step id")
    passed: bool
    detail: str = Field(description="One short sentence")


class ValidationReport(BaseModel):
    """validate_model writes state['validation']."""

    passed: bool
    checks: List[CheckResult]


class RepairInput(BaseModel):
    plan: GeometryPlan
    script: GeneratedScript
    run: RunResult
    validation: ValidationReport
    attempt: int


class RepairDecision(BaseModel):
    """repair_model writes state['repair']."""

    action: RepairAction
    reason: str = Field(description="One short sentence")
    step_ids: List[str] = Field(description="Plan steps to redo; empty unless revising the plan")

    @model_validator(mode="after")
    def targets_match_action(self) -> "RepairDecision":
        if self.action in ("accept", "stop", "revise_code") and self.step_ids:
            raise ValueError("step_ids are only used when revising the plan")
        if self.action == "revise_plan" and not self.step_ids:
            raise ValueError("revise_plan needs step ids")
        return self


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    requirements: dict | None
    cad_spec: dict | None
    geometry_plan: dict | None
    knowledge: dict | None
    script: dict | None
    run_result: dict | None
    validation: dict | None
    repair: dict | None
    attempt: int
    # Current retrieval and codegen nodes. Replaced by knowledge and script.
    selected_files: List[dict[str, str]] | None
    file_knowledge: List[dict[str, str]] | None
    generated_code: str | None


def align_spec_parameters(requirements: RequirementSpec, spec: CadSpec) -> CadSpec:
    """Keep requirement values and drop parameter names the normalizer did not emit."""
    req_by_name = {parameter.name: parameter for parameter in requirements.parameters}
    part_names = {part.name for part in spec.parts}
    default_part = spec.parts[0].name if len(spec.parts) == 1 else None
    aligned: List[SpecParameter] = []
    seen: set[str] = set()

    for parameter in spec.parameters:
        source = req_by_name.get(parameter.name)
        if source is None:
            continue
        part = parameter.part if parameter.part in part_names or parameter.part == "assembly" else default_part
        if part is None:
            raise ValueError(f"parameter {parameter.name} has no owning part")
        aligned.append(SpecParameter(
            name=source.name,
            value=source.value,
            unit=source.unit,
            status=source.status,
            part=part,
        ))
        seen.add(source.name)

    for source in requirements.parameters:
        if source.name in seen:
            continue
        if default_part is None:
            raise ValueError(f"parameter {source.name} was dropped and has multiple possible parts")
        aligned.append(SpecParameter(
            name=source.name,
            value=source.value,
            unit=source.unit,
            status=source.status,
            part=default_part,
        ))

    spec.parameters = aligned
    spec.constraints = list(requirements.constraints)
    return spec


def validate_plan_against_spec(spec: CadSpec, plan: GeometryPlan) -> None:
    parts = {part.name for part in spec.parts}
    parameters = {parameter.name for parameter in spec.parameters}
    planned_parts = {step.part for step in plan.steps}
    used_parameters = {
        use.ref
        for step in plan.steps
        for use in step.uses
        if use.source == "parameter"
    }
    if missing_parts := parts - planned_parts:
        raise ValueError(f"plan misses parts: {sorted(missing_parts)}")
    if missing_parameters := parameters - used_parameters:
        raise ValueError(f"plan ignores parameters: {sorted(missing_parameters)}")
    if spec.interfaces and not any(step.operation == "placement" for step in plan.steps):
        raise ValueError("interfaces require a placement step")
    for step in plan.steps:
        if step.part not in parts:
            raise ValueError(f"step {step.id} uses unknown part {step.part}")
        for use in step.uses:
            if use.source == "parameter" and use.ref not in parameters:
                raise ValueError(f"step {step.id} uses unknown parameter {use.ref}")


def validate_knowledge(plan: GeometryPlan, knowledge: KnowledgePack) -> None:
    used = {step.operation for step in plan.steps}
    for snippet in knowledge.snippets:
        if snippet.operation not in used:
            raise ValueError(f"knowledge for unused operation {snippet.operation}")


def validate_repair(plan: GeometryPlan, decision: RepairDecision) -> None:
    ids = {step.id for step in plan.steps}
    missing = [step_id for step_id in decision.step_ids if step_id not in ids]
    if missing:
        raise ValueError(f"repair targets unknown steps: {missing}")
