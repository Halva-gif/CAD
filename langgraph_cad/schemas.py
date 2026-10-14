"""Shared CAD contract.

Each agent reads the contracts relevant to its role. The coding agent receives
requirements, CAD specification, geometry plan and verbatim retrieved knowledge.
State stores model_dump() dicts. Names are the join keys:

- part name: CadSpec -> GeometryPlan
- parameter name: RequirementSpec -> CadSpec -> PlanStep.uses
- step id: GeometryPlan -> GeneratedScript
- operation: GeometryPlan -> KnowledgePack
- object name: GeneratedScript metadata (execution currently disabled)
"""

from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


FactStatus = Literal["given", "unknown", "assumed"]
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



class Parameter(Contract):
    name: str = Field(description="Dimension name, 1-3 words")
    value: float | None = Field(description="Stated number, otherwise null")
    unit: str | None = Field(description="Stated unit, otherwise null")
    status: FactStatus
    assumption_reason: str | None = None

    @model_validator(mode='after')
    def values_match_status(self):
        if self.status == 'unknown' and (self.value is not None or self.unit is not None):
            raise ValueError('unknown dimensions must have null value and unit')
        if self.status != 'unknown' and self.value is None:
            raise ValueError('known dimensions require a value')
        if self.status == 'assumed' and not self.unit:
            raise ValueError('assumed dimensions require a unit')
        if self.status == 'assumed' and not self.assumption_reason:
            raise ValueError('assumptions need a recorded reason')
        return self


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


class ResolvedDimensions(Contract):
    parameters: List[Parameter]


class PartSpec(BaseModel):
    name: str = Field(description="Distinct solid name")
    quantity: int = Field(ge=1, description="Stated count, otherwise 1")
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
        if len(names) != len(self.parts):
            raise ValueError('part names must be unique')
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError('parameter names must be unique')
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


class GeometryCheck(Contract):
    """Acceptance criteria established before code generation."""
    metric: Literal['bbox', 'point_inside', 'volume_range']
    part: str
    axis: Literal['x', 'y', 'z'] | None = None
    parameter: str | None = None
    expected: float | None = None
    tolerance: float = Field(default=0.01, gt=0)
    point_mm: List[float] | None = None
    inside: bool | None = None
    minimum: float | None = None
    maximum: float | None = None
    reason: str

    @model_validator(mode='after')
    def sufficient_inputs(self):
        if self.metric == 'bbox' and (self.axis is None or (self.parameter is None and self.expected is None)):
            raise ValueError('bbox needs an axis and expected value or source parameter')
        if self.metric == 'point_inside' and (self.point_mm is None or len(self.point_mm) != 3 or self.inside is None):
            raise ValueError('point_inside needs three mm coordinates and an inside flag')
        if self.metric == 'volume_range' and (self.minimum is None or self.maximum is None or self.minimum > self.maximum):
            raise ValueError('volume_range needs ordered lower/upper bounds in mm^3')
        return self


class GeometryPlan(BaseModel):
    """plan_geometry writes state['geometry_plan']."""

    steps: List[PlanStep]
    checks: List[GeometryCheck] = Field(default_factory=list)

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
    excerpt: str = Field(default='', description='Verbatim source section, including prose and complete code blocks')
    source_hash: str = ''


class KnowledgePack(BaseModel):
    """retrieve_knowledge writes state['knowledge']."""

    snippets: List[KnowledgeSnippet]


class GenerateInput(BaseModel):
    requirements: RequirementSpec
    cad_spec: CadSpec
    plan: GeometryPlan
    knowledge: KnowledgePack


class PartObjects(Contract):
    part: str
    object_names: List[str]


class StepObjects(Contract):
    step_id: str
    object_names: List[str]


class GeneratedScript(BaseModel):
    """generate_code writes state['script']."""

    code: str
    object_names: List[str] = Field(description="FreeCAD object names the script creates")
    part_objects: List[PartObjects] = Field(default_factory=list, description='Final solid objects per exact cad_spec part name; omit construction tools')
    step_objects: List[StepObjects] = Field(default_factory=list, description='Document objects implementing each exact geometry_plan step id')
    knowledge_sources: List[str] = Field(default_factory=list, description='Exact documentation filenames read and used')
    assumptions: List[Parameter] = Field(default_factory=list, description='Additional dimensional choices; status assumed, with values, units and reasons')


class CodeGenerationResult(Contract):
    status: Literal['ready', 'needs_clarification', 'knowledge_insufficient']
    script: GeneratedScript | None
    questions: List[str]
    reason: str

    @model_validator(mode='after')
    def disposition(self):
        if self.status == 'ready' and (self.script is None or self.questions):
            raise ValueError('ready needs a script and no unanswered questions')
        if self.status == 'needs_clarification' and not self.questions:
            raise ValueError('clarification needs concrete questions')
        return self


class State(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    requirements: dict | None
    cad_spec: dict | None
    geometry_plan: dict | None
    knowledge: dict | None
    script: dict | None
    # Original keys retained alongside structured knowledge/script for compatibility.
    selected_files: List[dict[str, str]] | None
    file_knowledge: List[dict[str, str]] | None
    generated_code: str | None
    status: str
    questions: List[str]
    errors: List[str]
    parameter_overrides: dict
    allow_assumptions: bool
    output_directory: str
    usage: dict
    requirements_original: dict
    warnings: List[str]
    selection_candidates: dict
    index_stats: dict
    retrieval_stats: dict
    generation_stats: dict


def align_spec_parameters(requirements: RequirementSpec, spec: CadSpec) -> CadSpec:
    """Keep requirement values and reject parameter names the normalizer did not emit."""
    req_by_name = {parameter.name: parameter for parameter in requirements.parameters}
    part_names = {part.name for part in spec.parts}
    default_part = spec.parts[0].name if len(spec.parts) == 1 else None
    aligned: List[SpecParameter] = []
    seen: set[str] = set()

    for parameter in spec.parameters:
        source = req_by_name.get(parameter.name)
        if source is None:
            raise ValueError(f'CAD spec invented parameter {parameter.name}')
        part = parameter.part if parameter.part in part_names or parameter.part == "assembly" else default_part
        if part is None:
            raise ValueError(f"parameter {parameter.name} has no owning part")
        aligned.append(SpecParameter(
            name=source.name,
            value=source.value,
            unit=source.unit,
            status=source.status,
            assumption_reason=source.assumption_reason,
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
            assumption_reason=source.assumption_reason,
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
    for check in plan.checks:
        if check.part not in parts:
            raise ValueError('geometry check references an unknown part')
        if check.parameter is not None and check.parameter not in parameters:
            raise ValueError('geometry check references an unknown parameter')


def validate_knowledge(plan: GeometryPlan, knowledge: KnowledgePack) -> None:
    used = {step.operation for step in plan.steps}
    for snippet in knowledge.snippets:
        if snippet.operation not in used:
            raise ValueError(f"knowledge for unused operation {snippet.operation}")
    missing = used - {snippet.operation for snippet in knowledge.snippets}
    if missing:
        raise ValueError(f'No retrieved knowledge for operations: {sorted(missing)}')

