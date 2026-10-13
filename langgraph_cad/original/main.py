import os
from dotenv import load_dotenv

import json

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

from typing import List
from pydantic import BaseModel, ValidationError

from langchain_core.exceptions import OutputParserException

from langgraph.graph import StateGraph, START, END

from schemas import (
    CadSpec,
    GeometryPlan,
    RequirementSpec,
    State,
    align_spec_parameters,
    validate_plan_against_spec,
)

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from langchain_core.callbacks import UsageMetadataCallbackHandler

load_dotenv()

json_path = 'document_chooser.json'

model = init_chat_model('gpt-6-luna', temperature=0.2)
# model = init_chat_model('gpt-4o-mini', temperature=0.2)

callback = UsageMetadataCallbackHandler()


def _token_totals() -> dict[str, int]:
    totals = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    for usage in callback.usage_metadata.values():
        for key in totals:
            totals[key] += usage.get(key, 0) or 0
    return totals


def _print_agent_usage(name: str, before: dict[str, int]) -> None:
    after = _token_totals()
    print(
        f'{name} tokens: '
        f'input={after["input_tokens"] - before["input_tokens"]} '
        f'output={after["output_tokens"] - before["output_tokens"]} '
        f'total={after["total_tokens"] - before["total_tokens"]}'
    )

class SelectedFile(BaseModel):
    file_name: str
    read: str


class DocNames(BaseModel):
    selected_files: List[SelectedFile]


class KnowledgeNote(BaseModel):
    file_name: str
    note: str


class CodeDoc(BaseModel):
    retrieved_knowledge: List[KnowledgeNote]


def chunk_json_data(json_path: str) -> List[Document]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    return [
        Document(
            page_content=f'{k}: {v}',
        ) for k, v in data.items()
    ]

json_embedding_model = OpenAIEmbeddings()

def create_json_retriever(json_path: str) -> VectorStoreRetriever:
    chunked_data = chunk_json_data(json_path)

    vectore_store = InMemoryVectorStore(json_embedding_model)
    vectore_store.add_documents(chunked_data)

    return vectore_store.as_retriever(
        search_kwargs={'k': 5}
    )

def chunk_md_data(md_path: str) -> List[Document]:
    headers_to_split_on = [
        ('#', 'h1'),
        ('##', 'h2'),
        ('###', 'h3'),
    ]
    with open(md_path, 'r') as f:
        markdown = f.read()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )

    docs = splitter.split_text(markdown)[:12]

    return docs

md_embedding_model = OpenAIEmbeddings()

def create_md_retriever(md_path: str) -> VectorStoreRetriever:
    chunked_data = chunk_md_data(md_path)

    vectore_store = InMemoryVectorStore(md_embedding_model)
    vectore_store.add_documents(chunked_data)

    return vectore_store.as_retriever(
        search_kwargs={'k': 4}
    )

_json_retriever: VectorStoreRetriever | None = None
_md_retrievers: dict[str, VectorStoreRetriever] = {}


def get_json_retriever() -> VectorStoreRetriever:
    global _json_retriever
    if _json_retriever is None:
        print('Indexing document catalog...')
        _json_retriever = create_json_retriever(json_path)
    return _json_retriever


def search_catalog(query: str) -> str:
    docs = get_json_retriever().invoke(query)
    if not docs:
        return 'No relevant file found.'
    return '\n'.join(doc.page_content for doc in docs)


def get_md_retriever(file_name: str) -> VectorStoreRetriever | None:
    path = os.path.join('py_wiki', file_name)
    if not os.path.isfile(path):
        return None
    retriever = _md_retrievers.get(file_name)
    if retriever is None:
        print(f'Indexing {file_name}...')
        retriever = create_md_retriever(path)
        _md_retrievers[file_name] = retriever
    return retriever


def read_code_excerpt(file_name: str, query: str, limit: int = 1800) -> str:
    retriever = get_md_retriever(file_name)
    if retriever is None:
        return f'{file_name}: file not found.'
    docs = retriever.invoke(query)
    text = '\n\n'.join(doc.page_content for doc in docs)
    return text[:limit]

NORMALIZE_REQUEST_PROMPT = """Normalize a CAD request into facts for the next agent.

Do not plan parts, choose FreeCAD operations, or write code.

Rules:
- Copy every stated number and unit exactly.
- parameters are measurable sizes only, such as height, diameter, thickness, length, or angle. Never leave the list empty.
- Appearance words such as smooth, soft, or curved are constraints, not parameters.
- status=given only when the user stated the number.
- status=unknown when that size is missing. Set value and unit to null. Do not guess a number or a unit.
- ambiguities lists one item for each unknown parameter.
- constraints keeps only limits or shape qualities the user stated.
- Keep every text value to one short sentence.

Example: "a smooth curved glass" -> parameters height, rim_diameter, and wall_thickness are unknown; constraints are smooth and curved.
"""


def _latest_user_text(messages: List) -> str:
    for message in reversed(messages):
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content", "")
        else:
            role = getattr(message, "type", None)
            content = getattr(message, "content", "")
        if role in ("user", "human") and isinstance(content, str) and content.strip():
            return content.strip()
    raise ValueError("normalize_request requires a user message")


def _requirement_gap(spec: RequirementSpec) -> str | None:
    if not spec.parameters:
        return (
            "parameters is empty. Add every measurable size needed to model the object. "
            "Use status=unknown with null value and unit when the user stated no number."
        )
    qualities = ("smooth", "soft", "curv", "round", "aesthetic")
    appearance = [
        parameter.name
        for parameter in spec.parameters
        if any(quality in parameter.name.lower() for quality in qualities)
    ]
    if appearance:
        return (
            f"These are constraints, not sizes: {appearance}. "
            "Parameters must be measurable sizes such as height, diameter, and thickness."
        )
    invented = [
        parameter.name
        for parameter in spec.parameters
        if parameter.status == "unknown" and (parameter.value is not None or parameter.unit is not None)
    ]
    if invented:
        return f"Unknown parameters must have null value and unit: {invented}"
    missing_values = [
        parameter.name
        for parameter in spec.parameters
        if parameter.status == "given" and parameter.value is None
    ]
    if missing_values:
        return f"Given parameters need the user's number: {missing_values}"
    return None


def _record_unknowns(spec: RequirementSpec) -> None:
    for parameter in spec.parameters:
        if parameter.status == "unknown" and not any(parameter.name in item for item in spec.ambiguities):
            spec.ambiguities.append(f"{parameter.name} is unspecified")


def normalize_request(state: State):
    extractor = model.with_structured_output(RequirementSpec)
    request = _latest_user_text(state['messages'])
    feedback = None
    spec = None

    print('Normalizing request...')
    usage_before = _token_totals()

    for _ in range(2):
        content = request if feedback is None else f'{request}\n\nRejected: {feedback}'
        try:
            spec = extractor.invoke(
                [
                    {'role': 'system', 'content': NORMALIZE_REQUEST_PROMPT},
                    {'role': 'user', 'content': content},
                ],
                config={'callbacks': [callback]},
            )
            feedback = _requirement_gap(spec)
            if feedback is None:
                break
            spec = None
        except (ValidationError, OutputParserException) as exc:
            feedback = str(exc)[:800]
            spec = None

    if spec is None:
        raise ValueError(f'requirements failed validation: {feedback}')

    _record_unknowns(spec)
    requirements = spec.model_dump()
    _print_agent_usage('normalize_request', usage_before)
    print(requirements)
    return {'requirements': requirements}


CAD_SPEC_PROMPT = """Turn the normalized CAD facts into a part specification.

Do not choose FreeCAD operations or write code.
Do not invent numbers, units, tolerances, or materials.

Rules:
- One entry per distinct solid. A single object stays one part.
- quantity is 1 unless the user stated a count.
- Copy every parameter name, value, unit, and status. Do not add or rename parameters.
- Assign each parameter to its owning part. Use "assembly" only for a relation between parts.
- shape is one sentence using the user's form words. It is not a parameter. Do not rename the object to a cylinder, box, or sphere unless the user used that word.
- features are short identity phrases, such as open rim or wall. Do not put sizes or primitive names there.
- constraints are copied unchanged.
- interfaces are mates between two parts. Leave them empty when there is one part.
- Use the same part names in parts, parameters, and interfaces.
"""


def specify_cad(state: State):
    extractor = model.with_structured_output(CadSpec)
    requirements = state.get('requirements')
    if not requirements:
        raise ValueError('specify_cad requires normalized requirements')

    print('Specifying CAD parts...')
    usage_before = _token_totals()

    spec = extractor.invoke(
        [
            {'role': 'system', 'content': CAD_SPEC_PROMPT},
            {
                'role': 'user',
                'content': json.dumps(requirements, ensure_ascii=False),
            },
        ],
        config={'callbacks': [callback]},
    )
    spec = align_spec_parameters(RequirementSpec.model_validate(requirements), spec)
    cad_spec = spec.model_dump()

    _print_agent_usage('specify_cad', usage_before)
    print(cad_spec)

    return {'cad_spec': cad_spec}


GEOMETRY_PLAN_PROMPT = """Turn the CAD specification into an ordered geometry plan.

Do not write code or invent dimensions.

Rules:
- Use part names exactly as given.
- A parameter ref must be one of the allowed parameter names. shape, features, quantity, and name are descriptions, not parameters.
- If no parameter names are allowed, do not emit source=parameter.
- Every part has at least one step. Every allowed parameter is used at least once.
- step.part is a part name, never assembly. An assembly parameter belongs to the placement step.
- id is "{part}.{short_name}". depends_on and source=step may cite only earlier ids.
- Keep unknown parameters referenced by name. Do not assign values.
- If interfaces is not empty, end with one placement step for that mate.
- Shortest sequence: sketch, then pad for a prismatic solid or revolve for a turned or curved profile; pocket for a hole; thickness for a wall; fillet or chamfer for an edge; boolean only to cut or fuse existing solids.
- intent is one sentence about the resulting shape.
"""


def _plan_request(spec: CadSpec, error: str | None = None) -> str:
    names = [parameter.name for parameter in spec.parameters]
    allowed = ", ".join(names) if names else "(none)"
    content = (
        f"Allowed parameter refs: {allowed}\n"
        "Do not use any other parameter ref.\n"
        + json.dumps(spec.model_dump(), ensure_ascii=False)
    )
    if error is not None:
        content += f"\n\nRejected plan: {error}\nUse only the allowed parameter refs."
    return content


def plan_geometry(state: State):
    cad_spec = state.get('cad_spec')
    if not cad_spec:
        raise ValueError('plan_geometry requires a CAD specification')

    spec = CadSpec.model_validate(cad_spec)
    extractor = model.with_structured_output(GeometryPlan)
    error = None

    print('Planning geometry...')
    usage_before = _token_totals()

    for _ in range(2):
        content = _plan_request(spec, error)
        try:
            plan = extractor.invoke(
                [
                    {'role': 'system', 'content': GEOMETRY_PLAN_PROMPT},
                    {'role': 'user', 'content': content},
                ],
                config={'callbacks': [callback]},
            )
            validate_plan_against_spec(spec, plan)
            break
        except (ValidationError, ValueError, OutputParserException) as exc:
            error = str(exc)[:800]
            plan = None

    if plan is None:
        raise ValueError(f'geometry plan failed validation: {error}')

    geometry_plan = plan.model_dump()
    _print_agent_usage('plan_geometry', usage_before)
    print(geometry_plan)
    return {'geometry_plan': geometry_plan}

def select_files(state: State):
    operations: list[str] = []
    for step in state['geometry_plan']['steps']:
        operation = step['operation']
        if operation not in operations:
            operations.append(operation)

    print('Selecting the relevant docs...')
    usage_before = _token_totals()
    matches = []
    for operation in operations:
        print(f'Searching catalog for {operation}...')
        matches.append(f'## {operation}\n{search_catalog(operation)}')

    selector = model.with_structured_output(DocNames)
    selected = selector.invoke(
        [
            {
                'role': 'system',
                'content': """Pick FreeCAD documentation files for the operations below.

The catalog matches are already retrieved. Each line is `file_name: description`.

Rules:
- Use only file names present in the matches.
- Choose at most two files for each operation.
- read: one sentence saying what Python to copy from that file.
- Do not invent file names.
""",
            },
            {'role': 'user', 'content': '\n\n'.join(matches)},
        ],
        config={'callbacks': [callback]},
    )

    _print_agent_usage('select_files', usage_before)
    selected_files = [item.model_dump() for item in selected.selected_files]
    print(selected_files)
    return {'selected_files': selected_files}


def code_knowledge_extracter(state: State):
    print('Searching into docs...')
    usage_before = _token_totals()
    blocks = []
    for item in state['selected_files'] or []:
        file_name = item.get('file_name', '')
        query = item.get('read') or file_name
        print(f'Extracting knowledge from {file_name}')
        blocks.append(f'# {file_name}\n{read_code_excerpt(file_name, query)}')

    if not blocks:
        _print_agent_usage('code_knowledge_extracter', usage_before)
        return {'file_knowledge': []}

    extractor = model.with_structured_output(CodeDoc)
    extracted = extractor.invoke(
        [
            {
                'role': 'system',
                'content': """Extract only the FreeCAD Python needed for the supplied operations.

For each useful file return file_name and a short code-focused note.
Do not invent APIs. Keep every note compact.
""",
            },
            {'role': 'user', 'content': '\n\n'.join(blocks)},
        ],
        config={'callbacks': [callback]},
    )

    _print_agent_usage('code_knowledge_extracter', usage_before)
    file_knowledge = [item.model_dump() for item in extracted.retrieved_knowledge]
    print(file_knowledge)
    return {'file_knowledge': file_knowledge}

def code_generation(state: State):
    agent = create_agent(
        model=model,
        system_prompt=f"""You are a FreeCAD Python developer.
Task:
    1. Read the FreeCAD library documentation (mentioned) and learn how to code using this lib.
    2. Generate FreeCAD Python code to generate the requested CAD model.
    3. Double check the code and return a bug-free and runnable code for CAD model.

IMPORTANT: You must read the lib document and generate code ONLY and ONLY using this knowledge not your pretrained data.

FreeCAD lib document:
{state['file_knowledge']}
""",
    )

    print(f'Generating CAD code for {state["messages"]}...')
    usage_before = _token_totals()

    generated_code = agent.invoke(
        {
            'messages': {
                'role': 'user',
                'content': f'Generate CAD model script for \n{state["messages"]}'
            }
        },
        config={'callbacks': [callback]}
    )

    _print_agent_usage('code_generation', usage_before)
    print(generated_code['messages'][-1].content)

    return {'generated_code': generated_code['messages'][-1].content}

builder = StateGraph(State)

builder.add_node('normalize_request', normalize_request)
builder.add_node('specify_cad', specify_cad)
builder.add_node('plan_geometry', plan_geometry)
builder.add_node('select_files', select_files)
builder.add_node('code_knowledge_extracter', code_knowledge_extracter)
builder.add_node('code_generation', code_generation)

builder.add_edge(START, 'normalize_request')
builder.add_edge('normalize_request', 'specify_cad')
builder.add_edge('specify_cad', 'plan_geometry')
builder.add_edge('plan_geometry', 'select_files')
builder.add_edge('select_files', 'code_knowledge_extracter')
builder.add_edge('code_knowledge_extracter', 'code_generation')
builder.add_edge('code_generation', END)

graph = builder.compile()

if __name__ == '__main__':
    result = graph.invoke(
        {
            'messages': [
                {
                    'role': 'user',
                    'content': 'CAD model of a glass. The model must have smooth and soft and curved surfacing.'
                }
            ],
            'requirements': None,
            'cad_spec': None,
            'geometry_plan': None,
            'knowledge': None,
            'script': None,
            'run_result': None,
            'validation': None,
            'repair': None,
            'attempt': 0,
            'selected_files': None,
        }
    )

    print(result['requirements'])
    print(result['cad_spec'])
    print(result['geometry_plan'])

    for i in result['selected_files']:
        print(i)
        print('\n\n\n')

    for i in result['file_knowledge']:
        print(i)

    print(result['generated_code'])