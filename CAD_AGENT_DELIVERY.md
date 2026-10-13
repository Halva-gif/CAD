# Original LangGraph CAD project — corrected delivery

[Download the corrected original LangGraph project](https://github.com/Halva-gif/CAD/raw/refs/heads/main/cad_langgraph_corrected.zip)

[Browse corrected source](./langgraph_cad/README.md) · [main.py diff against the uploaded original](./langgraph_cad/verification/main.py.diff) · [schemas.py diff](./langgraph_cad/verification/schemas.py.diff)

This delivery starts from the user's uploaded main.py and schemas.py and preserves LangGraph, the original six nodes, vector RAG, and the create_agent coding agent. The agent writes actual Python FreeCAD code using supplied documentation and optional knowledge-search/read tools. There is no CAD compiler, construction DSL, or fixed geometry generator.

Corrections: full requirements/spec/plan context reaches code generation; source code examples survive retrieval; embeddings have a persistent cache; unknown dimensions follow an explicit clarification/assumption policy; actual FreeCAD execution, geometric checks, artifact roundtrips and bounded repair are connected to the graph.

Validation: 29 passing tests, including real LangGraph/create_agent tools and real FreeCAD Sketcher/PartDesign execution and repair. Model responses and proxy HTTP are simulated. The user's real knowledge document bodies and live proxy credentials were not supplied; no live model-quality benchmark is claimed.

Configure OPENAI_BASE_URL, OPENAI_API_KEY and CAD_MODEL in .env. Supply the actual py_wiki document bodies. Tool/function calling is required. See the Persian README and DEBUG.md for details.

The earlier cad_agent_v2.zip is a different compiler-based rewrite and is superseded by this corrected original-project delivery. Use cad_langgraph_corrected.zip for the requested architecture.
