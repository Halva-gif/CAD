# Current delivery

Use [cad_langgraph_optimized.zip](https://raw.githubusercontent.com/Halva-gif/CAD/main/cad_langgraph_optimized.zip).

SHA-256: `e63139d90a9e2c1552c2eaa0e73bf15aea80da0cee05b0e7c99b69b64b62317a`

This update preserves the original LangGraph coding-agent architecture, adds persistent heading-only knowledge indexing and bounded prompts/tools, and ends after FreeCAD Python code generation. Runtime execution and feedback were removed as requested. The archive excludes .env, credentials, caches, runs and virtual environments. Supply your own py_wiki files and proxy configuration.

47 tests passed. Model and proxy responses were mocked; no live-model cost reduction percentage or CAD correctness is claimed. See langgraph_cad/README.md and verification/ for configuration and evidence. Previous versions remain in repository history.
