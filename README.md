# FreeCAD LangGraph agents — persistent knowledge, code generation only

نسخهٔ فعلی روی پروژهٔ اولیهٔ LangGraph اصلاح شده است. ایجنت `create_agent` همچنان کد واقعی Python FreeCAD می‌نویسد.

[دانلود ZIP نسخهٔ فعلی](https://raw.githubusercontent.com/Halva-gif/CAD/main/cad_langgraph_optimized.zip)

[راهنمای نصب و اجرا](langgraph_cad/README.md) · [کد اصلی](langgraph_cad/main.py) · [نتیجهٔ آزمون](langgraph_cad/verification/tests.txt)

- Persistent incremental SQLite index over all py_wiki Markdown, with vector + exact API-name retrieval.
- Chunk boundaries only at #, ## and ### outside source examples; no size/paragraph/code splitting.
- Unique source sections, bounded tool/model calls and per-call usage logs.
- Execution, CAD validation, debugging and runtime feedback temporarily removed at the user's request.
- Output: generated_code.py and state.json; code_ready does not mean validated CAD.
- 47 tests passed with actual LangGraph/create_agent and mocked model/HTTP responses. User knowledge and live proxy were unavailable.

```bash
python main.py --index-only
python main.py --allow-assumptions --out runs
```

Run these commands inside the extracted project after configuring .env and supplying py_wiki.

The older cad_langgraph_corrected.zip and cad_agent_v2.zip are historical packages; use cad_langgraph_optimized.zip for the current version. Original uploads remain in langgraph_cad/original/.
