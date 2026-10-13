# CAD Agent v2 delivery

[Download the latest project ZIP](https://github.com/Halva-gif/CAD/raw/refs/heads/main/cad_agent_v2.zip)

Includes corrected Python source, Persian review, instructions, original files, FreeCAD FCStd and STEP examples, preview and verification logs.

## OpenAI-compatible proxy configuration

```dotenv
OPENAI_API_KEY=YOUR_PROXY_API_KEY
OPENAI_BASE_URL=https://your-proxy.example/v1
CAD_PLANNER_MODEL=YOUR_PROVIDER_MODEL_ID
```

You can also use `--base-url` or `OpenAIPlanner(base_url=..., api_key=...)`. Explicit URL overrides the environment. Use the provider's actual API base and model ID. The provider/model must support Chat Completions and JSON Schema structured outputs.

Validation: original 50 tests passed. All 5 SDK transport tests passed after this change, including custom URL routing and precedence. No live request to the user's proxy was made.

See README.md and REPORT_FA.md inside the ZIP for full instructions and supported operations.
