# Agentic workflow alignment evaluation

This directory stores the evidence and results for complete workflow evaluation.

For transfer to the Linux evaluator host, see `LINUX_QUICKSTART.md`. The dependency-free `run_evaluator.py` script sends the frozen prompt, requirements, workflow evidence, and JSON schema to an Ollama-compatible local API and saves versioned run records.

## Directory structure

- `workflows/`: one JSON evidence record per agent workflow.
- `requirements/`: request-derived requirement checklists created before evaluator judgment.
- `artifacts/`: copies of, or stable references to, files produced by evaluated workflows.
- `evaluation_results/`: evaluator judgments. Do not place judgments here until the workflow evidence has been frozen.

## Evidence rule

Each workflow record must contain the verbatim original request, tools available to the agent, ordered tool calls and their returned results or errors, generated artefacts, and the verbatim final response. Missing evidence must remain `null` or empty and the record must remain `incomplete`; it must not be inferred from filenames, tests, or expected outputs.

## Current state

`T001.json` records the first supplied natural-language workflow. The expanded integration section reveals the execution tool sequence, but exact execution arguments and structured responses remain unavailable. This limitation is recorded explicitly rather than reconstructed.
