# Evaluator configuration

Use `evaluator_prompt_v1.txt` unchanged for every judgment in the first evaluation run. Replace only the two placeholders with the corresponding frozen requirements JSON and workflow JSON.

Require output conforming to `evaluator_output_schema_v1.json`. Record the exact GPT-OSS 120B model identifier, hosting provider or local runtime, model/reasoning settings, sampling settings, prompt version, schema version, and execution timestamp with every judgment.

Do not evaluate `T001` yet. First run a schema/prompt smoke test, then manually inspect whether the evaluator correctly distinguishes missing evidence from demonstrated failure.
