# Linux quick start: GPT-OSS evaluator

## 1. Inspect the server

```bash
nvidia-smi
free -h
df -h
```

For `gpt-oss:120b`, use an approximately 80 GB GPU or a supported multi-GPU configuration with sufficient total memory. Keep at least 100 GB of free disk space.

## 2. Extract and enter the package

```bash
unzip alignment_evaluation_portable.zip
cd alignment_evaluation
```

## 3. Install and start Ollama

Follow the Ollama installation procedure approved for the Linux server. On a managed university cluster, ask the administrator before installing system services or downloading large model weights.

Confirm that Ollama is reachable:

```bash
ollama --version
ollama list
```

Download the exact evaluator model:

```bash
ollama pull gpt-oss:120b
```

If Ollama is not already running as a service, start it in a separate terminal or job allocation:

```bash
ollama serve
```

## 4. Test model access

```bash
curl http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss:120b","messages":[{"role":"user","content":"Return only JSON: {\"status\":\"working\"}"}],"stream":false,"format":"json"}'
```

## 5. Run one smoke-test judgment

No third-party Python packages are required.

```bash
python3 run_evaluator.py --task-id T001 --model gpt-oss:120b --runs 1
```

The result will be written to:

```text
evaluation_results/T001/run_01.json
```

Inspect it before running repeated judgments:

```bash
python3 -m json.tool evaluation_results/T001/run_01.json
```

## 6. Run five recorded judgments after the smoke test is approved

Remove or move the smoke-test result first so it is not mistaken for a final experimental run. Then execute:

```bash
python3 run_evaluator.py --task-id T001 --model gpt-oss:120b --runs 5 --temperature 0.2 --seed 42000
```

Do not substitute `gpt-oss:20b` while reporting the evaluator as GPT-OSS 120B. Preserve the generated result JSON files and the exact `nvidia-smi`, `ollama --version`, and `ollama list` outputs as experiment metadata.
