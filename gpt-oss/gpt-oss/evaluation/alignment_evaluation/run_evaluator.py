#!/usr/bin/env python3
"""Run the frozen workflow-alignment evaluator against an Ollama server."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path):
    return json.loads(read_text(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_judgment(judgment: dict, task_id: str) -> None:
    required = {
        "task_id",
        "rubric_version",
        "requirement_results",
        "dimension_scores",
        "overall_label",
        "critical_error",
        "primary_error_type",
        "confidence",
        "justification",
    }
    missing = sorted(required - judgment.keys())
    if missing:
        raise ValueError(f"judgment missing required fields: {missing}")
    if judgment["task_id"] != task_id:
        raise ValueError(f"expected task_id {task_id}, received {judgment['task_id']}")
    if judgment["overall_label"] not in {
        "FULLY_SUCCESSFUL",
        "PARTIALLY_SUCCESSFUL",
        "FAILED",
    }:
        raise ValueError("invalid overall_label")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="T001")
    parser.add_argument("--model", default="gpt-oss:120b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    requirements_path = ROOT / "requirements" / f"{args.task_id}_requirements.json"
    workflow_path = ROOT / "workflows" / f"{args.task_id}.json"
    prompt_path = ROOT / "evaluator" / "evaluator_prompt_v1.txt"
    schema_path = ROOT / "evaluator" / "evaluator_output_schema_v1.json"

    requirements = read_json(requirements_path)
    workflow = read_json(workflow_path)
    schema = read_json(schema_path)
    prompt_template = read_text(prompt_path)
    prompt = prompt_template.replace(
        "{{REQUIREMENTS_JSON}}", json.dumps(requirements, ensure_ascii=False, indent=2)
    ).replace("{{WORKFLOW_JSON}}", json.dumps(workflow, ensure_ascii=False, indent=2))

    if "{{REQUIREMENTS_JSON}}" in prompt or "{{WORKFLOW_JSON}}" in prompt:
        raise RuntimeError("prompt placeholders were not replaced")

    output_dir = ROOT / "evaluation_results" / args.task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = args.base_url.rstrip("/") + "/api/chat"

    for index in range(args.runs):
        run_number = index + 1
        seed = args.seed + index
        started = datetime.now(timezone.utc)
        payload = {
            "model": args.model,
            "messages": [{"role": "system", "content": prompt}],
            "stream": False,
            "format": schema,
            "options": {"temperature": args.temperature, "seed": seed, "num_ctx": 16384, "num_predict": 4096},
        }
        response = None
        try:
            response = post_json(endpoint, payload, args.timeout)
            content = response.get("message", {}).get("content", "")
            judgment = json.loads(content)
            validate_judgment(judgment, args.task_id)
            status = "valid"
            error = None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            judgment = None
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        record = {
            "task_id": args.task_id,
            "run_number": run_number,
            "status": status,
            "error": error,
            "configuration": {
                "model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "seed": seed,
                "prompt_version": "1.0",
                "num_ctx": 16384,
                "num_predict": 4096,
                "schema_version": "1.0",
                "requirements_sha256": sha256(requirements_path),
                "workflow_sha256": sha256(workflow_path),
                "prompt_sha256": sha256(prompt_path),
                "schema_sha256": sha256(schema_path),
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "judgment": judgment,
            "ollama_response_metadata": {
                key: response.get(key)
                for key in (
                    "model",
                    "created_at",
                    "done_reason",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
                if isinstance(response, dict) and key in response
            },
            "raw_model_content": (
                response.get("message", {}).get("content")
                if isinstance(response, dict)
                else None
            ),
        }
        output_path = output_dir / f"run_{run_number:02d}.json"
        output_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{output_path}: {status}")
        if error:
            print(error, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
