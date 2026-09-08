import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable



WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "files_mock_claude_response"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


import torch
import tiatoolbox

from tia_tools import (
    tool_predict_kongnet_nucleus_detection,
    tool_predict_semantic_segmentation,
)


# JSON tasks may call only functions registered here.
FUNCTION_REGISTRY: dict[str, Callable[..., Any]] = {
    "predict_semantic_segmentation":
        tool_predict_semantic_segmentation,

    "predict_kongnet_nucleus_detection":
        tool_predict_kongnet_nucleus_detection,
}


def resolve_workspace_path(raw_path: str) -> Path:
    """Resolve absolute or workspace-relative paths."""

    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = WORKSPACE_ROOT / path

    return path.resolve()


def load_task(task_path: Path) -> dict[str, Any]:
    """Load and validate a task specification."""

    if not task_path.is_file():
        raise FileNotFoundError(
            f"Task specification does not exist: {task_path}"
        )

    with task_path.open("r", encoding="utf-8") as file:
        task = json.load(file)

    required_fields = {
        "task_id",
        "function",
        "arguments",
        "expected",
    }

    missing_fields = required_fields.difference(task)

    if missing_fields:
        raise ValueError(
            "Task specification is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if not isinstance(task["task_id"], str):
        raise TypeError("'task_id' must be a string.")

    if not isinstance(task["function"], str):
        raise TypeError("'function' must be a string.")

    if not isinstance(task["arguments"], dict):
        raise TypeError("'arguments' must be a JSON object.")

    if not isinstance(task["expected"], dict):
        raise TypeError("'expected' must be a JSON object.")

    function_name = task["function"]

    if function_name not in FUNCTION_REGISTRY:
        raise ValueError(
            f"Unsupported function: {function_name}. "
            f"Allowed functions: {sorted(FUNCTION_REGISTRY)}"
        )

    return task


def prepare_function_arguments(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Validate input and output paths."""

    prepared = dict(arguments)

    if not prepared.get("wsi_path"):
        raise ValueError(
            "The task requires a nonempty 'wsi_path'."
        )

    wsi_path = resolve_workspace_path(
        prepared["wsi_path"]
    )

    if not wsi_path.is_file():
        raise FileNotFoundError(
            f"Input WSI does not exist: {wsi_path}"
        )

    prepared["wsi_path"] = str(wsi_path)

    if not prepared.get("output_json_path"):
        raise ValueError(
            "The task requires 'output_json_path'."
        )

    output_json_path = resolve_workspace_path(
        prepared["output_json_path"]
    )

    output_json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared["output_json_path"] = str(
        output_json_path
    )

    if not prepared.get("save_dir"):
        raise ValueError(
            "The task requires 'save_dir'."
        )

    save_dir = resolve_workspace_path(
        prepared["save_dir"]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared["save_dir"] = str(save_dir)

    return prepared


def validate_compute_device(
    arguments: dict[str, Any],
) -> None:
    """Fail clearly when CUDA is requested without a GPU."""

    requested_device = str(
        arguments.get("device", "auto")
    ).lower()

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The task requested CUDA, but PyTorch "
                "cannot access a GPU. Submit this task "
                "through a Slurm GPU job."
            )


def validate_outputs(
    expected: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Check required files and AnnotationStore outputs."""

    missing_files: list[str] = []
    invalid_json_files: list[str] = []
    required_files_found: list[str] = []

    for raw_path in expected.get(
        "required_files",
        [],
    ):
        path = resolve_workspace_path(raw_path)

        if not path.is_file():
            missing_files.append(str(path))
            continue

        if path.stat().st_size == 0:
            missing_files.append(str(path))
            continue

        required_files_found.append(str(path))

        if path.suffix.lower() == ".json":
            try:
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    json.load(file)

            except (json.JSONDecodeError, OSError):
                invalid_json_files.append(str(path))

    annotationstore_files: list[str] = []

    require_annotationstore = bool(
        expected.get(
            "require_annotationstore",
            False,
        )
    )

    if require_annotationstore:
        save_dir = Path(arguments["save_dir"])

        annotationstore_files = sorted(
            str(path)
            for path in save_dir.rglob("*.db")
            if path.is_file()
            and path.stat().st_size > 0
        )

    annotationstore_missing = (
        require_annotationstore
        and not annotationstore_files
    )

    passed = (
        not missing_files
        and not invalid_json_files
        and not annotationstore_missing
    )

    return {
        "passed": passed,
        "required_files_found":
            required_files_found,
        "missing_files": missing_files,
        "invalid_json_files":
            invalid_json_files,
        "annotationstore_files":
            annotationstore_files,
        "annotationstore_missing":
            annotationstore_missing,
    }


def collect_environment() -> dict[str, Any]:
    """Collect environment information for reproducibility."""

    cuda_available = torch.cuda.is_available()

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "tiatoolbox_version":
            tiatoolbox.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_build":
            torch.version.cuda,
        "cuda_available": cuda_available,
        "visible_gpu_count":
            torch.cuda.device_count(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
    }


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write formatted JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a pathology function from "
            "a JSON task specification."
        )
    )

    parser.add_argument(
        "--task",
        required=True,
        help="Path to the JSON task specification.",
    )

    command_line_arguments = parser.parse_args()

    task_path = resolve_workspace_path(
        command_line_arguments.task
    )

    try:
        task = load_task(task_path)

        function_name = task["function"]
        function = FUNCTION_REGISTRY[
            function_name
        ]

        function_arguments = (
            prepare_function_arguments(
                task["arguments"]
            )
        )

        output_json_path = Path(
            function_arguments[
                "output_json_path"
            ]
        )

        result_directory = (
            output_json_path.parent
        )

        evaluation_path = (
            result_directory
            / "evaluation_result.json"
        )

    except Exception as error:
        error_result = {
            "status":
                "task_configuration_error",
            "input_task_path": str(
                task_path
            ),
            "error_type":
                type(error).__name__,
            "error": str(error),
            "traceback":
                traceback.format_exc(),
        }

        fallback_path = (
            WORKSPACE_ROOT
            / "evaluation"
            / "results"
            / "configuration_errors"
            / "evaluation_result.json"
        )

        write_json(
            fallback_path,
            error_result,
        )

        print(
            json.dumps(
                error_result,
                indent=2,
            )
        )

        print(
            f"Evaluation record: "
            f"{fallback_path}"
        )

        return 1

    started_at = time.time()
    started_counter = time.perf_counter()

    try:
        validate_compute_device(
            function_arguments
        )

        function_result = function(
            **function_arguments
        )

        runtime_seconds = (
            time.perf_counter()
            - started_counter
        )

        output_validation = (
            validate_outputs(
                task["expected"],
                function_arguments,
            )
        )

        status = (
            "passed"
            if output_validation["passed"]
            else "failed_validation"
        )

        evaluation_result = {
            "task_id": task["task_id"],
            "function": function_name,
            "status": status,
            "started_at_unix": started_at,
            "runtime_seconds":
                runtime_seconds,
            "input_task_path":
                str(task_path),
            "function_arguments":
                function_arguments,
            "function_result":
                function_result,
            "output_validation":
                output_validation,
            "environment":
                collect_environment(),
        }

    except Exception as error:
        runtime_seconds = (
            time.perf_counter()
            - started_counter
        )

        evaluation_result = {
            "task_id": task["task_id"],
            "function": function_name,
            "status": "execution_error",
            "started_at_unix": started_at,
            "runtime_seconds":
                runtime_seconds,
            "input_task_path":
                str(task_path),
            "function_arguments":
                function_arguments,
            "environment":
                collect_environment(),
            "error_type":
                type(error).__name__,
            "error": str(error),
            "traceback":
                traceback.format_exc(),
        }

    write_json(
        evaluation_path,
        evaluation_result,
    )

    print(
        json.dumps(
            evaluation_result,
            indent=2,
        )
    )

    print(
        f"Evaluation record: "
        f"{evaluation_path}"
    )

    return (
        0
        if evaluation_result["status"]
        == "passed"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())