import sys
import json
import traceback
import os
import uuid
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict

from tia_tools import (
    tool_health,
    tool_echo,
    tool_list_files,
    tool_wsi_metadata,
    tool_wsi_thumbnail,
    tool_tissue_mask,
    tool_extract_patches,
    tool_analyze_patch_statistics,
    tool_predict_kather_resnet18,
    tool_predict_kongnet_nucleus_detection,
    tool_export_kongnet_nuclei_to_csv,
    tool_find_cells_within_radius,
    tool_compute_cell_type_cooccurrence,
    tool_compute_nearest_neighbour_features,
    tool_aggregate_kather_metrics,
    tool_summarize_kather_results,
    tool_generate_confidence_histogram,
    tool_compare_masked_vs_unmasked_runs,
    tool_threshold_sensitivity_analysis,
    tool_extract_top_abnormal_patches,
    tool_generate_final_ai_report,
    tool_generate_kongnet_ai_report,
)

PROTOCOL_VERSION = "2025-06-18"

PENDING_PLANS: Dict[str, Dict[str, Any]] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def optional_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def bool_arg(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def process_exists(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information,
                False,
                int(pid),
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return True

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def prediction_jobs_dir(args: Dict[str, Any]) -> str:
    save_dir = args.get("save_dir")
    output_json_path = args.get("output_json_path")

    if isinstance(save_dir, str) and save_dir.strip():
        base_dir = os.path.dirname(save_dir.rstrip("\\/")) or save_dir
    elif isinstance(output_json_path, str) and output_json_path.strip():
        base_dir = os.path.dirname(output_json_path) or os.getcwd()
    else:
        base_dir = os.getcwd()

    return os.path.join(base_dir, "prediction_jobs")


def start_prediction_job(args: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    jobs_dir = prediction_jobs_dir(args)
    os.makedirs(jobs_dir, exist_ok=True)

    job_args_path = os.path.join(jobs_dir, f"{job_id}.args.json")
    status_path = os.path.join(jobs_dir, f"{job_id}.status.json")
    stdout_path = os.path.join(jobs_dir, f"{job_id}.stdout.log")
    stderr_path = os.path.join(jobs_dir, f"{job_id}.stderr.log")

    job_tool = str(args.get("job_tool", args.get("tool_name", "predict_kather_resnet18")))
    default_model_name = (
        "KongNet_PanNuke_1"
        if job_tool == "predict_kongnet_nucleus_detection"
        else "resnet18-kather100k"
    )

    job_args = {
        "patch_dir": args.get("patch_dir"),
        "output_json_path": args.get("output_json_path"),
        "output_csv_path": args.get("output_csv_path"),
        "model_name": str(args.get("model_name", default_model_name)),
        "batch_size": int(args.get("batch_size", 64)),
        "device": str(args.get("device", "auto")),
        "input_size": int(args.get("input_size", 224)),
        "wsi_path": args.get("wsi_path"),
        "save_dir": args.get("save_dir"),
        "ioconfig": args.get("ioconfig"),
        "output_type": str(args.get("output_type", "annotationstore")),
        "patch_mode": bool_arg(args.get("patch_mode"), False),
        "job_tool": job_tool,
        "auto_get_mask": bool_arg(args.get("auto_get_mask"), False),
        "num_workers": args.get("num_workers"),
        "overwrite": bool_arg(args.get("overwrite"), True),
    }

    atomic_write_json(job_args_path, job_args)
    atomic_write_json(status_path, {
        "job_id": job_id,
        "status": "queued",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "pid": None,
        "args_path": job_args_path,
        "status_path": status_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "output_json_path": job_args.get("output_json_path"),
        "save_dir": job_args.get("save_dir"),
    })

    with open(stdout_path, "ab") as stdout_f, open(stderr_path, "ab") as stderr_f:
        proc = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--run-predict-job",
                job_id,
                job_args_path,
                status_path,
            ],
            cwd=os.getcwd(),
            stdout=stdout_f,
            stderr=stderr_f,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )

    status = load_prediction_job_status(status_path=status_path)
    status.update({
        "status": "running",
        "pid": proc.pid,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    })
    atomic_write_json(status_path, status)

    return {
        "status": "started",
        "job_id": job_id,
        "pid": proc.pid,
        "status_path": status_path,
        "args_path": job_args_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "message": (
            "Prediction is running in a background process so the MCP request can "
            "return before Claude's timeout. Use check_prediction_job with this "
            "job_id or status_path to monitor completion."
        ),
    }


def load_prediction_job_status(
    job_id: str = "",
    status_path: str = "",
    search_dir: str = "",
) -> Dict[str, Any]:
    if not status_path:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("check_prediction_job requires job_id or status_path.")

        roots = []
        if isinstance(search_dir, str) and search_dir.strip():
            roots.append(search_dir)
            roots.append(os.path.join(search_dir, "prediction_jobs"))
        roots.append(os.path.join(os.getcwd(), "prediction_jobs"))

        for root in roots:
            candidate = os.path.join(root, f"{job_id}.status.json")
            if os.path.exists(candidate):
                status_path = candidate
                break

        if not status_path:
            raise FileNotFoundError(
                f"Could not find status for job_id={job_id!r}. Pass status_path directly."
            )

    with open(status_path, "r", encoding="utf-8") as f:
        status = json.load(f)

    pid = status.get("pid")
    if status.get("status") == "running" and isinstance(pid, int):
        if not process_exists(pid):
            status["status_note"] = (
                "Process is no longer visible. If result or error is not populated, "
                "check stderr_path."
            )

    return status


def run_predict_job(job_id: str, args_path: str, status_path: str) -> int:
    try:
        with open(args_path, "r", encoding="utf-8") as f:
            job_args = json.load(f)

        status = load_prediction_job_status(status_path=status_path)
        status.update({
            "job_id": job_id,
            "status": "running",
            "pid": os.getpid(),
            "started_at": status.get("started_at") or utc_now_iso(),
            "updated_at": utc_now_iso(),
        })
        atomic_write_json(status_path, status)

        job_tool = str(job_args.pop("job_tool", "predict_kather_resnet18"))
        if job_tool == "predict_kongnet_nucleus_detection":
            allowed = {
                "wsi_path",
                "output_json_path",
                "model_name",
                "batch_size",
                "device",
                "save_dir",
                "output_type",
                "patch_mode",
                "auto_get_mask",
                "num_workers",
                "overwrite",
            }
            result = tool_predict_kongnet_nucleus_detection(
                **{key: value for key, value in job_args.items() if key in allowed}
            )
        else:
            allowed = {
                "patch_dir",
                "output_json_path",
                "output_csv_path",
                "model_name",
                "batch_size",
                "device",
                "input_size",
                "wsi_path",
                "save_dir",
                "ioconfig",
                "output_type",
                "patch_mode",
            }
            result = tool_predict_kather_resnet18(
                **{key: value for key, value in job_args.items() if key in allowed}
            )

        status.update({
            "status": "completed",
            "completed_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "result": result,
        })
        atomic_write_json(status_path, status)
        return 0
    except Exception as exc:
        status = {
            "job_id": job_id,
            "status": "failed",
            "updated_at": utc_now_iso(),
            "error": str(exc),
            "trace": traceback.format_exc(),
            "args_path": args_path,
            "status_path": status_path,
        }
        try:
            existing = load_prediction_job_status(status_path=status_path)
            existing.update(status)
            status = existing
        except Exception:
            pass
        atomic_write_json(status_path, status)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def send(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def jsonrpc_error(req_id, code: int, message: str, data: Any = None) -> None:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    send({"jsonrpc": "2.0", "id": req_id, "error": err})


def tool_result(req_id, payload: Any) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    send({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": False
        }
    })


def tool_error(req_id, message: str, data: Any = None) -> None:
    text = f"Error: {message}" if data is None else json.dumps(
        {"error": message, "data": data},
        indent=2
    )
    send({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": True
        }
    })


def ensure_output_dir(output_dir: str) -> None:
    if output_dir and isinstance(output_dir, str):
        os.makedirs(output_dir, exist_ok=True)


def infer_task_type(user_prompt: str) -> str:
    request = user_prompt.lower()

    if any(k in request for k in [
        "co-occurrence",
        "cooccurrence",
        "nearest neighbour",
        "nearest neighbor",
        "within radius",
        "cell radius",
        "spatial biology",
        "immune-to-epithelial",
        "immune to epithelial",
        "immune-to-neoplastic",
        "immune to neoplastic",
        "export nuclei",
        "nuclei csv",
        "nucleus csv",
    ]):
        return "nucleus_spatial_analysis"

    if any(k in request for k in [
        "kongnet",
        "nucleus",
        "nuclei",
        "nuclear",
        "instance segmentation",
        "nucleus detection",
        "tiaviz",
    ]):
        return "kongnet_nucleus_detection"

    if any(k in request for k in [
        "final ai report",
        "final report",
        "tumour likelihood map",
        "tumor likelihood map",
        "threshold sensitivity",
        "top abnormal",
        "top-k",
        "top k",
        "most abnormal patches",
        "post-process",
        "postprocess",
        "summary",
        "summarize",
        "hotspot",
        "histogram",
        "confidence",
        "compare masked",
        "masked vs unmasked",
    ]):
        return "post_processing"

    if (
        "thumbnail" in request
        or "overview image" in request
        or "preview image" in request
    ) and not any(k in request for k in [
        "mask",
        "patch",
        "heterogeneity",
        "statistics",
        "entropy",
        "prediction",
        "predict",
        "classify",
        "kather",
        "tumour",
        "tumor",
    ]):
        return "thumbnail_only"

    if (
        "tissue mask" in request
        or "mask" in request
        or "segment tissue" in request
    ) and not any(k in request for k in [
        "patch",
        "heterogeneity",
        "statistics",
        "prediction",
        "predict",
        "classify",
        "kather",
        "tumour",
        "tumor",
    ]):
        return "tissue_mask_only"

    if (
        "patch" in request
        or "extract patches" in request
    ) and not any(k in request for k in [
        "heterogeneity",
        "statistics",
        "prediction",
        "predict",
        "classify",
        "kather",
        "tumour",
        "tumor",
    ]):
        return "patches_only"

    if (
        "prediction" in request
        or "predict" in request
        or "kather" in request
        or "resnet18" in request
        or "classify" in request
        or "tumour" in request
        or "tumor" in request
        or "likelihood" in request
        or "abnormality" in request
        or "tissue class" in request
    ):
        return "kather_prediction"

    if (
        "heterogeneity" in request
        or "statistics" in request
        or "entropy" in request
        or "variance" in request
        or "cluster" in request
        or "overlay" in request
    ):
        return "spatial_heterogeneity"

    return "general_wsi_analysis"


def build_plan(user_prompt: str, wsi_path: str, output_dir: str) -> Dict[str, Any]:
    task_type = infer_task_type(user_prompt)
    approval_token = str(uuid.uuid4())

    ensure_output_dir(output_dir)

    if task_type == "thumbnail_only":
        plan = {
            "task_type": task_type,
            "goal": "Generate a thumbnail overview of the WSI.",
            "steps": [
                "Read WSI metadata to confirm the slide can be opened.",
                "Generate a low-resolution thumbnail image.",
                "Save the thumbnail in the requested output directory."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "wsi_thumbnail"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "thumbnail.png")
            ]
        }

    elif task_type == "tissue_mask_only":
        plan = {
            "task_type": task_type,
            "goal": "Generate a tissue mask and optional overlay for the WSI.",
            "steps": [
                "Read WSI metadata.",
                "Generate a thumbnail for visual reference.",
                "Generate a tissue mask using either morphological or Otsu masking.",
                "Save the tissue mask and overlay images."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "tissue_mask"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "tissue_overlay.png")
            ]
        }

    elif task_type == "patches_only":
        plan = {
            "task_type": task_type,
            "goal": "Extract tissue-rich patches from the WSI.",
            "steps": [
                "Read WSI metadata.",
                "Generate a thumbnail.",
                "Generate or use tissue masking to avoid background regions.",
                "Extract tissue-rich patches and save them to the output folder."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "extract_patches"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "patches")
            ]
        }

    elif task_type == "post_processing":
        plan = {
            "task_type": task_type,
            "goal": "Run interpretability post-processing on existing Kather100K prediction outputs.",
            "steps": [
                "Use saved Kather prediction, metric, and patch-statistic files.",
                "Generate a plain-English summary of the Kather results.",
                "Generate a confidence histogram from patch-level prediction confidence scores.",
                "Generate a hotspot overlay showing spatial clusters of high-abnormality patches.",
                "Generate a continuous tumour-relevant likelihood map from abnormality scores.",
                "Run threshold sensitivity analysis across multiple abnormality thresholds.",
                "Extract the top-K most abnormal patches and save them separately.",
                "Generate a final AI interpretability report.",
                "Optionally compare masked and unmasked metric files if both are provided."
            ],
            "suggested_tools_after_approval": [
                "compare_masked_vs_unmasked_runs"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "masked_vs_unmasked_comparison.txt")
            ],
            "default_parameters": {
                "abnormality_threshold": 0.5,
                "histogram_bins": 20,
                "patch_size": 224,
                "top_k": 20,
                "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7]
            },
            "clinical_warning": (
                "The post-processing outputs explain model-confidence behaviour, not clinical diagnosis."
            )
        }

    elif task_type == "nucleus_spatial_analysis":
        plan = {
            "task_type": task_type,
            "goal": "Compute spatial cell features from an existing KongNet AnnotationStore without rerunning inference.",
            "steps": [
                "Read existing KongNet nucleus points and class labels from the AnnotationStore.",
                "Export nucleus coordinates, classes, and probabilities to CSV.",
                "Count selected target cells within a configurable physical radius.",
                "Compute the cell-type co-occurrence matrix and inflammatory cell ratios.",
                "Compute nearest-neighbour distances by source and target cell type."
            ],
            "suggested_tools_after_approval": [
                "export_kongnet_nuclei_to_csv",
                "find_cells_within_radius",
                "compute_cell_type_cooccurrence",
                "compute_nearest_neighbour_features",
                "generate_kongnet_ai_report"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kongnet_nuclei.csv"),
                os.path.join(output_dir, "radius_neighbourhoods.csv"),
                os.path.join(output_dir, "cell_type_cooccurrence.json"),
                os.path.join(output_dir, "nearest_neighbours.csv"),
                os.path.join(output_dir, "kongnet_ai_interpretability_report.txt")
            ],
            "default_parameters": {
                "radius": 50.0,
                "distance_units": "microns",
                "min_probability": 0.0
            },
            "clinical_warning": (
                "These are model-derived spatial research features, not a clinical diagnosis."
            )
        }

    elif task_type == "kongnet_nucleus_detection":
        plan = {
            "task_type": task_type,
            "goal": "Run KongNet PanNuke nucleus detection on the WSI and generate a TIAViz-compatible AnnotationStore.",
            "steps": [
                "Read WSI metadata.",
                "Run TIAToolbox NucleusDetector with model KongNet_PanNuke_1.",
                "Save nucleus detections as a TIAViz-compatible AnnotationStore (.db).",
                "Save a JSON run summary containing the TIAViz launch command.",
                "Open the slide and nucleus overlay in TIAViz using the generated command."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "predict_kongnet_nucleus_detection",
                "check_prediction_job",
                "save_run_report"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kongnet_nucleus_predictions.json"),
                os.path.join(output_dir, "kongnet_nucleus_annotationstore"),
                os.path.join(output_dir, "run_report.json")
            ],
            "default_parameters": {
                "model_name": "KongNet_PanNuke_1",
                "batch_size": 16,
                "output_type": "annotationstore",
                "patch_mode": False,
                "auto_get_mask": False,
                "run_async": True
            },
            "class_mapping": {
                "Neoplastic": 0,
                "Inflammatory": 1,
                "Connective": 2,
                "Dead": 3,
                "Epithelial": 4
            },
            "clinical_warning": (
                "The output is nucleus detection/classification model output, not a clinical diagnosis."
            )
        }

    elif task_type == "kather_prediction":
        plan = {
            "task_type": task_type,
            "goal": "Run ResNet18-Kather100K WSI tissue classification and generate a TIAViz-compatible AnnotationStore.",
            "steps": [
                "Read WSI metadata.",
                "Run pretrained ResNet18-Kather100K classification directly on the WSI.",
                "Save prediction output as a TIAViz-compatible AnnotationStore (.db).",
                "Save a JSON run summary containing the TIAViz launch command.",
                "Open the slide and overlay in TIAViz using the generated command."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "predict_kather_resnet18",
                "check_prediction_job",
                "save_run_report"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kather_predictions.json"),
                os.path.join(output_dir, "wsi_predictions_annotationstore"),
                os.path.join(output_dir, "run_report.json")
            ],
            "default_parameters": {
                "model_name": "resnet18-kather100k",
                "batch_size": 64,
                "output_type": "annotationstore",
                "run_async": True
            },
            "clinical_warning": (
                "The output is tissue-type classification and model-confidence analysis, not a clinical diagnosis."
            )
        }

    else:
        plan = {
            "task_type": task_type,
            "goal": "Perform baseline spatial heterogeneity analysis on the WSI.",
            "steps": [
                "Read WSI metadata.",
                "Generate a thumbnail.",
                "Generate a tissue mask and overlay.",
                "Extract dense tissue-rich patches.",
                "Compute baseline patch-level heterogeneity metrics such as colour variance and entropy.",
                "Save all visual outputs and metrics."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "analyze_patch_statistics"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "patch_statistics.json")
            ]
        }

    plan["wsi_path"] = wsi_path
    plan["output_dir"] = output_dir
    plan["approval_token"] = approval_token
    plan["instruction"] = (
        "Show this plan to the user and wait for explicit approval before calling execution tools. "
        "After approval, call only the relevant MCP tools listed in suggested_tools_after_approval."
    )

    PENDING_PLANS[approval_token] = plan
    return plan


def require_plan(args: Dict[str, Any]) -> str:
    token = args.get("approval_token")

    if not isinstance(token, str) or not token.strip():
        raise RuntimeError(
            "Execution requires approval_token from propose_pathology_plan. "
            "First call propose_pathology_plan, show the plan to the user, and wait for approval."
        )

    if token not in PENDING_PLANS:
        raise RuntimeError("Invalid or expired approval_token. Generate a fresh plan first.")

    return token


def handle_initialize(req: Dict[str, Any]) -> None:
    req_id = req["id"]
    params = req.get("params", {})
    agreed_version = params.get("protocolVersion") or PROTOCOL_VERSION

    send({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": agreed_version,
            "capabilities": {
                "tools": {
                    "listChanged": False
                }
            },
            "serverInfo": {
                "name": "tiatoolbox-mcp-only-agent-server",
                "title": "MCP-only Single WSI Pathology Agent Server",
                "version": "13.0.0-kather-advanced-mvp"
            }
        }
    })


def handle_tools_list(req: Dict[str, Any]) -> None:
    tools = [
        {
            "name": "propose_pathology_plan",
            "title": "Propose Pathology Plan",
            "description": "Required first step for every pathology request.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_prompt": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "output_dir": {"type": "string"}
                },
                "required": ["user_prompt", "wsi_path", "output_dir"],
                "additionalProperties": False
            }
        },
        {
            "name": "health",
            "title": "Health Check",
            "description": "Returns whether the MCP-only pathology server is alive.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}
        },
        {
            "name": "echo",
            "title": "Echo",
            "description": "Echoes back text.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False
            }
        },
        {
            "name": "list_files",
            "title": "List Files",
            "description": "Lists files and folders in a directory. Creates the directory if it does not exist.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "max_items": {"type": "integer"}
                },
                "required": ["directory"],
                "additionalProperties": False
            }
        },
        {
            "name": "wsi_metadata",
            "title": "WSI Metadata",
            "description": "Reads WSI metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "path": {"type": "string"}
                },
                "required": ["approval_token", "path"],
                "additionalProperties": False
            }
        },
        {
            "name": "wsi_thumbnail",
            "title": "WSI Thumbnail",
            "description": "Generates and saves a WSI thumbnail.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "resolution": {"type": "number"},
                    "units": {"type": "string"}
                },
                "required": ["approval_token", "path", "output_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "tissue_mask",
            "title": "Tissue Mask",
            "description": "Generates a tissue mask.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "path": {"type": "string"},
                    "output_mask_path": {"type": "string"},
                    "output_overlay_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "method": {"type": "string", "enum": ["morphological", "otsu"]}
                },
                "required": ["approval_token", "path", "output_mask_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "extract_patches",
            "title": "Extract Tissue Patches",
            "description": "Extracts dense tissue-rich patches from a WSI.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "path": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "patch_size": {"type": "integer"},
                    "stride": {"type": "integer"},
                    "level": {"type": "integer"},
                    "max_patches": {"type": "integer"},
                    "min_tissue_fraction": {"type": "number"},
                    "mpp": {"type": "number"}
                },
                "required": ["approval_token", "path", "output_dir"],
                "additionalProperties": False
            }
        },
        {
            "name": "analyze_patch_statistics",
            "title": "Analyze Patch Statistics",
            "description": "Computes colour variance and patch entropy.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "patch_dir": {"type": "string"},
                    "output_path": {"type": "string"}
                },
                "required": ["approval_token", "patch_dir"],
                "additionalProperties": False
            }
        },
        {
            "name": "predict_kather_resnet18",
            "title": "Predict WSI Tissue Classes With ResNet18-Kather100K",
            "description": "Runs pretrained resnet18-kather100k directly on a WSI and saves a TIAViz-compatible annotationstore. Patch-folder mode is retained as fallback.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "save_dir": {"type": "string"},
                    "ioconfig": {"type": "object"},
                    "output_type": {"type": "string"},
                    "patch_mode": {"type": "boolean"},
                    "patch_dir": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "model_name": {"type": "string"},
                    "batch_size": {"type": "integer"},
                    "device": {"type": "string"},
                    "input_size": {"type": "integer"},
                    "run_async": {"type": "boolean"},
                    "wait": {"type": "boolean"}
                },
                "required": ["approval_token"],
                "additionalProperties": False
            }
        },
        {
            "name": "predict_kongnet_nucleus_detection",
            "title": "Detect Nuclei With KongNet PanNuke",
            "description": "Runs TIAToolbox KongNet_PanNuke_1 nucleus detection on a WSI and saves a TIAViz-compatible annotationstore.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "save_dir": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "model_name": {"type": "string"},
                    "batch_size": {"type": "integer"},
                    "device": {"type": "string"},
                    "output_type": {"type": "string"},
                    "patch_mode": {"type": "boolean"},
                    "auto_get_mask": {"type": "boolean"},
                    "num_workers": {"type": "integer"},
                    "overwrite": {"type": "boolean"},
                    "run_async": {"type": "boolean"},
                    "wait": {"type": "boolean"}
                },
                "required": ["approval_token", "wsi_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "check_prediction_job",
            "title": "Check Background Prediction Job",
            "description": "Checks a background prediction job started by predict_kather_resnet18 or predict_kongnet_nucleus_detection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "job_id": {"type": "string"},
                    "status_path": {"type": "string"},
                    "search_dir": {"type": "string"}
                },
                "required": ["approval_token"],
                "additionalProperties": False
            }
        },
        {
            "name": "export_kongnet_nuclei_to_csv",
            "title": "Export KongNet Nuclei To CSV",
            "description": "Exports nucleus IDs, centroids, classes, probabilities, and optional physical coordinates from a KongNet AnnotationStore.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "min_probability": {"type": "number"},
                    "cell_types": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["approval_token", "annotationstore_path", "output_csv_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "find_cells_within_radius",
            "title": "Find Cells Within Radius",
            "description": "Counts selected target nuclei around each source nucleus using pixel or physical distances.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "radius": {"type": "number"},
                    "distance_units": {"type": "string", "enum": ["microns", "pixels"]},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "source_types": {"type": "array", "items": {"type": "string"}},
                    "target_types": {"type": "array", "items": {"type": "string"}},
                    "min_probability": {"type": "number"}
                },
                "required": ["approval_token", "annotationstore_path", "output_csv_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_cell_type_cooccurrence",
            "title": "Compute Cell Type Co-occurrence",
            "description": "Computes an undirected cell-type pair matrix within a configurable radius.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "radius": {"type": "number"},
                    "distance_units": {"type": "string", "enum": ["microns", "pixels"]},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "cell_types": {"type": "array", "items": {"type": "string"}},
                    "min_probability": {"type": "number"}
                },
                "required": ["approval_token", "annotationstore_path", "output_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_nearest_neighbour_features",
            "title": "Compute Nearest Neighbour Features",
            "description": "Finds each source nucleus's nearest selected target and summarizes distances by cell-type pair.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "distance_units": {"type": "string", "enum": ["microns", "pixels"]},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "source_types": {"type": "array", "items": {"type": "string"}},
                    "target_types": {"type": "array", "items": {"type": "string"}},
                    "min_probability": {"type": "number"}
                },
                "required": ["approval_token", "annotationstore_path", "output_csv_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "aggregate_kather_metrics",
            "title": "Aggregate Kather Metrics",
            "description": "Aggregates Kather predictions into class percentages, abnormality percentage, entropy, and clusters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "predictions_json_path": {"type": "string"},
                    "output_metrics_path": {"type": "string"},
                    "abnormality_threshold": {"type": "number"},
                    "cluster_distance": {"type": "number"}
                },
                "required": ["approval_token", "predictions_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "summarize_kather_results",
            "title": "Summarize Kather Results",
            "description": "Reads saved Kather predictions, metrics, and patch statistics and produces a plain-English interpretation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "predictions_json_path": {"type": "string"},
                    "metrics_json_path": {"type": "string"},
                    "patch_statistics_json_path": {"type": "string"},
                    "output_summary_path": {"type": "string"}
                },
                "required": ["approval_token", "predictions_json_path", "metrics_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "generate_confidence_histogram",
            "title": "Generate Confidence Histogram",
            "description": "Creates a histogram showing patch-level prediction confidence distribution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "predictions_json_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "bins": {"type": "integer"}
                },
                "required": ["approval_token", "predictions_json_path", "output_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "compare_masked_vs_unmasked_runs",
            "title": "Compare Masked vs Unmasked Runs",
            "description": "Compares saved Kather metric files from masked and unmasked analysis runs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "masked_metrics_path": {"type": "string"},
                    "unmasked_metrics_path": {"type": "string"},
                    "output_path": {"type": "string"}
                },
                "required": ["approval_token", "masked_metrics_path", "unmasked_metrics_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "threshold_sensitivity_analysis",
            "title": "Threshold Sensitivity Analysis",
            "description": "Evaluates abnormality percentage and cluster count across multiple abnormality thresholds.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "predictions_json_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "output_plot_path": {"type": "string"},
                    "thresholds": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "required": ["approval_token", "predictions_json_path", "output_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "extract_top_abnormal_patches",
            "title": "Extract Top Abnormal Patches",
            "description": "Copies the top-K highest abnormality-score patches and optionally creates CSV/grid outputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "predictions_json_path": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "output_csv_path": {"type": "string"},
                    "output_grid_path": {"type": "string"}
                },
                "required": ["approval_token", "predictions_json_path", "output_dir"],
                "additionalProperties": False
            }
        },
        {
            "name": "generate_final_ai_report",
            "title": "Generate Final AI Report",
            "description": "Generates a final structured AI interpretability report from saved outputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "predictions_json_path": {"type": "string"},
                    "metrics_json_path": {"type": "string"},
                    "patch_statistics_json_path": {"type": "string"},
                    "threshold_sensitivity_json_path": {"type": "string"},
                    "output_report_path": {"type": "string"}
                },
                "required": ["approval_token", "predictions_json_path", "metrics_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "generate_kongnet_ai_report",
            "title": "Generate KongNet AI Interpretability Report",
            "description": "Generates and saves a plain-text (.txt) KongNet nucleus-composition, confidence, and spatial interpretability report. Do not use save_run_report for this artifact.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "nuclei_csv_path": {"type": "string"},
                    "cooccurrence_json_path": {"type": "string"},
                    "neighbourhood_json_path": {"type": "string"},
                    "nearest_neighbour_json_path": {"type": "string"},
                    "output_report_path": {
                        "type": "string",
                        "description": "Destination text file. A .txt suffix is enforced even if another extension is supplied."
                    }
                },
                "required": ["approval_token", "nuclei_csv_path"],
                "additionalProperties": False
            }
        },
        
        {
            "name": "save_run_report",
            "title": "Save Run Report",
            "description": "Saves a JSON run report to disk.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "output_path": {"type": "string"},
                    "report": {}
                },
                "required": ["approval_token", "output_path", "report"],
                "additionalProperties": False
            }
        }
    ]

    send({
        "jsonrpc": "2.0",
        "id": req["id"],
        "result": {"tools": tools}
    })


def handle_tools_call(req: Dict[str, Any]) -> None:
    req_id = req["id"]
    params = req.get("params", {})
    name = params.get("name")
    args = params.get("arguments", {}) or {}

    try:
        if name == "propose_pathology_plan":
            user_prompt = args.get("user_prompt")
            wsi_path = args.get("wsi_path")
            output_dir = args.get("output_dir")

            if not isinstance(user_prompt, str) or not user_prompt.strip():
                tool_error(req_id, 'propose_pathology_plan requires "user_prompt".')
                return

            if not isinstance(wsi_path, str) or not wsi_path.strip():
                tool_error(req_id, 'propose_pathology_plan requires "wsi_path".')
                return

            if not isinstance(output_dir, str) or not output_dir.strip():
                tool_error(req_id, 'propose_pathology_plan requires "output_dir".')
                return

            tool_result(
                req_id,
                build_plan(
                    user_prompt=user_prompt.strip(),
                    wsi_path=wsi_path.strip(),
                    output_dir=output_dir.strip()
                )
            )
            return

        if name == "health":
            tool_result(req_id, tool_health())
            return

        if name == "echo":
            tool_result(req_id, tool_echo(args.get("text", "")))
            return

        if name == "list_files":
            tool_result(
                req_id,
                tool_list_files(
                    directory=args.get("directory", ""),
                    max_items=int(args.get("max_items", 50))
                )
            )
            return

        if name == "wsi_metadata":
            require_plan(args)
            tool_result(req_id, tool_wsi_metadata(args.get("path", "")))
            return

        if name == "wsi_thumbnail":
            require_plan(args)
            tool_result(
                req_id,
                tool_wsi_thumbnail(
                    path=args.get("path", ""),
                    output_path=args.get("output_path", ""),
                    resolution=float(args.get("resolution", 2.0)),
                    units=str(args.get("units", "power"))
                )
            )
            return

        if name == "tissue_mask":
            require_plan(args)
            tool_result(
                req_id,
                tool_tissue_mask(
                    path=args.get("path", ""),
                    output_mask_path=args.get("output_mask_path", ""),
                    output_overlay_path=args.get("output_overlay_path"),
                    mpp=float(args.get("mpp", 2.0)),
                    method=str(args.get("method", "morphological"))
                )
            )
            return

        if name == "extract_patches":
            require_plan(args)

            stride = args.get("stride")

            tool_result(
                req_id,
                tool_extract_patches(
                    path=args.get("path", ""),
                    output_dir=args.get("output_dir", ""),
                    patch_size=int(args.get("patch_size", 224)),
                    stride=int(stride) if stride is not None else None,
                    level=int(args.get("level", 0)),
                    max_patches=int(args.get("max_patches", 2000)),
                    min_tissue_fraction=float(args.get("min_tissue_fraction", 0.15)),
                    mpp=float(args.get("mpp", 2.0))
                )
            )
            return

        if name == "analyze_patch_statistics":
            require_plan(args)
            tool_result(
                req_id,
                tool_analyze_patch_statistics(
                    patch_dir=args.get("patch_dir", ""),
                    output_path=args.get("output_path")
                )
            )
            return

        if name == "predict_kather_resnet18":
            require_plan(args)

            run_async = bool_arg(args.get("run_async"), True)
            wait = bool_arg(args.get("wait"), False)

            if run_async and not wait:
                tool_result(req_id, start_prediction_job(args))
            else:
                tool_result(
                    req_id,
                    tool_predict_kather_resnet18(
                        patch_dir=args.get("patch_dir"),
                        output_json_path=args.get("output_json_path"),
                        output_csv_path=args.get("output_csv_path"),
                        model_name=str(args.get("model_name", "resnet18-kather100k")),
                        batch_size=int(args.get("batch_size", 64)),
                        device=str(args.get("device", "auto")),
                        input_size=int(args.get("input_size", 224)),
                        wsi_path=args.get("wsi_path"),
                        save_dir=args.get("save_dir"),
                        ioconfig=args.get("ioconfig"),
                        output_type=str(args.get("output_type", "annotationstore")),
                        patch_mode=bool_arg(args.get("patch_mode"), False),
                    )
                )
            return

        if name == "predict_kongnet_nucleus_detection":
            require_plan(args)

            run_async = bool_arg(args.get("run_async"), True)
            wait = bool_arg(args.get("wait"), False)

            if run_async and not wait:
                job_args = dict(args)
                job_args["job_tool"] = "predict_kongnet_nucleus_detection"
                tool_result(req_id, start_prediction_job(job_args))
            else:
                num_workers = args.get("num_workers")
                tool_result(
                    req_id,
                    tool_predict_kongnet_nucleus_detection(
                        wsi_path=args.get("wsi_path", ""),
                        output_json_path=args.get("output_json_path"),
                        model_name=str(args.get("model_name", "KongNet_PanNuke_1")),
                        batch_size=int(args.get("batch_size", 16)),
                        device=str(args.get("device", "auto")),
                        save_dir=args.get("save_dir"),
                        output_type=str(args.get("output_type", "annotationstore")),
                        patch_mode=bool_arg(args.get("patch_mode"), False),
                        auto_get_mask=bool_arg(args.get("auto_get_mask"), False),
                        num_workers=int(num_workers) if num_workers is not None else None,
                        overwrite=bool_arg(args.get("overwrite"), True),
                    )
                )
            return

        if name == "check_prediction_job":
            require_plan(args)
            tool_result(
                req_id,
                load_prediction_job_status(
                    job_id=optional_str(args.get("job_id")),
                    status_path=optional_str(args.get("status_path")),
                    search_dir=optional_str(args.get("search_dir")),
                )
            )
            return

        if name == "export_kongnet_nuclei_to_csv":
            require_plan(args)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_export_kongnet_nuclei_to_csv(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_csv_path=args.get("output_csv_path", ""),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    min_probability=float(args.get("min_probability", 0.0)),
                    cell_types=args.get("cell_types"),
                )
            )
            return

        if name == "find_cells_within_radius":
            require_plan(args)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_find_cells_within_radius(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_csv_path=args.get("output_csv_path", ""),
                    output_json_path=args.get("output_json_path"),
                    radius=float(args.get("radius", 50.0)),
                    distance_units=str(args.get("distance_units", "microns")),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    source_types=args.get("source_types"),
                    target_types=args.get("target_types"),
                    min_probability=float(args.get("min_probability", 0.0)),
                )
            )
            return

        if name == "compute_cell_type_cooccurrence":
            require_plan(args)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_compute_cell_type_cooccurrence(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_csv_path=args.get("output_csv_path"),
                    radius=float(args.get("radius", 50.0)),
                    distance_units=str(args.get("distance_units", "microns")),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    cell_types=args.get("cell_types"),
                    min_probability=float(args.get("min_probability", 0.0)),
                )
            )
            return

        if name == "compute_nearest_neighbour_features":
            require_plan(args)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_compute_nearest_neighbour_features(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_csv_path=args.get("output_csv_path", ""),
                    output_json_path=args.get("output_json_path"),
                    distance_units=str(args.get("distance_units", "microns")),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    source_types=args.get("source_types"),
                    target_types=args.get("target_types"),
                    min_probability=float(args.get("min_probability", 0.0)),
                )
            )
            return

        if name == "aggregate_kather_metrics":
            require_plan(args)
            cluster_distance = args.get("cluster_distance")

            tool_result(
                req_id,
                tool_aggregate_kather_metrics(
                    predictions_json_path=args.get("predictions_json_path", ""),
                    output_metrics_path=args.get("output_metrics_path"),
                    abnormality_threshold=float(args.get("abnormality_threshold", 0.5)),
                    cluster_distance=float(cluster_distance) if cluster_distance is not None else None,
                )
            )
            return


        if name == "summarize_kather_results":
            require_plan(args)
            tool_result(
                req_id,
                tool_summarize_kather_results(
                    predictions_json_path=args.get("predictions_json_path", ""),
                    metrics_json_path=args.get("metrics_json_path", ""),
                    patch_statistics_json_path=args.get("patch_statistics_json_path"),
                    output_summary_path=args.get("output_summary_path"),
                )
            )
            return

        if name == "generate_confidence_histogram":
            require_plan(args)
            tool_result(
                req_id,
                tool_generate_confidence_histogram(
                    predictions_json_path=args.get("predictions_json_path", ""),
                    output_path=args.get("output_path", ""),
                    bins=int(args.get("bins", 20)),
                )
            )
            return


        if name == "compare_masked_vs_unmasked_runs":
            require_plan(args)
            tool_result(
                req_id,
                tool_compare_masked_vs_unmasked_runs(
                    masked_metrics_path=args.get("masked_metrics_path", ""),
                    unmasked_metrics_path=args.get("unmasked_metrics_path", ""),
                    output_path=args.get("output_path"),
                )
            )
            return


        if name == "threshold_sensitivity_analysis":
            require_plan(args)
            tool_result(
                req_id,
                tool_threshold_sensitivity_analysis(
                    predictions_json_path=args.get("predictions_json_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_csv_path=args.get("output_csv_path"),
                    output_plot_path=args.get("output_plot_path"),
                    thresholds=args.get("thresholds"),
                )
            )
            return

        if name == "extract_top_abnormal_patches":
            require_plan(args)
            tool_result(
                req_id,
                tool_extract_top_abnormal_patches(
                    predictions_json_path=args.get("predictions_json_path", ""),
                    output_dir=args.get("output_dir", ""),
                    top_k=int(args.get("top_k", 20)),
                    output_csv_path=args.get("output_csv_path"),
                    output_grid_path=args.get("output_grid_path"),
                )
            )
            return

        if name == "generate_final_ai_report":
            require_plan(args)
            tool_result(
                req_id,
                tool_generate_final_ai_report(
                    predictions_json_path=args.get("predictions_json_path", ""),
                    metrics_json_path=args.get("metrics_json_path", ""),
                    patch_statistics_json_path=args.get("patch_statistics_json_path"),
                    threshold_sensitivity_json_path=args.get("threshold_sensitivity_json_path"),
                    output_report_path=args.get("output_report_path"),
                )
            )
            return

        if name == "generate_kongnet_ai_report":
            require_plan(args)
            tool_result(
                req_id,
                tool_generate_kongnet_ai_report(
                    nuclei_csv_path=args.get("nuclei_csv_path", ""),
                    cooccurrence_json_path=args.get("cooccurrence_json_path"),
                    neighbourhood_json_path=args.get("neighbourhood_json_path"),
                    nearest_neighbour_json_path=args.get("nearest_neighbour_json_path"),
                    output_report_path=args.get("output_report_path"),
                )
            )
            return

        

        if name == "save_run_report":
            require_plan(args)

            output_path = args.get("output_path")
            report = args.get("report")

            if not isinstance(output_path, str) or not output_path.strip():
                tool_error(req_id, 'save_run_report requires "output_path".')
                return

            output_name = os.path.basename(output_path).casefold()
            if "kongnet" in output_name and "interpretability" in output_name:
                tool_error(
                    req_id,
                    "KongNet interpretability reports must be created with "
                    "generate_kongnet_ai_report, which saves a plain-text .txt file. "
                    "save_run_report is only for JSON run metadata.",
                )
                return

            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            tool_result(req_id, {"status": "saved", "output_path": output_path})
            return

        jsonrpc_error(req_id, -32602, f"Unknown tool: {name}")

    except Exception as e:
        tool_error(req_id, str(e), {"trace": traceback.format_exc()})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        try:
            req = json.loads(line)
            method = req.get("method")
            req_id = req.get("id")

            if method == "initialize":
                handle_initialize(req)

            elif method == "notifications/initialized":
                pass

            elif method == "tools/list":
                handle_tools_list(req)

            elif method == "tools/call":
                handle_tools_call(req)

            else:
                if req_id is not None:
                    jsonrpc_error(req_id, -32601, f"Method not found: {method}")

        except Exception as e:
            print("Server error:\n" + traceback.format_exc(), file=sys.stderr)

            try:
                maybe_req = json.loads(line)
                if "id" in maybe_req:
                    jsonrpc_error(
                        maybe_req["id"],
                        -32603,
                        "Internal error",
                        {"detail": str(e)}
                    )
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--run-predict-job":
        raise SystemExit(run_predict_job(sys.argv[2], sys.argv[3], sys.argv[4]))

    main()
