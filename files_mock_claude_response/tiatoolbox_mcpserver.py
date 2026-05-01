import sys
import json
import traceback
import os
import uuid
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
    tool_aggregate_kather_metrics,
    tool_generate_kather_overlay,
    tool_summarize_kather_results,
    tool_generate_confidence_histogram,
    tool_generate_hotspot_overlay,
    tool_compare_masked_vs_unmasked_runs,
)

PROTOCOL_VERSION = "2025-06-18"

PENDING_PLANS: Dict[str, Dict[str, Any]] = {}


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
        "post",
        "summary",
        "hotspot",
        "histogram",
        "compare",
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
        "post",
        "summary",
        "hotspot",
        "histogram",
        "compare",
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
        "post",
        "summary",
        "hotspot",
        "histogram",
        "compare",
    ]):
        return "patches_only"

    if (
        "summarize" in request
        or "summary" in request
        or "post-process" in request
        or "postprocess" in request
        or "hotspot" in request
        or "histogram" in request
        or "confidence" in request
        or "compare masked" in request
        or "compare" in request
        or "masked vs unmasked" in request
    ):
        return "post_processing"

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
                "wsi_thumbnail",
                "tissue_mask"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "thumbnail.png"),
                os.path.join(output_dir, "tissue_mask.png"),
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
                "wsi_thumbnail",
                "tissue_mask",
                "extract_patches"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "thumbnail.png"),
                os.path.join(output_dir, "tissue_mask.png"),
                os.path.join(output_dir, "tissue_overlay.png"),
                os.path.join(output_dir, "patches")
            ]
        }

    elif task_type == "post_processing":
        plan = {
            "task_type": task_type,
            "goal": "Run post-processing on existing Kather100K prediction outputs.",
            "steps": [
                "Use saved Kather prediction, metric, and patch-statistic files.",
                "Generate a plain-English summary of the Kather results.",
                "Generate a confidence histogram from patch-level prediction confidence scores.",
                "Generate a hotspot overlay showing spatial clusters of high-abnormality patches.",
                "Optionally compare masked and unmasked metric files if both are provided.",
                "Save all post-processing outputs to the output directory."
            ],
            "suggested_tools_after_approval": [
                "summarize_kather_results",
                "generate_confidence_histogram",
                "generate_hotspot_overlay",
                "compare_masked_vs_unmasked_runs"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kather_summary.txt"),
                os.path.join(output_dir, "confidence_histogram.png"),
                os.path.join(output_dir, "hotspot_overlay.png"),
                os.path.join(output_dir, "masked_vs_unmasked_comparison.txt")
            ],
            "default_parameters": {
                "abnormality_threshold": 0.5,
                "histogram_bins": 20,
                "patch_size": 224
            },
            "clinical_warning": (
                "The post-processing outputs explain model-confidence behaviour, not clinical diagnosis."
            )
        }

    elif task_type == "kather_prediction":
        plan = {
            "task_type": task_type,
            "goal": "Run ResNet18-Kather100K patch-level tissue classification and generate interpretable outputs.",
            "steps": [
                "Read WSI metadata.",
                "Generate a thumbnail.",
                "Generate a tissue mask and tissue overlay unless the user asks to skip masking.",
                "Extract dense tissue-rich 224x224 patches suitable for Kather100K inference.",
                "Run pretrained ResNet18-Kather100K classification on extracted patches.",
                "Aggregate predictions into tissue-class percentages.",
                "Estimate tumour-relevant indicators using TUM, STR, and abnormality score.",
                "Compute class entropy, high-abnormality percentage, and cluster count.",
                "Compute colour variance and grayscale entropy from patches.",
                "Generate tissue-class and heterogeneity overlays.",
                "Run post-processing: summary, confidence histogram, and hotspot overlay.",
                "Save all outputs and a run report."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "wsi_thumbnail",
                "tissue_mask",
                "extract_patches",
                "predict_kather_resnet18",
                "aggregate_kather_metrics",
                "analyze_patch_statistics",
                "generate_kather_overlay",
                "summarize_kather_results",
                "generate_confidence_histogram",
                "generate_hotspot_overlay",
                "save_run_report"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "thumbnail.png"),
                os.path.join(output_dir, "tissue_mask.png"),
                os.path.join(output_dir, "tissue_overlay.png"),
                os.path.join(output_dir, "patches"),
                os.path.join(output_dir, "kather_predictions.json"),
                os.path.join(output_dir, "kather_prediction_table.csv"),
                os.path.join(output_dir, "kather_metrics.json"),
                os.path.join(output_dir, "patch_statistics.json"),
                os.path.join(output_dir, "kather_class_overlay.png"),
                os.path.join(output_dir, "heterogeneity_overlay.png"),
                os.path.join(output_dir, "kather_summary.txt"),
                os.path.join(output_dir, "confidence_histogram.png"),
                os.path.join(output_dir, "hotspot_overlay.png"),
                os.path.join(output_dir, "run_report.json")
            ],
            "default_parameters": {
                "model_name": "resnet18-kather100k",
                "patch_size": 224,
                "stride": 112,
                "max_patches": 2000,
                "min_tissue_fraction": 0.15,
                "abnormality_threshold": 0.5,
                "histogram_bins": 20
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
                "wsi_thumbnail",
                "tissue_mask",
                "extract_patches",
                "analyze_patch_statistics"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "thumbnail.png"),
                os.path.join(output_dir, "tissue_mask.png"),
                os.path.join(output_dir, "tissue_overlay.png"),
                os.path.join(output_dir, "patches"),
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
                "version": "12.0.0-kather-postprocessing"
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
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        },
        {
            "name": "echo",
            "title": "Echo",
            "description": "Echoes back text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
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
                    "method": {
                        "type": "string",
                        "enum": ["morphological", "otsu"]
                    }
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
            "title": "Predict Tissue Classes With ResNet18-Kather100K",
            "description": "Runs pretrained resnet18-kather100k over extracted patches.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "patch_dir": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "model_name": {"type": "string"},
                    "batch_size": {"type": "integer"},
                    "device": {"type": "string"},
                    "input_size": {"type": "integer"}
                },
                "required": ["approval_token", "patch_dir", "output_json_path"],
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
            "name": "generate_kather_overlay",
            "title": "Generate Kather Overlay",
            "description": "Generates visible tissue-class and heterogeneity overlays.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "predictions_json_path": {"type": "string"},
                    "thumbnail_path": {"type": "string"},
                    "output_overlay_path": {"type": "string"},
                    "output_heterogeneity_path": {"type": "string"},
                    "alpha": {"type": "number"},
                    "patch_size": {"type": "integer"},
                    "min_display_size": {"type": "integer"},
                    "draw_legend": {"type": "boolean"}
                },
                "required": [
                    "approval_token",
                    "wsi_path",
                    "predictions_json_path",
                    "thumbnail_path",
                    "output_overlay_path"
                ],
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
                "required": [
                    "approval_token",
                    "predictions_json_path",
                    "metrics_json_path"
                ],
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
                "required": [
                    "approval_token",
                    "predictions_json_path",
                    "output_path"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "generate_hotspot_overlay",
            "title": "Generate Hotspot Overlay",
            "description": "Draws boxes around spatial clusters of high-abnormality patches.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "predictions_json_path": {"type": "string"},
                    "thumbnail_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "abnormality_threshold": {"type": "number"},
                    "patch_size": {"type": "integer"},
                    "min_display_size": {"type": "integer"}
                },
                "required": [
                    "approval_token",
                    "wsi_path",
                    "predictions_json_path",
                    "thumbnail_path",
                    "output_path"
                ],
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
                "required": [
                    "approval_token",
                    "masked_metrics_path",
                    "unmasked_metrics_path"
                ],
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
        "result": {
            "tools": tools
        }
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

            tool_result(
                req_id,
                tool_predict_kather_resnet18(
                    patch_dir=args.get("patch_dir", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_csv_path=args.get("output_csv_path"),
                    model_name=str(args.get("model_name", "resnet18-kather100k")),
                    batch_size=int(args.get("batch_size", 16)),
                    device=str(args.get("device", "auto")),
                    input_size=int(args.get("input_size", 224)),
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

        if name == "generate_kather_overlay":
            require_plan(args)

            tool_result(
                req_id,
                tool_generate_kather_overlay(
                    wsi_path=args.get("wsi_path", ""),
                    predictions_json_path=args.get("predictions_json_path", ""),
                    thumbnail_path=args.get("thumbnail_path", ""),
                    output_overlay_path=args.get("output_overlay_path", ""),
                    output_heterogeneity_path=args.get("output_heterogeneity_path"),
                    alpha=float(args.get("alpha", 0.75)),
                    patch_size=int(args.get("patch_size", 224)),
                    min_display_size=int(args.get("min_display_size", 8)),
                    draw_legend=bool(args.get("draw_legend", True)),
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

        if name == "generate_hotspot_overlay":
            require_plan(args)

            tool_result(
                req_id,
                tool_generate_hotspot_overlay(
                    wsi_path=args.get("wsi_path", ""),
                    predictions_json_path=args.get("predictions_json_path", ""),
                    thumbnail_path=args.get("thumbnail_path", ""),
                    output_path=args.get("output_path", ""),
                    abnormality_threshold=float(args.get("abnormality_threshold", 0.5)),
                    patch_size=int(args.get("patch_size", 224)),
                    min_display_size=int(args.get("min_display_size", 10)),
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

        if name == "save_run_report":
            require_plan(args)

            output_path = args.get("output_path")
            report = args.get("report")

            if not isinstance(output_path, str) or not output_path.strip():
                tool_error(req_id, 'save_run_report requires "output_path".')
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
    main()
