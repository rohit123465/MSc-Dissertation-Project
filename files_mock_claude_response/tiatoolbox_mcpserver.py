import sys
import json
import traceback
import os
import uuid
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from tia_tools import (
    tool_health,
    tool_echo,
    tool_list_files,
    tool_wsi_metadata,
    tool_validate_qupath_roi_pair,
    tool_wsi_thumbnail,
    tool_tissue_mask,
    tool_extract_patches,
    tool_analyze_patch_statistics,
    tool_predict_patch_model,
    tool_predict_kather_resnet18,
    tool_predict_kongnet_nucleus_detection,
    tool_predict_nucleus_instance_segmentation,
    tool_predict_multi_task_segmentation,
    tool_predict_semantic_segmentation,
    tool_export_kongnet_nuclei_to_csv,
    tool_find_cells_within_radius,
    tool_compute_cell_type_cooccurrence,
    tool_compute_nearest_neighbour_features,
    tool_analyze_kongnet_regions,
    tool_compute_kongnet_point_pattern_statistics,
    tool_compute_kongnet_morans_i,
    tool_compute_kongnet_local_morans_i,
    tool_compute_kongnet_spatial_entropy,
    tool_compute_kongnet_cross_g_function,
    tool_export_kongnet_regions_to_annotationstore,
    tool_characterize_kongnet_cell_neighbourhoods,
    tool_aggregate_kather_metrics,
    tool_summarize_kather_results,
    tool_generate_confidence_histogram,
    tool_compare_masked_vs_unmasked_runs,
    tool_threshold_sensitivity_analysis,
    tool_extract_top_abnormal_patches,
    tool_generate_final_ai_report,
    tool_generate_kongnet_ai_report,
    tool_generate_nucleus_instance_segmentation_report,
    tool_run_kongnet_spatial_workflow,
    tool_rank_kongnet_regions,
    tool_answer_kongnet_spatial_question,
    tool_generate_kongnet_region_heatmaps,
    tool_generate_kongnet_point_pattern_overlays,
    tool_generate_kongnet_slide_summary,
    tool_build_common_spatial_features,
    tool_validate_spatial_capabilities,
    tool_compute_common_roi_morans_i,
    tool_compute_common_roi_entropy,
    tool_compute_common_point_pattern_statistics,
    tool_compute_common_cross_g,
    tool_compute_common_cooccurrence,
    tool_compute_common_neighbour_distances,
    PATCH_PREDICTION_MODEL_CATALOG,
    _patch_prediction_model_summary,
    KONGNET_MODEL_CATALOG,
    _kongnet_model_summary,
    NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG,
    _nucleus_instance_segmentation_model_summary,
    MULTI_TASK_SEGMENTATION_MODEL_CATALOG,
    _multi_task_segmentation_model_summary,
    SEMANTIC_SEGMENTATION_MODEL_CATALOG,
    _semantic_segmentation_model_summary,
)

PROTOCOL_VERSION = "2025-06-18"

PENDING_PLANS: Dict[str, Dict[str, Any]] = {}
APPROVED_PLANS: Dict[str, Dict[str, Any]] = {}


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
    if job_tool == "predict_kongnet_nucleus_detection":
        default_model_name = "KongNet_PanNuke_1"
        default_batch_size = 16
    elif job_tool == "predict_nucleus_instance_segmentation":
        default_model_name = "hovernet_fast-monusac"
        default_batch_size = 8
    elif job_tool == "predict_multi_task_segmentation":
        default_model_name = "hovernetplus-oed"
        default_batch_size = 8
    elif job_tool == "predict_semantic_segmentation":
        default_model_name = "fcn_resnet50_unet-bcss"
        default_batch_size = 8
    else:
        default_model_name = "resnet18-kather100k"
        default_batch_size = 64

    job_args = {
        "patch_dir": args.get("patch_dir"),
        "output_json_path": args.get("output_json_path"),
        "output_csv_path": args.get("output_csv_path"),
        "model_name": str(args.get("model_name", default_model_name)),
        "batch_size": int(args.get("batch_size", default_batch_size)),
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
        elif job_tool == "predict_nucleus_instance_segmentation":
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
            result = tool_predict_nucleus_instance_segmentation(
                **{key: value for key, value in job_args.items() if key in allowed}
            )
        elif job_tool == "predict_multi_task_segmentation":
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
            result = tool_predict_multi_task_segmentation(
                **{key: value for key, value in job_args.items() if key in allowed}
            )
        elif job_tool == "predict_semantic_segmentation":
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
            result = tool_predict_semantic_segmentation(
                **{key: value for key, value in job_args.items() if key in allowed}
            )
        elif job_tool in {"predict_kather_resnet18", "predict_patch_model"}:
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
            result = tool_predict_patch_model(
                **{key: value for key, value in job_args.items() if key in allowed}
            )
        else:
            raise ValueError(f"Unsupported prediction job tool: {job_tool}")

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

    non_kongnet_spatial_models = [
        "hovernet", "hover-net", "hovernetplus", "hover-net plus",
        "bcss", "semantic segmentation", "kather", "pcam", "patch classification",
    ]
    requests_spatial_analysis = any(term in request for term in [
        "spatial analysis", "spatial statistics", "spatial statistic",
        "analyse spatial", "analyze spatial",
    ])
    requests_object_statistics = any(term in request for term in [
        "point pattern", "point-pattern", "ripley", "cross-g", "cross g",
        "co-occurrence", "cooccurrence", "nearest neighbour", "nearest-neighbour", "nni",
    ])
    if requests_object_statistics and any(model in request for model in non_kongnet_spatial_models + ["common object", "common-object"]):
        return "common_object_spatial_analysis"
    if ("common" in request or "cross-model" in request or "cross model" in request) and "moran" in request:
        return "common_roi_morans_i"
    if ("common" in request or "cross-model" in request or "cross model" in request) and "entropy" in request:
        return "common_roi_entropy"
    if any(term in request for term in ["common spatial format", "common spatial feature", "model-specific adapter", "cross-model spatial", "cross model spatial"]):
        return "common_spatial_adapter"
    if requests_spatial_analysis and any(model in request for model in non_kongnet_spatial_models):
        return "common_spatial_adapter"

    if any(k in request for k in [
        "validate roi pair",
        "validate the roi pair",
        "validate qupath roi",
        "roi-pair validation",
        "roi pair validation",
        "check roi geojson",
        "verify roi geojson",
    ]):
        return "roi_pair_validation"

    metadata_requested = any(k in request for k in [
        "metadata",
        "image dimensions",
        "level 0 dimensions",
        "microns-per-pixel",
        "microns per pixel",
        "mpp",
        "pyramid levels",
        "objective information",
        "objective power",
    ])
    execution_prohibited = any(k in request for k in [
        "do not start prediction",
        "do not run prediction",
        "do not predict",
        "no prediction",
        "metadata only",
        "only the metadata",
        "read only the metadata",
    ])
    analysis_requested = any(k in request for k in [
        "run prediction",
        "start prediction",
        "predict using",
        "run segmentation",
        "start segmentation",
        "extract patches",
        "run spatial",
    ])
    if metadata_requested and (execution_prohibited or not analysis_requested):
        return "metadata_only"

    requests_individual_spatial_tools = any(k in request for k in [
        "individual tools",
        "individual spatial tools",
        "per-tool route",
        "separate radii",
        "different radii",
        "exact radii",
    ])
    requests_multiple_spatial_operations = (
        any(k in request for k in ["co-occurrence", "cooccurrence"])
        and any(k in request for k in ["within radius", "within 25", "radius based", "radius-based"])
        and any(k in request for k in ["cell density", "density"])
    )
    if requests_individual_spatial_tools or requests_multiple_spatial_operations:
        return "custom_nucleus_spatial_analysis"

    if any(k in request for k in [
        "local moran",
        "local moran's i",
        "local morans i",
        "lisa cluster",
        "high-high roi",
        "low-low roi",
    ]):
        return "kongnet_local_morans_i_analysis"

    if any(k in request for k in [
        "moran",
        "moran's i",
        "morans i",
        "moran’s i",
        "spatial autocorrelation",
        "roi autocorrelation",
        "hotspot/coldspot",
        "hotspot coldspot",
        "coldspot",
    ]):
        return "kongnet_morans_i_analysis"

    if any(k in request for k in [
        "spatial entropy",
        "shannon entropy",
        "entropy",
        "composition entropy",
        "compositional entropy",
        "roi diversity",
        "cell diversity",
        "mixed microenvironment",
        "homogeneous region",
        "homogeneous roi",
        "heterogeneous roi",
        "heterogeneous region",
    ]):
        return "kongnet_spatial_entropy_analysis"

    if any(k in request for k in [
        "cross-g",
        "cross g",
        "cross-g function",
        "cross g function",
        "g-function",
        "g function",
        "gij",
        "g_i_j",
        "nearest target probability",
        "probability of immune cell nearby",
        "immune cell nearby",
        "immune exclusion",
        "immune-excluded",
        "source target proximity",
    ]):
        return "kongnet_cross_g_analysis"

    if any(k in request for k in [
        "show me tumour-immune",
        "show me tumor-immune",
        "top inflammatory region",
        "top neoplastic region",
        "top tumour-immune",
        "top tumor-immune",
        "most inflammatory region",
        "most tumour-rich region",
        "most tumor-rich region",
        "most interactive region",
        "spatial pathology question",
    ]):
        return "spatial_pathology_question"

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
        "tumour-immune",
        "tumor-immune",
        "inflammatory region",
        "neoplastic region",
        "stromal region",
        "epithelial region",
        "rank region",
        "top inflammatory",
        "top neoplastic",
        "spatial pathology question",
        "region heatmap",
        "interaction heatmap",
        "density heatmap",
        "pointpats",
        "point pattern",
        "ripley",
        "ripley's",
        "quadrat",
        "clustered-cell roi overlay",
        "clustered cell roi overlay",
        "nni heatmap",
        "quadrat vmr heatmap",
        "ripley clustering",
        "point-pattern overlay",
        "point pattern overlay",
        "spatial randomness",
        "nearest-neighbour index",
        "nearest-neighbor index",
        "csr",
        "slide summary",
        "spatial workflow",
        "full kongnet workflow",
    ]):
        return "nucleus_spatial_analysis"

    if any(k in request for k in [
        "semantic segmentation",
        "semantic segment",
        "fcn_resnet50_unet",
        "fcn_resnet50_unet-bcss",
        "bcss",
        "breast cancer segmentation",
        "segment tumor stroma necrosis",
        "segment tumour stroma necrosis",
    ]):
        return "semantic_segmentation"

    if any(k in request for k in [
        "hovernetplus",
        "hovernet plus",
        "hovernetplus-oed",
        "multi-task segmentation",
        "multi task segmentation",
        "output region",
        "region class",
        "oral epithelial dysplasia",
        "oed",
    ]):
        return "multi_task_segmentation"

    if any(k in request for k in [
        "hovernet",
        "hover-net",
        "monusac",
        "consep",
        "kumar",
        "pannuke hovernet",
        "hovernet pannuke",
        "hovernet_fast-pannuke",
        "hovernet_original-consep",
        "hovernet_original-kumar",
        "instance segmentation",
        "nucleus instance segmentation",
        "nuclei instance segmentation",
    ]):
        return "nucleus_instance_segmentation"

    if any(k in request for k in [
        "kongnet",
        "nucleus",
        "nuclei",
        "nuclear",
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
        or "pcam" in request
        or "wide_resnet50_2" in request
        or "wide_resnet50_2-pcam" in request
        or "classify" in request
        or "metastatic tissue" in request
        or "metastasis" in request
        or "lymph node" in request
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


def build_plan(
    user_prompt: str,
    wsi_path: str,
    output_dir: str,
    geojson_path: str = "",
    threshold_parameters: Optional[Dict[str, Any]] = None,
    feature_parameters: Optional[Dict[str, Any]] = None,
    clarification_plan_id: Optional[str] = None,
) -> Dict[str, Any]:
    task_type = infer_task_type(user_prompt)
    plan_id = str(uuid.uuid4())
    slide_stem = os.path.splitext(os.path.basename(wsi_path))[0] if wsi_path else ""
    slide_prefix = f"{slide_stem}_" if slide_stem else ""

    # Spatial statistics are sensitive to user-defined distance, confidence,
    # classification, and significance thresholds.  Defaults may be useful as
    # suggestions, but they must not silently become analysis choices.  The
    # questions below are attached to the draft plan so the client LLM asks the
    # user to confirm them before requesting approval.
    spatial_threshold_questions = {
        "common_object_spatial_analysis": [
            ("radii", "At which radii should Ripley, point-pattern, and Cross-G statistics be evaluated?"),
            ("cooccurrence_radius", "What radius should be used for cell-type co-occurrence?"),
            ("neighbourhood_radius", "What radius should be used for source-to-target neighbourhood counts?"),
            ("min_probability", "What minimum object-class probability should be retained?"),
            ("min_points_per_pattern", "What minimum number of points is required per class for point-pattern statistics?"),
        ],
        "common_spatial_adapter": [
            ("region_size", "What common ROI/grid size should be used?"),
            ("min_probability", "What minimum prediction-probability threshold should be used?"),
            ("distance_units", "Should coordinates and grid size use pixels or microns?"),
        ],
        "common_roi_morans_i": [
            ("alpha", "What statistical-significance threshold (alpha) should be used for Moran's I?"),
        ],
        "common_roi_entropy": [
            ("low_threshold", "What upper threshold should define low normalized spatial entropy?"),
            ("high_threshold", "What lower threshold should define high normalized spatial entropy?"),
        ],
        "custom_nucleus_spatial_analysis": [
            ("cooccurrence_radius", "What co-occurrence distance threshold should be used?"),
            ("neighbourhood_radius", "What neighbourhood distance threshold should be used?"),
            ("distance_units", "Which distance units should be used: microns or pixels?"),
            ("min_probability", "What minimum nucleus-class probability threshold should be used?"),
        ],
        "kongnet_morans_i_analysis": [
            ("alpha", "What statistical-significance threshold (alpha) should be used for Moran's I?"),
        ],
        "kongnet_local_morans_i_analysis": [
            ("distance_threshold", "What ROI-centroid radius should define neighbours for Local Moran's I?"),
            ("alpha", "What statistical-significance threshold (alpha) should be used for Local Moran's I?"),
        ],
        "kongnet_spatial_entropy_analysis": [
            ("low_threshold", "What upper threshold should define low normalized spatial entropy?"),
            ("high_threshold", "What lower threshold should define high normalized spatial entropy?"),
        ],
        "kongnet_cross_g_analysis": [
            ("radii", "At which distance threshold(s) should the cross-G statistic be evaluated?"),
            ("distance_units", "Which distance units should be used: microns or pixels?"),
            ("min_probability", "What minimum nucleus-class probability threshold should be used?"),
        ],
        "nucleus_spatial_analysis": [
            ("neighbourhood_radius", "What cell-neighbourhood/co-occurrence distance threshold should be used?"),
            ("point_pattern_radii", "At which radii should the point-pattern statistics be evaluated (in microns)?"),
            ("min_probability", "What minimum nucleus-class probability threshold should be used?"),
            ("min_cells_per_region", "What minimum number of cells should an ROI contain before its spatial statistics are computed?"),
        ],
    }
    spatial_feature_questions = {
        "common_object_spatial_analysis": [
            ("point_pattern_cell_types", "Which classes should receive per-class NNI, quadrat, and Ripley statistics?"),
            ("source_types", "Which class or classes should be the source population?"),
            ("target_types", "Which class or classes should be the target population?"),
            ("cooccurrence_cell_types", "Which classes should be included in co-occurrence analysis?"),
        ],
        "common_roi_morans_i": [
            ("metrics", "Which numeric common ROI feature(s) should Moran's I analyse?"),
        ],
        "common_roi_entropy": [
            ("cell_types", "Which adapted classes should contribute to spatial entropy?"),
        ],
        "custom_nucleus_spatial_analysis": [
            ("cooccurrence_cell_types", "Which cell types should be included in the co-occurrence statistic?"),
            ("neighbourhood_source_types", "Which cell type(s) should be the focal/source population for neighbourhood counts?"),
            ("neighbourhood_target_types", "Which cell type(s) should be counted as the target population?"),
        ],
        "kongnet_morans_i_analysis": [
            ("metrics", "Which numeric ROI features should Moran's I analyse?"),
        ],
        "kongnet_local_morans_i_analysis": [
            ("metric", "Which single numeric ROI feature should Local Moran's I analyse?"),
        ],
        "kongnet_spatial_entropy_analysis": [
            ("cell_types", "Which cell-type composition features should contribute to spatial entropy?"),
        ],
        "kongnet_cross_g_analysis": [
            ("source_types", "Which cell type(s) should be the source population for cross-G?"),
            ("target_types", "Which cell type(s) should be the target population for cross-G?"),
        ],
        "nucleus_spatial_analysis": [
            ("cooccurrence_cell_types", "Which cell types should be included in the co-occurrence statistic?"),
            ("neighbourhood_source_types", "Which cell type(s) should be the focal/source population for neighbourhood analysis?"),
            ("neighbourhood_target_types", "Which cell type(s) should be counted as the target population?"),
            ("point_pattern_cell_types", "Which cell types should receive per-class point-pattern statistics?"),
            ("point_pattern_source_types", "Which source cell type(s) should be used for cross-type point-pattern proximity?"),
            ("point_pattern_target_types", "Which target cell type(s) should be used for cross-type point-pattern proximity?"),
        ],
    }

    if task_type == "common_object_spatial_analysis":
        plan = {
            "task_type": task_type,
            "goal": "Run compatible point-level spatial statistics on a standardized common-object representation.",
            "steps": [
                "Validate selected classes and common-object coordinates.",
                "Compute per-class NNI, quadrat statistics, and Ripley-style multi-radius statistics.",
                "Compute selected source-to-target Cross-G and nearest-neighbour/radius summaries.",
                "Compute selected-class co-occurrence within the approved radius.",
            ],
            "suggested_tools_after_approval": [
                "validate_spatial_capabilities", "compute_common_point_pattern_statistics",
                "compute_common_cross_g", "compute_common_cooccurrence", "compute_common_neighbour_distances",
            ],
            "expected_outputs": [
                os.path.join(output_dir, "common_point_pattern_statistics.json"),
                os.path.join(output_dir, "common_cross_g.json"),
                os.path.join(output_dir, "common_cooccurrence.json"),
                os.path.join(output_dir, "common_neighbour_distances.json"),
            ],
            "default_parameters": {"radii": [25.0, 50.0, 100.0], "cooccurrence_radius": 50.0,
                                   "neighbourhood_radius": 50.0, "min_probability": 0.0,
                                   "min_points_per_pattern": 10},
        }
    elif task_type == "common_spatial_adapter":
        plan = {
            "task_type": task_type,
            "goal": "Convert a model-family output into standardized spatial object and ROI feature tables and validate its statistical capabilities.",
            "steps": [
                "Read the selected model output and its geometry/class metadata.",
                "Convert object outputs to standardized centroids and region outputs to area-weighted grid features.",
                "Write common object, long-form ROI feature, common JSON, and capability files.",
                "Validate which spatial statistics are supported by the converted representation.",
            ],
            "suggested_tools_after_approval": ["build_common_spatial_features", "validate_spatial_capabilities"],
            "expected_outputs": [
                os.path.join(output_dir, "spatial_objects.csv"),
                os.path.join(output_dir, "spatial_roi_features.csv"),
                os.path.join(output_dir, "common_spatial_features.json"),
                os.path.join(output_dir, "spatial_capabilities.json"),
            ],
            "default_parameters": {"region_size": 256.0, "min_probability": 0.0, "distance_units": "pixels"},
        }
    elif task_type == "common_roi_morans_i":
        plan = {
            "task_type": task_type,
            "goal": "Compute Moran's I from numeric ROI features in a common cross-model spatial representation.",
            "steps": ["Validate the requested ROI features.", "Build ROI spatial weights.", "Compute Moran's I and permutation p-values for each selected feature."],
            "suggested_tools_after_approval": ["validate_spatial_capabilities", "compute_common_roi_morans_i"],
            "expected_outputs": [os.path.join(output_dir, "common_roi_morans_i.json"), os.path.join(output_dir, "common_roi_morans_i.txt")],
            "default_parameters": {"weights_method": "queen", "permutations": 999, "alpha": 0.05},
        }
    elif task_type == "common_roi_entropy":
        plan = {
            "task_type": task_type,
            "goal": "Compute class-composition entropy from a common cross-model spatial representation.",
            "steps": ["Validate the selected classes.", "Compute and normalize Shannon entropy per ROI.", "Classify and save ROI diversity results."],
            "suggested_tools_after_approval": ["validate_spatial_capabilities", "compute_common_roi_entropy"],
            "expected_outputs": [os.path.join(output_dir, "common_roi_entropy.json"), os.path.join(output_dir, "common_roi_entropy.csv"), os.path.join(output_dir, "common_roi_entropy.txt")],
            "default_parameters": {"normalize": True, "low_threshold": 0.4, "high_threshold": 0.7},
        }
    elif task_type == "roi_pair_validation":
        plan = {
            "task_type": task_type,
            "goal": "Validate that a QuPath GeoJSON annotation and exported ROI image form a consistent pair.",
            "steps": [
                "Read the exported ROI image metadata.",
                "Read and validate the selected Polygon or MultiPolygon annotation from the GeoJSON.",
                "Compare the GeoJSON bounding box with the image dimensions and infer the crop downsample.",
                "Calculate the original-WSI origin, crop-local polygon bounds, ROI fraction, and physical area.",
                "Save a JSON validation manifest; do not run prediction or spatial analysis."
            ],
            "suggested_tools_after_approval": [
                "validate_qupath_roi_pair"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "qupath_roi_pair_validation.json")
            ]
        }

    elif task_type == "metadata_only":
        plan = {
            "task_type": task_type,
            "goal": "Read and report WSI metadata without running analysis or prediction.",
            "steps": [
                "Open the image in read-only mode.",
                "Report dimensions, microns-per-pixel, pyramid levels, channels, and objective information.",
                "Do not run prediction, segmentation, masking, patch extraction, or report generation."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata"
            ],
            "expected_outputs": []
        }

    elif task_type == "thumbnail_only":
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

    elif task_type == "spatial_pathology_question":
        plan = {
            "task_type": task_type,
            "goal": "Answer a spatial pathology question using explainable KongNet ROI evidence.",
            "steps": [
                "Read the saved fixed-ROI spatial analysis.",
                "Infer whether the question concerns tumour-immune interaction, inflammatory infiltration, neoplastic composition, stromal composition, epithelial composition, or density.",
                "Rank regions using the corresponding transparent metric.",
                "Return the highest-ranking regions with values and interpretation limitations."
            ],
            "suggested_tools_after_approval": [
                "answer_kongnet_spatial_question"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kongnet_spatial_question_answer.txt")
            ],
            "default_parameters": {"top_k": 5},
            "clinical_warning": "Answers query model-derived ROI evidence and are not clinical diagnoses."
        }

    elif task_type == "custom_nucleus_spatial_analysis":
        plan = {
            "task_type": task_type,
            "goal": "Run individually parameterised KongNet ROI spatial analyses without rerunning inference.",
            "steps": [
                "Validate the QuPath OME-TIFF and GeoJSON ROI pair.",
                "Export filtered KongNet nucleus identifiers, coordinates, classes, and probabilities.",
                "Compute cell-type co-occurrence using the user-requested co-occurrence radius.",
                "Count selected target cell types around each requested source population using the independently requested radius.",
                "Divide the ROI into local regions and calculate cell composition and density.",
                "Save separate CSV and JSON outputs for each operation."
            ],
            "suggested_tools_after_approval": [
                "validate_qupath_roi_pair",
                "export_kongnet_nuclei_to_csv",
                "compute_cell_type_cooccurrence",
                "find_cells_within_radius",
                "analyze_kongnet_regions"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "qupath_roi_pair_validation.json"),
                os.path.join(output_dir, "roi_nuclei.csv"),
                os.path.join(output_dir, "cell_type_cooccurrence_100um.csv"),
                os.path.join(output_dir, "cell_type_cooccurrence_100um.json"),
                os.path.join(output_dir, "neoplastic_neighbours_25um.csv"),
                os.path.join(output_dir, "neoplastic_neighbours_25um.json"),
                os.path.join(output_dir, "roi_cell_density.csv"),
                os.path.join(output_dir, "roi_cell_density.json")
            ],
            "default_parameters": {
                "distance_units": "microns",
                "cooccurrence_radius": 100.0,
                "neighbourhood_radius": 25.0,
                "region_size": 100.0,
                "min_cells_per_region": 1,
                "neighbourhood_target_types": ["Neoplastic"],
                "min_probability": 0.0
            },
            "clinical_warning": (
                "These are model-derived ROI spatial research features, not a clinical diagnosis."
            )
        }

    elif task_type == "kongnet_local_morans_i_analysis":
        plan = {
            "task_type": task_type,
            "goal": "Identify individual KongNet ROIs that form significant local clusters or spatial outliers for one selected ROI feature.",
            "steps": [
                "Read the existing KongNet ROI results from kongnet_regions.json.",
                "Extract the selected numeric feature for every ROI.",
                "Build binary distance-band weights from ROI-centroid distances using the user-selected radius.",
                "Row-standardize the spatial weights.",
                "Compute Local Moran's I and conditional permutation p-values for every ROI.",
                "Classify significant ROIs as high-high, low-low, high-low, or low-high.",
                "Save JSON, CSV, and TXT results plus a TIAViz-compatible ROI AnnotationStore overlay."
            ],
            "suggested_tools_after_approval": [
                "compute_kongnet_local_morans_i"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kongnet_local_morans_i.json"),
                os.path.join(output_dir, "kongnet_local_morans_i.csv"),
                os.path.join(output_dir, "kongnet_local_morans_i.txt"),
                os.path.join(output_dir, "kongnet_local_morans_i_overlay.db")
            ],
            "default_parameters": {
                "regions_json_path": (
                    wsi_path
                    if str(wsi_path).lower().endswith(".json")
                    else os.path.join(output_dir, "kongnet_regions.json")
                ),
                "weights_method": "distance",
                "k_neighbours": 4,
                "permutations": 999,
                "alpha": 0.05,
                "seed": 42
            },
            "clinical_warning": (
                "Local Moran's I is exploratory and involves multiple ROI-level tests. "
                "It does not validate model predictions, prove causality, or provide a clinical diagnosis."
            )
        }

    elif task_type == "kongnet_morans_i_analysis":
        request = user_prompt.lower()
        if "distance" in request:
            weights_method = "distance"
        elif "knn" in request or "k nearest" in request or "k-nearest" in request:
            weights_method = "knn"
        else:
            weights_method = "queen"
        plan = {
            "task_type": task_type,
            "goal": "Compute ROI-level Moran's I spatial autocorrelation from existing KongNet region analysis results without rerunning nucleus detection or the full spatial workflow.",
            "steps": [
                "Read the existing KongNet ROI results from kongnet_regions.json.",
                "Extract one numeric value per ROI for each requested metric, such as neoplastic percentage, inflammatory percentage, cell density, and tumour-immune interaction strength.",
                "Build ROI spatial weights using PySAL/libpysal.",
                "Compute Moran's I and permutation p-values using esda.moran.Moran.",
                "Save the results as a machine-readable JSON file and a readable text report in the same output directory."
            ],
            "suggested_tools_after_approval": [
                "compute_kongnet_morans_i"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kongnet_morans_i.json"),
                os.path.join(output_dir, "kongnet_morans_i.txt")
            ],
            "default_parameters": {
                "regions_json_path": os.path.join(output_dir, "kongnet_regions.json"),
                "metrics": [
                    "Neoplastic_percentage",
                    "Inflammatory_percentage",
                    "cell_density",
                    "interaction_strength"
                ],
                "weights_method": weights_method,
                "k_neighbours": 4,
                "permutations": 999,
                "alpha": 0.05
            },
            "clinical_warning": (
                "Moran's I reports exploratory ROI-level spatial autocorrelation. "
                "It does not validate model predictions, prove causality, or provide a clinical diagnosis."
            )
        }

    elif task_type == "kongnet_spatial_entropy_analysis":
        plan = {
            "task_type": task_type,
            "goal": "Compute ROI-level spatial entropy from existing KongNet region composition results without rerunning nucleus detection or the full spatial workflow.",
            "steps": [
                "Read the existing KongNet ROI results from kongnet_regions.json.",
                "Extract the predicted cell-type counts for each ROI.",
                "Compute Shannon entropy from the ROI cell-type proportions.",
                "Normalize entropy to a 0-1 diversity score so regions are easier to compare.",
                "Label each ROI as low-diversity/homogeneous, moderate-diversity, or high-diversity/mixed.",
                "Save JSON, readable TXT, and CSV outputs in the same output directory."
            ],
            "suggested_tools_after_approval": [
                "compute_kongnet_spatial_entropy"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kongnet_spatial_entropy.json"),
                os.path.join(output_dir, "kongnet_spatial_entropy.txt"),
                os.path.join(output_dir, "kongnet_spatial_entropy.csv")
            ],
            "default_parameters": {
                "regions_json_path": os.path.join(output_dir, "kongnet_regions.json"),
                "normalize": True,
                "entropy_base": 2.718281828459045,
                "low_threshold": 0.40,
                "high_threshold": 0.70
            },
            "clinical_warning": (
                "Spatial entropy describes ROI-level compositional heterogeneity only. "
                "It does not prove biological interaction, validate model predictions, or provide a clinical diagnosis."
            )
        }

    elif task_type == "kongnet_cross_g_analysis":
        plan = {
            "task_type": task_type,
            "goal": "Compute a formal empirical cross-G function from existing KongNet nucleus detections to quantify source-to-target cell proximity.",
            "steps": [
                "Read existing KongNet nucleus coordinates and cell classes from the AnnotationStore.",
                "Select source cell types, defaulting to neoplastic/tumour cells when available and epithelial cells otherwise.",
                "Select target cell types, defaulting to immune/inflammatory KongNet classes.",
                "For each source cell, compute the nearest target-cell distance.",
                "For each radius, compute the empirical cross-G probability: the fraction of source cells with a target cell within that radius.",
                "Compare the empirical curve with a CSR/Poisson expectation based on target-cell density.",
                "Optionally compute the same cross-G curve separately inside existing ROIs from kongnet_regions.json.",
                "Save JSON, readable TXT, and CSV outputs in the same output directory."
            ],
            "suggested_tools_after_approval": [
                "compute_kongnet_cross_g_function"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kongnet_cross_g_function.json"),
                os.path.join(output_dir, "kongnet_cross_g_function.txt"),
                os.path.join(output_dir, "kongnet_cross_g_function.csv")
            ],
            "default_parameters": {
                "regions_json_path": os.path.join(output_dir, "kongnet_regions.json"),
                "radii": [25.0, 50.0, 100.0],
                "distance_units": "microns",
                "source_types": None,
                "target_types": None,
                "min_probability": 0.0
            },
            "clinical_warning": (
                "Cross-G is an exploratory distance-based spatial statistic. "
                "It does not prove biological interaction, validate model predictions, or provide a clinical diagnosis."
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
                "Compute nearest-neighbour distances by source and target cell type.",
                "Divide the slide into local ROIs and compute composition and spatial features per region.",
                "Compute point-pattern statistics including nearest-neighbour index, quadrat heterogeneity, Ripley-style clustering, and tumour/epithelial-immune proximity.",
                "Export point-pattern visual overlays for clustered-cell ROIs, NNI, quadrat VMR, and Ripley clustering strength.",
                "Export ROI rectangles as a TIAViz-compatible AnnotationStore for true WSI overlay.",
                "Characterise every cell by neighbour type and cluster the profiles into spatial communities.",
                "Include region and community findings in the text interpretability report."
                " Rank regions, generate heatmaps, answer an optional pathology question, and create a slide summary."
            ],
            "suggested_tools_after_approval": [
                "run_kongnet_spatial_workflow"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "kongnet_nuclei.csv"),
                os.path.join(output_dir, "radius_neighbourhoods.csv"),
                os.path.join(output_dir, "cell_type_cooccurrence.json"),
                os.path.join(output_dir, "nearest_neighbours.csv"),
                os.path.join(output_dir, "kongnet_regions.json"),
                os.path.join(output_dir, "kongnet_point_pattern_statistics.json"),
                os.path.join(output_dir, "kongnet_point_pattern_statistics.txt"),
                os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_clustered_cell_roi.db"),
                os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_nni_heatmap.db"),
                os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_quadrat_vmr_heatmap.db"),
                os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_ripley_clustering_strength.db"),
                os.path.join(output_dir, f"{slide_prefix}kongnet_region_boundaries.db"),
                os.path.join(output_dir, "kongnet_cell_neighbourhoods.csv"),
                os.path.join(output_dir, "kongnet_spatial_communities.json"),
                os.path.join(output_dir, "kongnet_region_rankings.txt"),
                os.path.join(output_dir, f"{slide_prefix}kongnet_density_heatmap.db"),
                os.path.join(output_dir, f"{slide_prefix}kongnet_inflammatory_heatmap.db"),
                os.path.join(output_dir, f"{slide_prefix}kongnet_tumour_immune_interaction_heatmap.db"),
                os.path.join(output_dir, "kongnet_slide_summary.txt"),
                os.path.join(output_dir, "kongnet_ai_interpretability_report.txt")
            ],
            "default_parameters": {
                "neighbourhood_radius": 50.0,
                "point_pattern_radii": [25.0, 50.0, 100.0],
                "region_size": 100.0,
                "min_cells_per_region": 1,
                "min_points_per_pattern": 1,
                "distance_units": "microns",
                "min_probability": 0.0
            },
            "clinical_warning": (
                "These are model-derived spatial research features, not a clinical diagnosis."
            )
        }

    elif task_type == "semantic_segmentation":
        plan = {
            "task_type": task_type,
            "goal": "Run semantic segmentation on the WSI and generate TIAViz-compatible outputs where supported.",
            "steps": [
                "Read WSI metadata.",
                "Run TIAToolbox SemanticSegmentor with the requested model.",
                "Save semantic region outputs in the requested AnnotationStore-compatible format.",
                "Save a JSON run summary containing model metadata and the TIAViz launch command.",
                "Open the slide and generated semantic overlay in TIAViz using the generated command."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "predict_semantic_segmentation",
                "check_prediction_job",
                "save_run_report"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "semantic_segmentation_predictions.json"),
                os.path.join(output_dir, "semantic_segmentation_annotationstore"),
                os.path.join(output_dir, "run_report.json")
            ],
            "default_parameters": {
                "model_name": "fcn_resnet50_unet-bcss",
                "batch_size": 8,
                "output_type": "annotationstore",
                "patch_mode": False,
                "auto_get_mask": False,
                "num_workers": 1,
                "run_async": True
            },
            "supported_models": {
                model_name: _semantic_segmentation_model_summary(model_name, model_meta)
                for model_name, model_meta in SEMANTIC_SEGMENTATION_MODEL_CATALOG.items()
            },
            "clinical_warning": (
                "The output is model-derived semantic segmentation, not a clinical diagnosis."
            )
        }

    elif task_type == "multi_task_segmentation":
        plan = {
            "task_type": task_type,
            "goal": "Run multi-task segmentation on the WSI and generate TIAViz-compatible outputs where supported.",
            "steps": [
                "Read WSI metadata.",
                "Run TIAToolbox MultiTaskSegmentor with the requested model.",
                "Save model outputs in the requested AnnotationStore-compatible format.",
                "Save a JSON run summary containing model metadata and the TIAViz launch command.",
                "Open the slide and generated overlays in TIAViz using the generated command."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "predict_multi_task_segmentation",
                "check_prediction_job",
                "save_run_report"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "multi_task_segmentation_predictions.json"),
                os.path.join(output_dir, "multi_task_segmentation_annotationstore"),
                os.path.join(output_dir, "run_report.json")
            ],
            "default_parameters": {
                "model_name": "hovernetplus-oed",
                "batch_size": 8,
                "output_type": "annotationstore",
                "patch_mode": False,
                "auto_get_mask": False,
                "num_workers": 1,
                "run_async": True
            },
            "supported_models": {
                model_name: _multi_task_segmentation_model_summary(model_name, model_meta)
                for model_name, model_meta in MULTI_TASK_SEGMENTATION_MODEL_CATALOG.items()
            },
            "clinical_warning": (
                "The output is model-derived multi-task segmentation, not a clinical diagnosis."
            )
        }

    elif task_type == "nucleus_instance_segmentation":
        plan = {
            "task_type": task_type,
            "goal": "Run nucleus instance segmentation on the WSI and generate a TIAViz-compatible AnnotationStore.",
            "steps": [
                "Read WSI metadata.",
                "Run TIAToolbox NucleusInstanceSegmentor with the requested model.",
                "Save nucleus instance boundaries/classes as a TIAViz-compatible AnnotationStore (.db).",
                "Save a JSON run summary containing the TIAViz launch command.",
                "Open the slide and instance segmentation overlay in TIAViz using the generated command."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "predict_nucleus_instance_segmentation",
                "check_prediction_job",
                "save_run_report"
            ],
            "expected_outputs": [
                os.path.join(output_dir, "nucleus_instance_segmentation_predictions.json"),
                os.path.join(output_dir, "nucleus_instance_segmentation_annotationstore"),
                os.path.join(output_dir, "run_report.json")
            ],
            "default_parameters": {
                "model_name": "hovernet_fast-monusac",
                "batch_size": 8,
                "output_type": "annotationstore",
                "patch_mode": False,
                "auto_get_mask": False,
                "num_workers": 1,
                "run_async": True
            },
            "supported_models": {
                model_name: _nucleus_instance_segmentation_model_summary(model_name, model_meta)
                for model_name, model_meta in NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG.items()
            },
            "clinical_warning": (
                "The output is model-derived nucleus instance segmentation, not a clinical diagnosis."
            )
        }

    elif task_type == "kongnet_nucleus_detection":
        plan = {
            "task_type": task_type,
            "goal": "Run KongNet nucleus detection on the WSI and generate a TIAViz-compatible AnnotationStore.",
            "steps": [
                "Read WSI metadata.",
                "Run TIAToolbox NucleusDetector with the requested KongNet model.",
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
                "num_workers": 1,
                "run_async": True
            },
            "supported_models": {
                model_name: _kongnet_model_summary(model_name, model_meta)
                for model_name, model_meta in KONGNET_MODEL_CATALOG.items()
            },
            "clinical_warning": (
                "The output is nucleus detection/classification model output, not a clinical diagnosis."
            )
        }

    elif task_type == "kather_prediction":
        plan = {
            "task_type": task_type,
            "goal": "Run patch-level WSI prediction and generate a TIAViz-compatible AnnotationStore.",
            "steps": [
                "Read WSI metadata.",
                "Run the requested pretrained patch prediction model directly on the WSI.",
                "Save prediction output as a TIAViz-compatible AnnotationStore (.db).",
                "Save a JSON run summary containing the TIAViz launch command.",
                "Open the slide and overlay in TIAViz using the generated command."
            ],
            "suggested_tools_after_approval": [
                "wsi_metadata",
                "predict_patch_model",
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
            "supported_models": {
                model_name: _patch_prediction_model_summary(model_name, model_meta)
                for model_name, model_meta in PATCH_PREDICTION_MODEL_CATALOG.items()
            },
            "clinical_warning": (
                "The output is patch-level classification/model-confidence analysis, not a clinical diagnosis."
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

    # A QuPath image/GeoJSON pair is always validated before any downstream
    # operation. This is based on supplied inputs, not prompt keyword matching.
    requires_roi_validation = bool(wsi_path and geojson_path)
    if requires_roi_validation:
        validation_output = os.path.join(output_dir, "qupath_roi_pair_validation.json")
        tools = plan.setdefault("suggested_tools_after_approval", [])
        if "validate_qupath_roi_pair" in tools:
            tools.remove("validate_qupath_roi_pair")
        tools.insert(0, "validate_qupath_roi_pair")

        steps = plan.setdefault("steps", [])
        validation_step = (
            "Validate the exact QuPath ROI image/GeoJSON pair and stop all downstream "
            "execution if validation fails."
        )
        steps[:] = [step for step in steps if step != validation_step]
        steps.insert(0, validation_step)

        outputs = plan.setdefault("expected_outputs", [])
        if validation_output not in outputs:
            outputs.insert(0, validation_output)

        plan["requires_roi_validation"] = True
        plan["roi_validation"] = {
            "status": "pending",
            "image_path": os.path.abspath(wsi_path),
            "geojson_path": os.path.abspath(geojson_path),
            "output_json_path": os.path.abspath(validation_output),
        }

    plan["wsi_path"] = wsi_path
    if geojson_path:
        plan["geojson_path"] = geojson_path
    plan["output_dir"] = output_dir
    plan["plan_id"] = plan_id
    plan["status"] = "pending_user_approval"
    supplied_thresholds = dict(threshold_parameters or {})
    threshold_specs = spatial_threshold_questions.get(task_type, [])
    if (
        task_type == "common_spatial_adapter"
        and supplied_thresholds.get("distance_units") == "microns"
    ):
        threshold_specs.append(("mpp", "What microns-per-pixel (MPP) value should be used?"))
    # Moran's I only needs a distance cut-off when distance weights are selected.
    if task_type == "kongnet_morans_i_analysis" and plan["default_parameters"]["weights_method"] == "distance":
        threshold_specs.append(
            ("distance_threshold", "What neighbour-distance threshold should be used for Moran's I distance weights?")
        )

    allowed_thresholds = {name for name, _ in threshold_specs}
    unknown_thresholds = sorted(set(supplied_thresholds) - allowed_thresholds)
    if unknown_thresholds:
        raise ValueError(
            f"Unsupported threshold parameter(s) for {task_type}: {unknown_thresholds}. "
            f"Allowed parameters: {sorted(allowed_thresholds)}."
        )

    for name, value in supplied_thresholds.items():
        if name == "distance_units" and value not in {"microns", "pixels"}:
            raise ValueError('distance_units must be either "microns" or "pixels".')
        if name in {"alpha", "min_probability", "low_threshold", "high_threshold"}:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be a number between 0 and 1.")
        if name in {"cooccurrence_radius", "neighbourhood_radius", "distance_threshold", "region_size", "mpp"}:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive number.")
        if name in {"radii", "point_pattern_radii"}:
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) or float(v) <= 0 for v in value)
            ):
                raise ValueError(f"{name} must be a non-empty list of positive numbers.")
        if name in {"min_cells_per_region", "min_points_per_pattern"} and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{name} must be an integer of at least 1.")

    if {"low_threshold", "high_threshold"}.issubset(supplied_thresholds):
        if float(supplied_thresholds["low_threshold"]) > float(supplied_thresholds["high_threshold"]):
            raise ValueError("low_threshold must not exceed high_threshold.")

    supplied_features = dict(feature_parameters or {})
    feature_specs = spatial_feature_questions.get(task_type, [])
    allowed_features = {name for name, _ in feature_specs}
    unknown_features = sorted(set(supplied_features) - allowed_features)
    if unknown_features:
        raise ValueError(
            f"Unsupported feature parameter(s) for {task_type}: {unknown_features}. "
            f"Allowed parameters: {sorted(allowed_features)}."
        )
    for name, value in supplied_features.items():
        if name == "metric":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("metric must be a non-empty ROI feature name.")
            continue
        if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{name} must be a non-empty list of feature or cell-type names.")

    missing_features = [name for name, _ in feature_specs if name not in supplied_features]

    # A client LLM must not silently invent feature or threshold selections in
    # its first planning call. Values are accepted only as answers to a prior
    # clarification plan for the same task type. This makes the required user
    # clarification turn enforceable by the server rather than advisory text.
    supplied_selections = bool(supplied_thresholds or supplied_features)
    if supplied_selections:
        if not clarification_plan_id:
            raise RuntimeError(
                "Feature/threshold selections cannot be supplied on the first planning call. "
                "First call propose_pathology_plan without selections, show its questions to "
                "the user, then call it again with clarification_plan_id and the user's answers."
            )
        clarification_plan = PENDING_PLANS.get(str(clarification_plan_id))
        if not clarification_plan:
            raise RuntimeError("clarification_plan_id is invalid, expired, or already used.")
        if clarification_plan.get("task_type") != task_type:
            raise RuntimeError(
                "clarification_plan_id belongs to a different task type; feature selections "
                "must answer the questions from the same analysis plan."
            )
        if not (
            clarification_plan.get("threshold_questions_for_user")
            or clarification_plan.get("feature_questions_for_user")
        ):
            raise RuntimeError("The referenced plan did not request feature or threshold clarification.")
        PENDING_PLANS.pop(str(clarification_plan_id), None)

    if supplied_features:
        plan["selected_feature_parameters"] = supplied_features
        plan["default_parameters"].update(supplied_features)

    missing_thresholds = [name for name, _ in threshold_specs if name not in supplied_thresholds]
    if supplied_thresholds:
        plan["selected_threshold_parameters"] = supplied_thresholds
        plan["default_parameters"].update(supplied_thresholds)

    if missing_thresholds or missing_features:
        plan["threshold_questions_for_user"] = [
            {"parameter": name, "question": question}
            for name, question in threshold_specs
            if name in missing_thresholds
        ]
        plan["threshold_selection_required"] = bool(missing_thresholds)
        plan["feature_questions_for_user"] = [
            {"parameter": name, "question": question}
            for name, question in feature_specs
            if name in missing_features
        ]
        plan["feature_selection_required"] = bool(missing_features)
        plan["instruction"] = (
            "Show this draft plan to the user and explicitly ask every question in "
            "threshold_questions_for_user and feature_questions_for_user. Explain that any listed defaults are suggestions, "
            "not automatically selected values. Stop and wait for the user's threshold and feature choices "
            "before asking them to approve the final plan. Do not call approve_pathology_plan "
            "yourself until the user has answered the threshold questions and explicitly approves "
            "the resulting exact plan in a later message. After approval, call "
            "approve_pathology_plan to obtain a separate execution token. Only then call tools "
            "listed in suggested_tools_after_approval."
        )
    else:
        if threshold_specs:
            plan["threshold_selection_required"] = False
        if feature_specs:
            plan["feature_selection_required"] = False
        plan["instruction"] = (
            "Show this plan to the user and stop. Do not call approve_pathology_plan yourself "
            "until the user explicitly approves this exact plan in a later message. After the "
            "user approves, call approve_pathology_plan to obtain a separate execution token. "
            "Only then call tools listed in suggested_tools_after_approval."
        )

    PENDING_PLANS[plan_id] = plan
    return plan


def approve_plan(plan_id: str, confirmation: str) -> Dict[str, Any]:
    if confirmation != "I approve this plan":
        raise RuntimeError('Exact confirmation required: "I approve this plan".')
    if plan_id not in PENDING_PLANS:
        raise RuntimeError("Invalid, expired, or already-approved plan_id.")

    pending_plan = PENDING_PLANS[plan_id]
    if pending_plan.get("threshold_selection_required") or pending_plan.get("feature_selection_required"):
        raise RuntimeError(
            "This spatial plan cannot be approved until the user supplies every value in "
            "threshold_questions_for_user and feature_questions_for_user. Call "
            "propose_pathology_plan again with those values in threshold_parameters and "
            "feature_parameters, show the resulting final plan, and then request approval."
        )

    plan = PENDING_PLANS.pop(plan_id)
    approval_token = str(uuid.uuid4())
    plan["status"] = "approved"
    plan["approved_at"] = utc_now_iso()
    plan["approval_token"] = approval_token
    APPROVED_PLANS[approval_token] = plan
    instruction = "Execution may now begin, limited to approved_tools."
    if plan.get("requires_roi_validation"):
        instruction = (
            "Run validate_qupath_roi_pair first for the exact approved TIFF/GeoJSON pair. "
            "All other approved tools remain blocked unless validation passes."
        )
    return {
        "plan_id": plan_id,
        "status": "approved",
        "approval_token": approval_token,
        "approved_tools": plan.get("suggested_tools_after_approval", []),
        "instruction": instruction,
    }


def require_plan(args: Dict[str, Any], tool_name: str) -> str:
    token = args.get("approval_token")

    if not isinstance(token, str) or not token.strip():
        raise RuntimeError(
            "Execution requires an approval_token from approve_pathology_plan. "
            "First propose a plan, show it to the user, wait for explicit approval, "
            "and then approve that plan."
        )

    if token not in APPROVED_PLANS:
        raise RuntimeError(
            "Invalid or unapproved approval_token. A pending plan cannot authorize execution."
        )

    plan = APPROVED_PLANS[token]
    approved_tools = plan.get("suggested_tools_after_approval", [])
    if tool_name not in approved_tools:
        raise RuntimeError(
            f'Tool "{tool_name}" is outside the approved plan. '
            f"Approved tools: {approved_tools}."
        )

    selected = plan.get("selected_threshold_parameters", {})
    selected_features = plan.get("selected_feature_parameters", {})
    bindings = {
        "compute_common_point_pattern_statistics": {
            "radii": "radii", "min_probability": "min_probability", "min_points_per_pattern": "min_points_per_pattern",
        },
        "compute_common_cross_g": {"radii": "radii", "min_probability": "min_probability"},
        "compute_common_cooccurrence": {"radius": "cooccurrence_radius", "min_probability": "min_probability"},
        "compute_common_neighbour_distances": {"radius": "neighbourhood_radius", "min_probability": "min_probability"},
        "build_common_spatial_features": {
            "region_size": "region_size",
            "min_probability": "min_probability",
            "distance_units": "distance_units",
            "mpp": "mpp",
        },
        "compute_common_roi_morans_i": {"alpha": "alpha"},
        "compute_common_roi_entropy": {
            "low_threshold": "low_threshold",
            "high_threshold": "high_threshold",
        },
        "run_kongnet_spatial_workflow": {
            "neighbourhood_radius": "neighbourhood_radius",
            "point_pattern_radii": "point_pattern_radii",
            "min_probability": "min_probability",
            "min_cells_per_region": "min_cells_per_region",
        },
        "compute_cell_type_cooccurrence": {
            "radius": "cooccurrence_radius",
            "distance_units": "distance_units",
            "min_probability": "min_probability",
        },
        "export_kongnet_nuclei_to_csv": {
            "min_probability": "min_probability",
        },
        "analyze_kongnet_regions": {
            "neighbourhood_radius": "neighbourhood_radius",
            "distance_units": "distance_units",
            "min_probability": "min_probability",
        },
        "find_cells_within_radius": {
            "radius": "neighbourhood_radius",
            "distance_units": "distance_units",
            "min_probability": "min_probability",
        },
        "compute_kongnet_morans_i": {
            "alpha": "alpha",
            "distance_threshold": "distance_threshold",
        },
        "compute_kongnet_local_morans_i": {
            "alpha": "alpha",
            "distance_threshold": "distance_threshold",
        },
        "compute_kongnet_spatial_entropy": {
            "low_threshold": "low_threshold",
            "high_threshold": "high_threshold",
        },
        "compute_kongnet_cross_g_function": {
            "radii": "radii",
            "distance_units": "distance_units",
            "min_probability": "min_probability",
        },
    }
    for argument_name, selected_name in bindings.get(tool_name, {}).items():
        if selected_name not in selected:
            continue
        selected_value = selected[selected_name]
        if argument_name in args and args[argument_name] != selected_value:
            raise RuntimeError(
                f'Execution parameter "{argument_name}" ({args[argument_name]!r}) does not match '
                f'the user-approved value ({selected_value!r}). Propose and approve a new plan '
                "to use a different threshold."
            )
        args[argument_name] = selected_value

    feature_bindings = {
        "compute_common_point_pattern_statistics": {
            "cell_types": "point_pattern_cell_types", "source_types": "source_types", "target_types": "target_types",
        },
        "compute_common_cross_g": {"source_types": "source_types", "target_types": "target_types"},
        "compute_common_cooccurrence": {"cell_types": "cooccurrence_cell_types"},
        "compute_common_neighbour_distances": {"source_types": "source_types", "target_types": "target_types"},
        "compute_common_roi_morans_i": {"metrics": "metrics"},
        "compute_common_roi_entropy": {"cell_types": "cell_types"},
        "run_kongnet_spatial_workflow": {
            "cooccurrence_cell_types": "cooccurrence_cell_types",
            "neighbourhood_source_types": "neighbourhood_source_types",
            "neighbourhood_target_types": "neighbourhood_target_types",
            "point_pattern_cell_types": "point_pattern_cell_types",
            "point_pattern_source_types": "point_pattern_source_types",
            "point_pattern_target_types": "point_pattern_target_types",
        },
        "compute_cell_type_cooccurrence": {"cell_types": "cooccurrence_cell_types"},
        "find_cells_within_radius": {
            "source_types": "neighbourhood_source_types",
            "target_types": "neighbourhood_target_types",
        },
        "compute_kongnet_morans_i": {"metrics": "metrics"},
        "compute_kongnet_local_morans_i": {"metric": "metric"},
        "compute_kongnet_spatial_entropy": {"cell_types": "cell_types"},
        "compute_kongnet_cross_g_function": {
            "source_types": "source_types",
            "target_types": "target_types",
        },
    }
    for argument_name, selected_name in feature_bindings.get(tool_name, {}).items():
        if selected_name not in selected_features:
            continue
        selected_value = selected_features[selected_name]
        if argument_name in args and args[argument_name] != selected_value:
            raise RuntimeError(
                f'Execution feature "{argument_name}" ({args[argument_name]!r}) does not match '
                f'the user-approved value ({selected_value!r}). Propose and approve a new plan '
                "to analyse different features."
            )
        args[argument_name] = selected_value

    if plan.get("requires_roi_validation") and tool_name != "validate_qupath_roi_pair":
        validation = plan.get("roi_validation") or {}
        status = validation.get("status", "pending")
        if status not in {"passed", "passed_with_warnings"}:
            raise RuntimeError(
                f'Cannot execute "{tool_name}" because mandatory QuPath ROI-pair '
                f'validation status is "{status}". Run validate_qupath_roi_pair first '
                "with this approval token. If validation fails, downstream execution "
                "remains blocked."
            )

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
            "description": "Required first step for every pathology request. Returns a non-executable pending plan. Show it to the user and stop until the user explicitly approves it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_prompt": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "geojson_path": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "threshold_parameters": {
                        "type": "object",
                        "description": (
                            "User-selected spatial-statistic thresholds from the preceding "
                            "clarification turn. Parameter names must match those returned in "
                            "threshold_questions_for_user. Never populate this object by silently "
                            "accepting suggested defaults."
                        ),
                        "additionalProperties": True
                    },
                    "feature_parameters": {
                        "type": "object",
                        "description": (
                            "User-selected metrics or cell populations from the preceding "
                            "clarification turn. Parameter names must match those returned in "
                            "feature_questions_for_user."
                        ),
                        "additionalProperties": True
                    },
                    "clarification_plan_id": {
                        "type": "string",
                        "description": "Plan ID from the immediately preceding clarification plan whose questions the user has answered."
                    }
                },
                "required": ["user_prompt", "wsi_path", "output_dir"],
                "additionalProperties": False
            }
        },
        {
            "name": "approve_pathology_plan",
            "title": "Approve Pathology Plan",
            "description": "Call only after the user explicitly approves the exact pending plan in a later message. Converts a pending plan into an approved plan and returns the execution token.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "confirmation": {
                        "type": "string",
                        "enum": ["I approve this plan"]
                    }
                },
                "required": ["plan_id", "confirmation"],
                "additionalProperties": False
            }
        },
        {
            "name": "build_common_spatial_features",
            "title": "Build Common Cross-Model Spatial Features",
            "description": "Adapts KongNet, HoVerNet, HoVerNetPlus, semantic-segmentation, or patch-classification AnnotationStore/CSV output into common object and ROI feature tables.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "source_path": {"type": "string"},
                    "model_family": {"type": "string", "enum": ["kongnet", "hovernet", "hovernetplus", "semantic_segmentation", "patch_classification"]},
                    "model_name": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "region_size": {"type": "number", "exclusiveMinimum": 0},
                    "min_probability": {"type": "number", "minimum": 0, "maximum": 1},
                    "distance_units": {"type": "string", "enum": ["pixels", "microns"]},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number", "exclusiveMinimum": 0}
                    ,"min_recommended_rois": {"type": "integer", "minimum": 3}
                },
                "required": ["approval_token", "source_path", "model_family", "model_name", "output_dir", "region_size", "min_probability"],
                "additionalProperties": False
            }
        },
        {
            "name": "validate_spatial_capabilities",
            "title": "Validate Spatial Capabilities",
            "description": "Checks whether adapted model output has the objects, classes, coordinates, and ROI features required by a requested statistic.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "capabilities_json_path": {"type": "string"},
                    "statistic": {"type": "string"},
                    "feature_name": {"type": "string"},
                    "source_types": {"type": "array", "items": {"type": "string"}},
                    "target_types": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["approval_token", "capabilities_json_path", "statistic"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_common_roi_morans_i",
            "title": "Compute Cross-Model ROI Moran's I",
            "description": "Computes Moran's I for selected numeric features from common_spatial_features_v1 output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"}, "common_spatial_json_path": {"type": "string"},
                    "output_json_path": {"type": "string"}, "output_txt_path": {"type": "string"},
                    "metrics": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "weights_method": {"type": "string", "enum": ["queen", "rook", "distance", "knn"]},
                    "k_neighbours": {"type": "integer"}, "distance_threshold": {"type": "number"},
                    "permutations": {"type": "integer"}, "alpha": {"type": "number"}
                    ,"min_rois": {"type": "integer", "minimum": 3}
                },
                "required": ["approval_token", "common_spatial_json_path", "output_json_path", "metrics", "alpha"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_common_roi_entropy",
            "title": "Compute Cross-Model ROI Entropy",
            "description": "Computes class-composition Shannon entropy for selected classes from common_spatial_features_v1 output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"}, "common_spatial_json_path": {"type": "string"},
                    "output_json_path": {"type": "string"}, "output_txt_path": {"type": "string"}, "output_csv_path": {"type": "string"},
                    "cell_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "normalize": {"type": "boolean"}, "entropy_base": {"type": "number"},
                    "low_threshold": {"type": "number"}, "high_threshold": {"type": "number"}
                },
                "required": ["approval_token", "common_spatial_json_path", "output_json_path", "cell_types", "low_threshold", "high_threshold"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_common_point_pattern_statistics",
            "title": "Compute Common-Object Point-Pattern Statistics",
            "description": "Computes NNI, quadrat statistics, Ripley-style statistics and optional cross-type proximity from spatial_objects.csv.",
            "inputSchema": {"type": "object", "properties": {
                "approval_token": {"type": "string"}, "common_spatial_json_path": {"type": "string"},
                "output_json_path": {"type": "string"}, "output_txt_path": {"type": "string"},
                "cell_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "source_types": {"type": "array", "items": {"type": "string"}}, "target_types": {"type": "array", "items": {"type": "string"}},
                "radii": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}, "minItems": 1},
                "quadrat_grid_size": {"type": "integer", "minimum": 1}, "min_points_per_pattern": {"type": "integer", "minimum": 1},
                "min_probability": {"type": "number", "minimum": 0, "maximum": 1}
            }, "required": ["approval_token", "common_spatial_json_path", "output_json_path", "cell_types", "radii"], "additionalProperties": False}
        },
        {
            "name": "compute_common_cross_g",
            "title": "Compute Common-Object Cross-G",
            "description": "Computes empirical Cross-G for selected source and target classes from common spatial objects.",
            "inputSchema": {"type": "object", "properties": {
                "approval_token": {"type": "string"}, "common_spatial_json_path": {"type": "string"},
                "output_json_path": {"type": "string"}, "output_csv_path": {"type": "string"},
                "source_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "target_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "radii": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}, "minItems": 1},
                "min_probability": {"type": "number", "minimum": 0, "maximum": 1}
            }, "required": ["approval_token", "common_spatial_json_path", "output_json_path", "source_types", "target_types", "radii"], "additionalProperties": False}
        },
        {
            "name": "compute_common_cooccurrence",
            "title": "Compute Common-Object Co-occurrence",
            "description": "Counts selected class pairs within a user-approved radius.",
            "inputSchema": {"type": "object", "properties": {
                "approval_token": {"type": "string"}, "common_spatial_json_path": {"type": "string"}, "output_json_path": {"type": "string"},
                "radius": {"type": "number", "exclusiveMinimum": 0}, "cell_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "min_probability": {"type": "number", "minimum": 0, "maximum": 1}
            }, "required": ["approval_token", "common_spatial_json_path", "output_json_path", "radius", "cell_types"], "additionalProperties": False}
        },
        {
            "name": "compute_common_neighbour_distances",
            "title": "Compute Common-Object Neighbour Distances",
            "description": "Computes nearest source-to-target distances and optional within-radius counts.",
            "inputSchema": {"type": "object", "properties": {
                "approval_token": {"type": "string"}, "common_spatial_json_path": {"type": "string"}, "output_json_path": {"type": "string"},
                "source_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "target_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "radius": {"type": "number", "exclusiveMinimum": 0}, "min_probability": {"type": "number", "minimum": 0, "maximum": 1}
            }, "required": ["approval_token", "common_spatial_json_path", "output_json_path", "source_types", "target_types"], "additionalProperties": False}
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
            "name": "validate_qupath_roi_pair",
            "title": "Validate QuPath ROI Pair",
            "description": "Validates that an exported ROI image and a QuPath GeoJSON polygon are a consistent pair. Reports geometry validity, dimension agreement, WSI origin, inferred downsample, crop-local bounds, ROI fraction, and physical area. Does not run prediction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "image_path": {"type": "string"},
                    "geojson_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "feature_id": {"type": "string"},
                    "dimension_tolerance_pixels": {"type": "number"}
                },
                "required": [
                    "approval_token",
                    "image_path",
                    "geojson_path",
                    "output_json_path"
                ],
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
            "name": "predict_patch_model",
            "title": "Predict WSI Patch Classes",
            "description": "Runs a registered patch-level prediction model directly on a WSI and saves a TIAViz-compatible annotationstore. Defaults to resnet18-kather100k and also supports wide_resnet50_2-pcam.",
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
                    "model_name": {
                        "type": "string",
                        "enum": list(PATCH_PREDICTION_MODEL_CATALOG.keys()),
                    },
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
            "name": "predict_kather_resnet18",
            "title": "Predict WSI Patch Classes (Legacy Alias)",
            "description": "Backward-compatible alias for predict_patch_model. Defaults to resnet18-kather100k and also supports registered patch prediction models.",
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
                    "model_name": {
                        "type": "string",
                        "enum": list(PATCH_PREDICTION_MODEL_CATALOG.keys()),
                    },
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
            "title": "Detect Nuclei With KongNet",
            "description": "Runs TIAToolbox KongNet nucleus detection on a WSI and saves a TIAViz-compatible annotationstore. Defaults to KongNet_PanNuke_1, but supports any registered KongNet model, including KongNet_CoNIC_1, KongNet_Det_MIDOG_1, KongNet_MONKEY_1, KongNet_PUMA_T1_3, and KongNet_PUMA_T2_3, when model_name is explicitly provided.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "save_dir": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "model_name": {
                        "type": "string",
                        "enum": list(KONGNET_MODEL_CATALOG.keys()),
                    },
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
            "name": "predict_nucleus_instance_segmentation",
            "title": "Segment Nucleus Instances",
            "description": "Runs TIAToolbox nucleus instance segmentation on a WSI and saves a TIAViz-compatible annotationstore. Defaults to hovernet_fast-monusac, and also supports hovernet_fast-pannuke, hovernet_original-consep, and hovernet_original-kumar through the separate nucleus instance segmentation model catalog.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "save_dir": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "model_name": {
                        "type": "string",
                        "enum": list(NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG.keys()),
                    },
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
            "name": "predict_multi_task_segmentation",
            "title": "Run Multi-Task Segmentation",
            "description": "Runs TIAToolbox multi-task segmentation on a WSI. Defaults to hovernetplus-oed, using the separate multi-task segmentation model catalog.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "save_dir": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "model_name": {
                        "type": "string",
                        "enum": list(MULTI_TASK_SEGMENTATION_MODEL_CATALOG.keys()),
                    },
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
            "name": "predict_semantic_segmentation",
            "title": "Run Semantic Segmentation",
            "description": "Runs TIAToolbox semantic segmentation on a WSI. Defaults to fcn_resnet50_unet-bcss, using the separate semantic segmentation model catalog.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "save_dir": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "model_name": {
                        "type": "string",
                        "enum": list(SEMANTIC_SEGMENTATION_MODEL_CATALOG.keys()),
                    },
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
            "description": "Checks a background prediction job started by predict_patch_model, predict_kather_resnet18, predict_kongnet_nucleus_detection, predict_nucleus_instance_segmentation, predict_multi_task_segmentation, or predict_semantic_segmentation.",
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
            "name": "run_kongnet_spatial_workflow",
            "title": "Run Full KongNet Spatial Workflow",
            "description": "Runs the complete single-WSI KongNet interpretability workflow: nuclei export, radius neighbourhoods, co-occurrence, nearest neighbours, local ROIs, point-pattern statistics, point-pattern TIAViz overlays, ROI overlay, per-cell communities, and a plain-text report.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "min_probability": {"type": "number"},
                    "neighbourhood_radius": {"type": "number"},
                    "point_pattern_radii": {
                        "type": "array",
                        "items": {"type": "number", "exclusiveMinimum": 0},
                        "minItems": 1
                    },
                    "cooccurrence_cell_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "neighbourhood_source_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "neighbourhood_target_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "point_pattern_cell_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "point_pattern_source_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "point_pattern_target_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "region_size": {
                        "oneOf": [
                            {"type": "number", "exclusiveMinimum": 0},
                            {"type": "string", "enum": ["auto"]}
                        ],
                        "description": "Explicit grid-cell size in distance units, or 'auto' to adapt to ROI area and nucleus count."
                    },
                    "min_cells_per_region": {"type": "integer"},
                    "community_count": {"type": "integer"},
                    "pathology_question": {"type": "string"},
                    "overwrite": {"type": "boolean"}
                },
                "required": ["approval_token", "annotationstore_path", "output_dir"],
                "additionalProperties": False
            }
        },
        {
            "name": "rank_kongnet_regions",
            "title": "Rank KongNet Regions",
            "description": "Ranks ROIs by inflammatory, neoplastic, epithelial, stromal, density, and normalized tumour-immune interaction measurements.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_txt_path": {"type": "string"},
                    "top_k": {"type": "integer"}
                },
                "required": ["approval_token", "regions_json_path", "output_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "answer_kongnet_spatial_question",
            "title": "Ask the KongNet Spatial Pathology Assistant",
            "description": "Answers explainable questions about tumour-immune interaction, inflammatory, neoplastic, epithelial, stromal, or dense regions using ROI evidence.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "question": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "output_txt_path": {"type": "string"},
                    "top_k": {"type": "integer"}
                },
                "required": ["approval_token", "question", "regions_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "generate_kongnet_region_heatmaps",
            "title": "Generate KongNet Region Heatmaps",
            "description": "Generates density, inflammatory, and tumour-immune interaction heatmaps as TIAViz-compatible AnnotationStores only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "overwrite": {"type": "boolean"}
                },
                "required": ["approval_token", "regions_json_path", "output_dir"],
                "additionalProperties": False
            }
        },
        {
            "name": "generate_kongnet_point_pattern_overlays",
            "title": "Generate KongNet Point-Pattern Overlays",
            "description": "Generates four TIAViz-compatible AnnotationStore overlays from point-pattern statistics: clustered-cell ROI overlay, NNI heatmap, quadrat VMR heatmap, and Ripley clustering-strength overlay.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "point_pattern_json_path": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "overwrite": {"type": "boolean"}
                },
                "required": ["approval_token", "point_pattern_json_path", "output_dir"],
                "additionalProperties": False
            }
        },
        {
            "name": "generate_kongnet_slide_summary",
            "title": "Generate KongNet Slide Summary",
            "description": "Creates a concise text summary of whole-slide cell composition and the highest-ranked inflammatory, neoplastic, and tumour-immune ROIs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "nuclei_csv_path": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "output_txt_path": {"type": "string"},
                    "output_json_path": {"type": "string"}
                },
                "required": ["approval_token", "nuclei_csv_path", "regions_json_path", "output_txt_path"],
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
            "name": "analyze_kongnet_regions",
            "title": "Analyse KongNet Local Regions",
            "description": "Divides detections into fixed local ROIs and computes cell composition, density, local pair counts, and nearest-neighbour distance separately for every region.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "region_size": {
                        "oneOf": [
                            {"type": "number", "exclusiveMinimum": 0},
                            {"type": "string", "enum": ["auto"]}
                        ],
                        "description": "Explicit grid-cell size in distance units, or 'auto' to adapt to ROI area and nucleus count."
                    },
                    "neighbourhood_radius": {"type": "number"},
                    "distance_units": {"type": "string", "enum": ["microns", "pixels"]},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "min_cells_per_region": {"type": "integer"},
                    "min_probability": {"type": "number"}
                },
                "required": ["approval_token", "annotationstore_path", "output_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_kongnet_point_pattern_statistics",
            "title": "Compute KongNet Point-Pattern Statistics",
            "description": "Computes point-pattern spatial statistics from KongNet nuclei: nearest-neighbour index, quadrat heterogeneity, Ripley-style multi-radius clustering, and tumour/epithelial-immune proximity. Uses pointpats when available and transparent NumPy/SciPy equivalents otherwise.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_txt_path": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "distance_units": {"type": "string", "enum": ["microns", "pixels"]},
                    "radii": {"type": "array", "items": {"type": "number"}},
                    "quadrat_grid_size": {"type": "integer"},
                    "min_points_per_pattern": {"type": "integer"},
                    "min_probability": {"type": "number"}
                },
                "required": ["approval_token", "annotationstore_path", "output_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_kongnet_morans_i",
            "title": "Compute KongNet ROI Moran's I",
            "description": "Computes ROI-level spatial autocorrelation using PySAL/libpysal/esda. This tests whether high or low regional values such as inflammatory percentage, neoplastic percentage, cell density, or interaction strength cluster near similar ROIs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_txt_path": {"type": "string"},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "weights_method": {"type": "string", "enum": ["queen", "rook", "distance", "knn"]},
                    "k_neighbours": {"type": "integer"},
                    "distance_threshold": {"type": "number"},
                    "permutations": {"type": "integer"},
                    "alpha": {"type": "number"}
                },
                "required": ["approval_token", "regions_json_path", "output_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_kongnet_local_morans_i",
            "title": "Compute KongNet ROI Local Moran's I",
            "description": "Computes one Local Moran's I statistic per ROI for a selected numeric ROI feature, identifies significant high-high, low-low, high-low, and low-high patterns, and creates a TIAViz-compatible ROI AnnotationStore overlay.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "output_txt_path": {"type": "string"},
                    "output_annotationstore_path": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "overwrite": {"type": "boolean"},
                    "metric": {"type": "string"},
                    "weights_method": {"type": "string", "enum": ["queen", "rook", "distance", "knn"]},
                    "k_neighbours": {"type": "integer"},
                    "distance_threshold": {"type": "number"},
                    "permutations": {"type": "integer"},
                    "alpha": {"type": "number"},
                    "seed": {"type": "integer"}
                },
                "required": ["approval_token", "regions_json_path", "output_json_path", "metric"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_kongnet_spatial_entropy",
            "title": "Compute KongNet ROI Spatial Entropy",
            "description": "Computes Shannon entropy from KongNet ROI class composition. This scores each ROI from homogeneous/cell-type-dominant to mixed/high-diversity microenvironment and saves JSON, TXT, and optional CSV outputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_txt_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "normalize": {"type": "boolean"},
                    "entropy_base": {"type": "number"},
                    "low_threshold": {"type": "number"},
                    "high_threshold": {"type": "number"}
                    ,"cell_types": {"type": "array", "items": {"type": "string"}, "minItems": 1}
                },
                "required": ["approval_token", "regions_json_path", "output_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "compute_kongnet_cross_g_function",
            "title": "Compute KongNet Cross-G Function",
            "description": "Computes the empirical cross-G function between source and target KongNet cell classes. This reports the probability that a source cell has at least one target cell within each radius, useful for tumour-immune proximity and immune-exclusion analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_json_path": {"type": "string"},
                    "output_txt_path": {"type": "string"},
                    "output_csv_path": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "source_types": {"type": "array", "items": {"type": "string"}},
                    "target_types": {"type": "array", "items": {"type": "string"}},
                    "radii": {"type": "array", "items": {"type": "number"}},
                    "distance_units": {"type": "string", "enum": ["microns", "pixels"]},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "min_probability": {"type": "number"}
                },
                "required": ["approval_token", "annotationstore_path", "output_json_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "characterize_kongnet_cell_neighbourhoods",
            "title": "Characterise KongNet Cell Neighbourhoods",
            "description": "Counts every predicted cell type around each individual cell and clusters similar local profiles into interpretable spatial communities.",
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
                    "min_probability": {"type": "number"},
                    "community_count": {"type": "integer"}
                },
                "required": ["approval_token", "annotationstore_path", "output_csv_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "export_kongnet_regions_to_annotationstore",
            "title": "Export KongNet ROI Boundaries For TIAViz",
            "description": "Converts fixed ROI results into coloured rectangular annotations in baseline WSI coordinates. Save this DB beside the nucleus AnnotationStore to view both on the real slide in TIAViz.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "regions_json_path": {"type": "string"},
                    "output_db_path": {"type": "string"},
                    "wsi_path": {"type": "string"},
                    "mpp": {"type": "number"},
                    "overwrite": {"type": "boolean"}
                },
                "required": ["approval_token", "regions_json_path", "output_db_path"],
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
                    "regions_json_path": {"type": "string"},
                    "communities_json_path": {"type": "string"},
                    "rankings_json_path": {"type": "string"},
                    "slide_summary_json_path": {"type": "string"},
                    "point_pattern_json_path": {"type": "string"},
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
            "name": "generate_nucleus_instance_segmentation_report",
            "title": "Generate Nucleus Instance Segmentation Report",
            "description": "Generates and saves a plain-text (.txt) report for HoVer-Net/nucleus instance segmentation AnnotationStore or CSV outputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "approval_token": {"type": "string"},
                    "annotationstore_path": {"type": "string"},
                    "output_report_path": {
                        "type": "string",
                        "description": "Destination text file. A .txt suffix is enforced even if another extension is supplied."
                    },
                    "min_probability": {"type": "number"}
                },
                "required": ["approval_token", "annotationstore_path"],
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
            geojson_path = args.get("geojson_path", "")
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
                    output_dir=output_dir.strip(),
                    geojson_path=geojson_path.strip() if isinstance(geojson_path, str) else "",
                    threshold_parameters=(
                        args.get("threshold_parameters")
                        if isinstance(args.get("threshold_parameters"), dict)
                        else None
                    ),
                    feature_parameters=(
                        args.get("feature_parameters")
                        if isinstance(args.get("feature_parameters"), dict)
                        else None
                    ),
                    clarification_plan_id=(
                        str(args.get("clarification_plan_id"))
                        if args.get("clarification_plan_id")
                        else None
                    ),
                )
            )
            return

        if name == "approve_pathology_plan":
            plan_id = args.get("plan_id")
            confirmation = args.get("confirmation")
            if not isinstance(plan_id, str) or not plan_id.strip():
                tool_error(req_id, 'approve_pathology_plan requires "plan_id".')
                return
            tool_result(
                req_id,
                approve_plan(
                    plan_id=plan_id.strip(),
                    confirmation=str(confirmation or "")
                )
            )
            return

        if name == "build_common_spatial_features":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(req_id, tool_build_common_spatial_features(
                source_path=args.get("source_path", ""), model_family=args.get("model_family", ""),
                model_name=args.get("model_name", ""), output_dir=args.get("output_dir", ""),
                region_size=float(args["region_size"]), min_probability=float(args["min_probability"]),
                distance_units=str(args.get("distance_units", "pixels")), wsi_path=args.get("wsi_path"),
                mpp=float(mpp) if mpp is not None else None,
                min_recommended_rois=int(args.get("min_recommended_rois", 9)),
            ))
            return

        if name == "validate_spatial_capabilities":
            require_plan(args, name)
            tool_result(req_id, tool_validate_spatial_capabilities(
                capabilities_json_path=args.get("capabilities_json_path", ""), statistic=args.get("statistic", ""),
                feature_name=args.get("feature_name"), source_types=args.get("source_types"), target_types=args.get("target_types"),
            ))
            return

        if name == "compute_common_roi_morans_i":
            require_plan(args, name)
            tool_result(req_id, tool_compute_common_roi_morans_i(
                common_spatial_json_path=args.get("common_spatial_json_path", ""), output_json_path=args.get("output_json_path", ""),
                output_txt_path=args.get("output_txt_path"), metrics=args.get("metrics"),
                weights_method=str(args.get("weights_method", "queen")), k_neighbours=int(args.get("k_neighbours", 4)),
                distance_threshold=float(args["distance_threshold"]) if args.get("distance_threshold") is not None else None,
                permutations=int(args.get("permutations", 999)), alpha=float(args["alpha"]),
                min_rois=int(args.get("min_rois", 9)),
            ))
            return

        if name == "compute_common_point_pattern_statistics":
            require_plan(args, name)
            tool_result(req_id, tool_compute_common_point_pattern_statistics(
                common_spatial_json_path=args.get("common_spatial_json_path", ""), output_json_path=args.get("output_json_path", ""),
                output_txt_path=args.get("output_txt_path"), cell_types=args.get("cell_types"),
                source_types=args.get("source_types"), target_types=args.get("target_types"), radii=args.get("radii"),
                quadrat_grid_size=int(args.get("quadrat_grid_size", 4)), min_points_per_pattern=int(args.get("min_points_per_pattern", 10)),
                min_probability=float(args.get("min_probability", 0.0)),
            ))
            return

        if name == "compute_common_cross_g":
            require_plan(args, name)
            tool_result(req_id, tool_compute_common_cross_g(
                common_spatial_json_path=args.get("common_spatial_json_path", ""), output_json_path=args.get("output_json_path", ""),
                output_csv_path=args.get("output_csv_path"), source_types=args.get("source_types"), target_types=args.get("target_types"),
                radii=args.get("radii"), min_probability=float(args.get("min_probability", 0.0)),
            ))
            return

        if name == "compute_common_cooccurrence":
            require_plan(args, name)
            tool_result(req_id, tool_compute_common_cooccurrence(
                common_spatial_json_path=args.get("common_spatial_json_path", ""), output_json_path=args.get("output_json_path", ""),
                radius=float(args["radius"]), cell_types=args.get("cell_types"), min_probability=float(args.get("min_probability", 0.0)),
            ))
            return

        if name == "compute_common_neighbour_distances":
            require_plan(args, name)
            tool_result(req_id, tool_compute_common_neighbour_distances(
                common_spatial_json_path=args.get("common_spatial_json_path", ""), output_json_path=args.get("output_json_path", ""),
                source_types=args.get("source_types"), target_types=args.get("target_types"),
                radius=float(args["radius"]) if args.get("radius") is not None else None,
                min_probability=float(args.get("min_probability", 0.0)),
            ))
            return

        if name == "compute_common_roi_entropy":
            require_plan(args, name)
            tool_result(req_id, tool_compute_common_roi_entropy(
                common_spatial_json_path=args.get("common_spatial_json_path", ""), output_json_path=args.get("output_json_path", ""),
                output_txt_path=args.get("output_txt_path"), output_csv_path=args.get("output_csv_path"),
                cell_types=args.get("cell_types"), normalize=bool_arg(args.get("normalize"), True),
                entropy_base=float(args.get("entropy_base", 2.718281828459045)),
                low_threshold=float(args["low_threshold"]), high_threshold=float(args["high_threshold"]),
            ))
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
            require_plan(args, name)
            tool_result(req_id, tool_wsi_metadata(args.get("path", "")))
            return

        if name == "validate_qupath_roi_pair":
            token = require_plan(args, name)
            plan = APPROVED_PLANS[token]
            validation = plan.get("roi_validation") or {}
            image_path = args.get("image_path", "")
            geojson_path = args.get("geojson_path", "")
            output_json_path = args.get("output_json_path", "")
            if plan.get("requires_roi_validation"):
                expected_image = os.path.normcase(os.path.abspath(validation["image_path"]))
                expected_geojson = os.path.normcase(os.path.abspath(validation["geojson_path"]))
                actual_image = os.path.normcase(os.path.abspath(image_path))
                actual_geojson = os.path.normcase(os.path.abspath(geojson_path))
                if actual_image != expected_image or actual_geojson != expected_geojson:
                    raise RuntimeError(
                        "Validation paths must match the exact TIFF/GeoJSON pair in the "
                        "approved plan. Propose a new plan to validate a different pair."
                    )
                output_json_path = validation["output_json_path"]

            result = tool_validate_qupath_roi_pair(
                image_path=image_path,
                geojson_path=geojson_path,
                output_json_path=output_json_path,
                feature_id=args.get("feature_id"),
                dimension_tolerance_pixels=float(
                    args.get("dimension_tolerance_pixels", 2.0)
                ),
            )
            with open(output_json_path, "r", encoding="utf-8") as validation_file:
                manifest = json.load(validation_file)
            validation["status"] = str(manifest.get("status", "failed"))
            validation["completed_at"] = utc_now_iso()
            validation["error_count"] = len(manifest.get("errors", []))
            validation["warning_count"] = len(manifest.get("warnings", []))
            plan["roi_validation"] = validation
            tool_result(
                req_id,
                result
            )
            return

        if name == "wsi_thumbnail":
            require_plan(args, name)
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
            require_plan(args, name)
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
            require_plan(args, name)

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
            require_plan(args, name)
            tool_result(
                req_id,
                tool_analyze_patch_statistics(
                    patch_dir=args.get("patch_dir", ""),
                    output_path=args.get("output_path")
                )
            )
            return

        if name in {"predict_patch_model", "predict_kather_resnet18"}:
            require_plan(args, name)

            run_async = bool_arg(args.get("run_async"), True)
            wait = bool_arg(args.get("wait"), False)

            if run_async and not wait:
                job_args = dict(args)
                job_args["job_tool"] = "predict_patch_model"
                tool_result(req_id, start_prediction_job(job_args))
            else:
                tool_result(
                    req_id,
                    tool_predict_patch_model(
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
            require_plan(args, name)

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

        if name == "predict_nucleus_instance_segmentation":
            require_plan(args, name)

            run_async = bool_arg(args.get("run_async"), True)
            wait = bool_arg(args.get("wait"), False)

            if run_async and not wait:
                job_args = dict(args)
                job_args["job_tool"] = "predict_nucleus_instance_segmentation"
                tool_result(req_id, start_prediction_job(job_args))
            else:
                num_workers = args.get("num_workers")
                tool_result(
                    req_id,
                    tool_predict_nucleus_instance_segmentation(
                        wsi_path=args.get("wsi_path", ""),
                        output_json_path=args.get("output_json_path"),
                        model_name=str(args.get("model_name", "hovernet_fast-monusac")),
                        batch_size=int(args.get("batch_size", 8)),
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

        if name == "predict_multi_task_segmentation":
            require_plan(args, name)

            run_async = bool_arg(args.get("run_async"), True)
            wait = bool_arg(args.get("wait"), False)

            if run_async and not wait:
                job_args = dict(args)
                job_args["job_tool"] = "predict_multi_task_segmentation"
                tool_result(req_id, start_prediction_job(job_args))
            else:
                num_workers = args.get("num_workers")
                tool_result(
                    req_id,
                    tool_predict_multi_task_segmentation(
                        wsi_path=args.get("wsi_path", ""),
                        output_json_path=args.get("output_json_path"),
                        model_name=str(args.get("model_name", "hovernetplus-oed")),
                        batch_size=int(args.get("batch_size", 8)),
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

        if name == "predict_semantic_segmentation":
            require_plan(args, name)

            run_async = bool_arg(args.get("run_async"), True)
            wait = bool_arg(args.get("wait"), False)

            if run_async and not wait:
                job_args = dict(args)
                job_args["job_tool"] = "predict_semantic_segmentation"
                tool_result(req_id, start_prediction_job(job_args))
            else:
                num_workers = args.get("num_workers")
                tool_result(
                    req_id,
                    tool_predict_semantic_segmentation(
                        wsi_path=args.get("wsi_path", ""),
                        output_json_path=args.get("output_json_path"),
                        model_name=str(args.get("model_name", "fcn_resnet50_unet-bcss")),
                        batch_size=int(args.get("batch_size", 8)),
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
            require_plan(args, name)
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
            require_plan(args, name)
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

        if name == "run_kongnet_spatial_workflow":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_run_kongnet_spatial_workflow(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_dir=args.get("output_dir", ""),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    min_probability=float(args.get("min_probability", 0.0)),
                    neighbourhood_radius=float(args.get("neighbourhood_radius", 50.0)),
                    point_pattern_radii=args.get("point_pattern_radii"),
                    cooccurrence_cell_types=args.get("cooccurrence_cell_types"),
                    neighbourhood_source_types=args.get("neighbourhood_source_types"),
                    neighbourhood_target_types=args.get("neighbourhood_target_types"),
                    point_pattern_cell_types=args.get("point_pattern_cell_types"),
                    point_pattern_source_types=args.get("point_pattern_source_types"),
                    point_pattern_target_types=args.get("point_pattern_target_types"),
                    region_size=(
                        None
                        if args.get("region_size", 100.0) in (None, "auto")
                        else float(args.get("region_size", 100.0))
                    ),
                    min_cells_per_region=int(args.get("min_cells_per_region", 1)),
                    community_count=int(args.get("community_count", 4)),
                    pathology_question=args.get("pathology_question"),
                    overwrite=bool_arg(args.get("overwrite"), True),
                )
            )
            return

        if name == "rank_kongnet_regions":
            require_plan(args, name)
            tool_result(
                req_id,
                tool_rank_kongnet_regions(
                    regions_json_path=args.get("regions_json_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_txt_path=args.get("output_txt_path"),
                    top_k=int(args.get("top_k", 5)),
                )
            )
            return

        if name == "answer_kongnet_spatial_question":
            require_plan(args, name)
            tool_result(
                req_id,
                tool_answer_kongnet_spatial_question(
                    question=args.get("question", ""),
                    regions_json_path=args.get("regions_json_path", ""),
                    output_txt_path=args.get("output_txt_path"),
                    top_k=int(args.get("top_k", 5)),
                )
            )
            return

        if name == "generate_kongnet_region_heatmaps":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_generate_kongnet_region_heatmaps(
                    regions_json_path=args.get("regions_json_path", ""),
                    output_dir=args.get("output_dir", ""),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    overwrite=bool_arg(args.get("overwrite"), True),
                )
            )
            return

        if name == "generate_kongnet_point_pattern_overlays":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_generate_kongnet_point_pattern_overlays(
                    point_pattern_json_path=args.get("point_pattern_json_path", ""),
                    output_dir=args.get("output_dir", ""),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    overwrite=bool_arg(args.get("overwrite"), True),
                )
            )
            return

        if name == "generate_kongnet_slide_summary":
            require_plan(args, name)
            tool_result(
                req_id,
                tool_generate_kongnet_slide_summary(
                    nuclei_csv_path=args.get("nuclei_csv_path", ""),
                    regions_json_path=args.get("regions_json_path", ""),
                    output_txt_path=args.get("output_txt_path", ""),
                    output_json_path=args.get("output_json_path"),
                )
            )
            return

        if name == "find_cells_within_radius":
            require_plan(args, name)
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
            require_plan(args, name)
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
            require_plan(args, name)
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

        if name == "analyze_kongnet_regions":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_analyze_kongnet_regions(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_csv_path=args.get("output_csv_path"),
                    region_size=(
                        None
                        if args.get("region_size", 100.0) in (None, "auto")
                        else float(args.get("region_size", 100.0))
                    ),
                    neighbourhood_radius=float(args.get("neighbourhood_radius", 50.0)),
                    distance_units=str(args.get("distance_units", "microns")),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    min_cells_per_region=int(args.get("min_cells_per_region", 1)),
                    min_probability=float(args.get("min_probability", 0.0)),
                )
            )
            return

        if name == "compute_kongnet_point_pattern_statistics":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_compute_kongnet_point_pattern_statistics(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_txt_path=args.get("output_txt_path"),
                    regions_json_path=args.get("regions_json_path"),
                    distance_units=str(args.get("distance_units", "microns")),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    radii=args.get("radii"),
                    quadrat_grid_size=int(args.get("quadrat_grid_size", 4)),
                    min_points_per_pattern=int(args.get("min_points_per_pattern", 1)),
                    min_probability=float(args.get("min_probability", 0.0)),
                )
            )
            return

        if name == "compute_kongnet_morans_i":
            require_plan(args, name)
            tool_result(
                req_id,
                tool_compute_kongnet_morans_i(
                    regions_json_path=args.get("regions_json_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_txt_path=args.get("output_txt_path"),
                    metrics=args.get("metrics"),
                    weights_method=str(args.get("weights_method", "queen")),
                    k_neighbours=int(args.get("k_neighbours", 4)),
                    distance_threshold=(
                        float(args["distance_threshold"])
                        if args.get("distance_threshold") is not None
                        else None
                    ),
                    permutations=int(args.get("permutations", 999)),
                    alpha=float(args.get("alpha", 0.05)),
                )
            )
            return

        if name == "compute_kongnet_local_morans_i":
            require_plan(args, name)
            tool_result(
                req_id,
                tool_compute_kongnet_local_morans_i(
                    regions_json_path=args.get("regions_json_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_csv_path=args.get("output_csv_path"),
                    output_txt_path=args.get("output_txt_path"),
                    output_annotationstore_path=args.get("output_annotationstore_path"),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(args["mpp"]) if args.get("mpp") is not None else None,
                    overwrite=bool_arg(args.get("overwrite"), True),
                    metric=str(args.get("metric", "")),
                    weights_method=str(args.get("weights_method", "distance")),
                    k_neighbours=int(args.get("k_neighbours", 4)),
                    distance_threshold=(
                        float(args["distance_threshold"])
                        if args.get("distance_threshold") is not None
                        else None
                    ),
                    permutations=int(args.get("permutations", 999)),
                    alpha=float(args.get("alpha", 0.05)),
                    seed=int(args.get("seed", 42)),
                )
            )
            return

        if name == "compute_kongnet_spatial_entropy":
            require_plan(args, name)
            tool_result(
                req_id,
                tool_compute_kongnet_spatial_entropy(
                    regions_json_path=args.get("regions_json_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_txt_path=args.get("output_txt_path"),
                    output_csv_path=args.get("output_csv_path"),
                    normalize=bool_arg(args.get("normalize"), True),
                    entropy_base=float(args.get("entropy_base", 2.718281828459045)),
                    low_threshold=float(args.get("low_threshold", 0.40)),
                    high_threshold=float(args.get("high_threshold", 0.70)),
                    cell_types=args.get("cell_types"),
                )
            )
            return

        if name == "compute_kongnet_cross_g_function":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_compute_kongnet_cross_g_function(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_json_path=args.get("output_json_path", ""),
                    output_txt_path=args.get("output_txt_path"),
                    output_csv_path=args.get("output_csv_path"),
                    regions_json_path=args.get("regions_json_path"),
                    source_types=args.get("source_types"),
                    target_types=args.get("target_types"),
                    radii=args.get("radii"),
                    distance_units=str(args.get("distance_units", "microns")),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    min_probability=float(args.get("min_probability", 0.0)),
                )
            )
            return

        if name == "characterize_kongnet_cell_neighbourhoods":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_characterize_kongnet_cell_neighbourhoods(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_csv_path=args.get("output_csv_path", ""),
                    output_json_path=args.get("output_json_path"),
                    radius=float(args.get("radius", 50.0)),
                    distance_units=str(args.get("distance_units", "microns")),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    min_probability=float(args.get("min_probability", 0.0)),
                    community_count=int(args.get("community_count", 4)),
                )
            )
            return

        if name == "export_kongnet_regions_to_annotationstore":
            require_plan(args, name)
            mpp = args.get("mpp")
            tool_result(
                req_id,
                tool_export_kongnet_regions_to_annotationstore(
                    regions_json_path=args.get("regions_json_path", ""),
                    output_db_path=args.get("output_db_path", ""),
                    wsi_path=args.get("wsi_path"),
                    mpp=float(mpp) if mpp is not None else None,
                    overwrite=bool_arg(args.get("overwrite"), True),
                )
            )
            return

        if name == "aggregate_kather_metrics":
            require_plan(args, name)
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
            require_plan(args, name)
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
            require_plan(args, name)
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
            require_plan(args, name)
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
            require_plan(args, name)
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
            require_plan(args, name)
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
            require_plan(args, name)
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
            require_plan(args, name)
            tool_result(
                req_id,
                tool_generate_kongnet_ai_report(
                    nuclei_csv_path=args.get("nuclei_csv_path", ""),
                    cooccurrence_json_path=args.get("cooccurrence_json_path"),
                    neighbourhood_json_path=args.get("neighbourhood_json_path"),
                    nearest_neighbour_json_path=args.get("nearest_neighbour_json_path"),
                    regions_json_path=args.get("regions_json_path"),
                    communities_json_path=args.get("communities_json_path"),
                    rankings_json_path=args.get("rankings_json_path"),
                    slide_summary_json_path=args.get("slide_summary_json_path"),
                    point_pattern_json_path=args.get("point_pattern_json_path"),
                    output_report_path=args.get("output_report_path"),
                )
            )
            return

        if name == "generate_nucleus_instance_segmentation_report":
            require_plan(args, name)
            min_probability = args.get("min_probability", 0.0)
            tool_result(
                req_id,
                tool_generate_nucleus_instance_segmentation_report(
                    annotationstore_path=args.get("annotationstore_path", ""),
                    output_report_path=args.get("output_report_path"),
                    min_probability=float(min_probability) if min_probability is not None else 0.0,
                )
            )
            return

        

        if name == "save_run_report":
            require_plan(args, name)

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
