"""
tia_tools.py
------------
MCP-only pathology agent tool logic.

Main MVP models:
- resnet18-kather100k patch classification via Hugging Face/timm
- KongNet nucleus detection via TIAToolbox NucleusDetector

Core outputs:
- WSI metadata
- Thumbnail
- Tissue mask
- Patch extraction
- Kather patch predictions
- KongNet nucleus AnnotationStore predictions
- Tissue-class aggregation
- Colour variance
- Patch entropy
- Tissue-class overlay
- Heterogeneity overlay

Post-processing:
- summarize_kather_results
- generate_confidence_histogram
- generate_hotspot_overlay
- compare_masked_vs_unmasked_runs
- generate_tumour_likelihood_map
- threshold_sensitivity_analysis
- extract_top_abnormal_patches
- generate_final_ai_report
"""

import os
import sys
import re
import csv
import json
import math
import random
import shutil
import uuid
from collections import Counter, deque
from typing import Optional, Dict, Any, List


WSI_EXTENSIONS = {
    ".svs",
    ".ndpi",
    ".tif",
    ".tiff",
    ".scn",
    ".mrxs",
    ".vms",
    ".vmu",
    ".bif",
}


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def tool_health() -> str:
    return "health is OK!"


def tool_echo(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError('echo requires a non-empty "text" argument.')
    return text


def tool_list_files(directory: str, max_items: int = 50) -> str:
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("list_files requires a non-empty directory path.")

    os.makedirs(directory, exist_ok=True)

    if not isinstance(max_items, int) or max_items <= 0:
        max_items = 50

    entries = sorted(os.listdir(directory), key=lambda x: x.lower())[:max_items]

    out = []
    for e in entries:
        full = os.path.join(directory, e)
        out.append(("[DIR]  " if os.path.isdir(full) else "[FILE] ") + e)

    return "Items in " + directory + ":\n" + ("\n".join(out) if out else "(empty)")


def tool_wsi_metadata(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError('wsi_metadata requires a non-empty "path" argument.')

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    from tiatoolbox.wsicore.wsireader import WSIReader

    wsi = WSIReader.open(path)

    lines = [
        f"Path: {path}",
        f"Reader: {type(wsi).__name__}",
        f"Dimensions (level 0): {wsi.info.slide_dimensions}",
        f"Level count: {wsi.info.level_count}",
    ]

    try:
        lines.append(f"Level dimensions: {wsi.info.level_dimensions}")
    except Exception:
        pass

    try:
        lines.append(f"MPP: {wsi.info.mpp}")
    except Exception:
        pass

    try:
        lines.append(f"Objective power: {wsi.info.objective_power}")
    except Exception:
        pass

    return "\n".join(lines)


def tool_wsi_thumbnail(
    path: str,
    output_path: str,
    resolution: float = 2.0,
    units: str = "power",
) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError('wsi_thumbnail requires a non-empty "path" argument.')

    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError('wsi_thumbnail requires a non-empty "output_path" argument.')

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ensure_parent_dir(output_path)

    from tiatoolbox.wsicore.wsireader import WSIReader
    import cv2

    wsi = WSIReader.open(path)
    thumb_rgb = wsi.slide_thumbnail(resolution=float(resolution), units=str(units))
    thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(output_path, thumb_bgr)
    if not ok:
        raise IOError(f"Failed to write PNG to: {output_path}")

    return (
        f"Thumbnail generated successfully.\n"
        f"Saved to: {output_path}\n"
        f"Resolution: {resolution} {units}\n"
        f"Thumbnail shape: {thumb_rgb.shape}"
    )


def tool_tissue_mask(
    path: str,
    output_mask_path: str,
    output_overlay_path: Optional[str] = None,
    mpp: float = 2.0,
    method: str = "morphological",
) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError('tissue_mask requires a non-empty "path" argument.')

    if not isinstance(output_mask_path, str) or not output_mask_path.strip():
        raise ValueError('tissue_mask requires a non-empty "output_mask_path" argument.')

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ensure_parent_dir(output_mask_path)
    if output_overlay_path:
        ensure_parent_dir(output_overlay_path)

    from tiatoolbox.wsicore.wsireader import WSIReader
    from tiatoolbox.tools.tissuemask import MorphologicalMasker, OtsuTissueMasker
    import cv2
    import numpy as np

    wsi = WSIReader.open(path)
    thumb_rgb = wsi.slide_thumbnail(resolution=float(mpp), units="mpp")

    if method == "otsu":
        masker = OtsuTissueMasker()
    else:
        masker = MorphologicalMasker(mpp=float(mpp))

    masks = masker.fit_transform([thumb_rgb])
    mask = masks[0]

    mask_uint8 = (mask.astype(bool).astype("uint8")) * 255
    ok = cv2.imwrite(output_mask_path, mask_uint8)

    if not ok:
        raise IOError(f"Failed to save mask to: {output_mask_path}")

    overlay_message = "Overlay: not requested"

    if output_overlay_path:
        overlay = thumb_rgb.copy()
        overlay[mask.astype(bool), 0] = 255
        overlay = (0.65 * overlay + 0.35 * thumb_rgb).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        ok2 = cv2.imwrite(output_overlay_path, overlay_bgr)
        if not ok2:
            raise IOError(f"Mask saved but failed to save overlay to: {output_overlay_path}")

        overlay_message = f"Overlay saved to: {output_overlay_path}"

    tissue_fraction = float(mask.astype(bool).mean())

    lines = [
        "Tissue mask generated successfully.",
        f"Method: {method}",
        f"Mask saved to: {output_mask_path}",
        overlay_message,
        f"Thumbnail size used: {thumb_rgb.shape[1]} x {thumb_rgb.shape[0]}",
        f"Tissue fraction: {tissue_fraction:.4f}",
    ]

    return "\n".join(lines)


def tool_extract_patches(
    path: str,
    output_dir: str,
    patch_size: int = 224,
    stride: Optional[int] = None,
    level: int = 0,
    max_patches: int = 2000,
    min_tissue_fraction: float = 0.15,
    mpp: float = 2.0,
) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError('extract_patches requires a non-empty "path" argument.')

    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError('extract_patches requires a non-empty "output_dir" argument.')

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    os.makedirs(output_dir, exist_ok=True)

    if stride is None:
        stride = max(1, patch_size // 2)

    from tiatoolbox.wsicore.wsireader import WSIReader
    from tiatoolbox.tools.tissuemask import MorphologicalMasker
    import cv2

    wsi = WSIReader.open(path)

    thumb_rgb = wsi.slide_thumbnail(resolution=float(mpp), units="mpp")
    masker = MorphologicalMasker(mpp=float(mpp))
    masks = masker.fit_transform([thumb_rgb])
    tissue_mask_thumb = masks[0].astype(bool)

    thumb_h, thumb_w = tissue_mask_thumb.shape[:2]

    level_dims = wsi.info.level_dimensions
    if level >= len(level_dims):
        raise ValueError(f"Requested level {level}, but slide only has {len(level_dims)} levels.")

    slide_w, slide_h = level_dims[level]

    scale_x = slide_w / float(thumb_w)
    scale_y = slide_h / float(thumb_h)

    xs = list(range(0, max(slide_w - patch_size, 1), stride))
    ys = list(range(0, max(slide_h - patch_size, 1), stride))
    grid = [(x, y) for x in xs for y in ys]
    random.shuffle(grid)

    saved = 0
    coords_saved = []

    for x0, y0 in grid:
        if saved >= max_patches:
            break

        cx_thumb = min(int((x0 + patch_size / 2) / scale_x), thumb_w - 1)
        cy_thumb = min(int((y0 + patch_size / 2) / scale_y), thumb_h - 1)

        if not tissue_mask_thumb[cy_thumb, cx_thumb]:
            continue

        patch_rgb = wsi.read_rect(
            location=(x0, y0),
            size=(patch_size, patch_size),
            resolution=level,
            units="level",
        )

        patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
        tissue_pix = (patch_hsv[:, :, 1] > 20) & (patch_hsv[:, :, 2] < 245)
        frac = float(tissue_pix.mean())

        if frac < min_tissue_fraction:
            continue

        patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, f"patch_l{level}_{x0}_{y0}.png")

        if cv2.imwrite(out_path, patch_bgr):
            saved += 1
            coords_saved.append((x0, y0, frac, out_path))

    if saved == 0:
        raise RuntimeError(
            "No tissue patches saved. Try lowering min_tissue_fraction or choosing a different level."
        )

    lines = [
        f"Extracted {saved} tissue patches successfully.",
        f"Saved to: {output_dir}",
        f"Patch size: {patch_size}px",
        f"Stride: {stride}px",
        f"Level: {level}",
        f"Minimum tissue fraction: {min_tissue_fraction}",
        "",
        "First patches:"
    ]

    for c in coords_saved[:10]:
        lines.append(f"  (x={c[0]}, y={c[1]}), tissue={c[2]:.2f} -> {c[3]}")

    return "\n".join(lines)


def tool_analyze_patch_statistics(
    patch_dir: str,
    output_path: Optional[str] = None,
) -> str:
    if not isinstance(patch_dir, str) or not os.path.isdir(patch_dir):
        raise ValueError("analyze_patch_statistics requires a valid patch_dir.")

    import cv2
    import numpy as np

    patch_files = [
        os.path.join(patch_dir, f)
        for f in os.listdir(patch_dir)
        if f.lower().endswith(".png")
    ]

    if len(patch_files) == 0:
        raise FileNotFoundError("No PNG patches found in directory.")

    means = []
    gray_all = []

    for p in patch_files:
        img = cv2.imread(p)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        means.append(np.mean(img_rgb, axis=(0, 1)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_all.append(gray.reshape(-1))

    if len(means) == 0:
        raise RuntimeError("Failed to read any patches.")

    import numpy as np

    means_arr = np.array(means)
    gray_concat = np.concatenate(gray_all, axis=0)

    patch_colour_variance = float(np.var(means_arr))

    hist, _ = np.histogram(gray_concat, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]

    entropy = -float(np.sum(hist * np.log2(hist)))
    heterogeneity_index = float(entropy * patch_colour_variance)

    result = {
        "patch_dir": patch_dir,
        "patch_count": len(patch_files),
        "patch_colour_variance": patch_colour_variance,
        "shannon_entropy_grayscale": entropy,
        "heterogeneity_index": heterogeneity_index,
    }

    lines = [
        "Patch statistics computed successfully.",
        f"Number of patches analysed: {len(patch_files)}",
        f"Patch-level colour variance: {patch_colour_variance:.6f}",
        f"Shannon entropy grayscale: {entropy:.6f}",
        f"Heterogeneity index: {heterogeneity_index:.6f}",
        "",
        "Interpretation:",
        "Higher variance and entropy suggest greater morphological or staining diversity across sampled tissue.",
    ]

    text = "\n".join(lines)

    if output_path:
        ensure_parent_dir(output_path)

        if output_path.lower().endswith(".json"):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

        text += f"\n\nSaved statistics to: {output_path}"

    return text


_PATCH_RE = re.compile(r"patch_l(?P<level>\d+)_(?P<x>\d+)_(?P<y>\d+)\.png$", re.IGNORECASE)

KATHER_CLASSES = [
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM",
]

KATHER_CLASS_DESCRIPTIONS = {
    "ADI": "adipose tissue",
    "BACK": "background",
    "DEB": "debris",
    "LYM": "lymphocytes",
    "MUC": "mucus",
    "MUS": "smooth muscle",
    "NORM": "normal colon mucosa",
    "STR": "cancer-associated stroma",
    "TUM": "tumour epithelium",
}

KATHER_INDEX_TO_CLASS = {
    0: "BACK",
    1: "NORM",
    2: "DEB",
    3: "TUM",
    4: "ADI",
    5: "MUC",
    6: "MUS",
    7: "STR",
    8: "LYM",
}

KATHER_CLASS_TO_INDEX = {label: idx for idx, label in KATHER_INDEX_TO_CLASS.items()}

KATHER_ANALYSIS_CLASSES = list(KATHER_INDEX_TO_CLASS.values())

KATHER_CLASS_RGB = {
    "ADI": (0, 237, 189),
    "BACK": (255, 0, 179),
    "DEB": (255, 158, 0),
    "LYM": (255, 0, 0),
    "MUC": (0, 255, 0),
    "MUS": (153, 255, 0),
    "NORM": (0, 166, 230),
    "STR": (158, 0, 255),
    "TUM": (0, 0, 255),
}

PATCH_PREDICTION_KATHER_CLASS_DICT = dict(KATHER_INDEX_TO_CLASS)

PATCH_PREDICTION_PCAM_CLASS_DICT = {
    0: "Non-Metastatic Tissue",
    1: "Metastatic Tissue",
}

PATCH_PREDICTION_MODEL_CATALOG = {
    "resnet18-kather100k": {
        "class_dict": PATCH_PREDICTION_KATHER_CLASS_DICT,
        "model_type": "patch_predictor",
        "target_node": "Patch-Level Prediction",
        "description": "Patch-level tissue classification using the Kather100K tissue classes.",
        "primary_site": "colorectal",
        "resolution_mpp": None,
        "patch_shape": None,
        "stride_shape": None,
        "limitation": "Research output only; not a clinical diagnosis.",
        "postprocessing": "kather",
    },
    "wide_resnet50_2-pcam": {
        "class_dict": PATCH_PREDICTION_PCAM_CLASS_DICT,
        "model_type": "patch_predictor",
        "target_node": "Patch-Level Prediction",
        "description": "Predict lymph node metastases status at a patch level in lymph node images.",
        "primary_site": "lymph node",
        "resolution_mpp": None,
        "patch_shape": None,
        "stride_shape": None,
        "limitation": "Patch-level metastasis prediction only; not a clinical diagnosis.",
        "postprocessing": "binary_metastasis",
    },
}


def _patch_prediction_model_metadata(model_name: str) -> Dict[str, Any]:
    try:
        return PATCH_PREDICTION_MODEL_CATALOG[model_name]
    except KeyError as exc:
        valid = ", ".join(sorted(PATCH_PREDICTION_MODEL_CATALOG))
        raise ValueError(
            f"Unsupported patch prediction model_name '{model_name}'. Valid models: {valid}."
        ) from exc


def _patch_prediction_model_summary(model_name: str, model_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "model_type": model_meta.get("model_type", "patch_predictor"),
        "target_node": model_meta.get("target_node", "Patch-Level Prediction"),
        "class_mapping": _class_mapping_from_dict(model_meta.get("class_dict", {})),
        "description": model_meta.get("description", ""),
        "primary_site": model_meta.get("primary_site"),
        "input_resolution": (
            {"units": "mpp", "value": model_meta.get("resolution_mpp")}
            if model_meta.get("resolution_mpp") is not None
            else None
        ),
        "patch_shape": model_meta.get("patch_shape"),
        "stride_shape": model_meta.get("stride_shape"),
        "limitation": model_meta.get("limitation"),
    }


def _parse_patch_filename(path: str) -> Dict[str, int]:
    name = os.path.basename(path)
    match = _PATCH_RE.match(name)

    if not match:
        return {"level": 0, "x": -1, "y": -1}

    return {
        "level": int(match.group("level")),
        "x": int(match.group("x")),
        "y": int(match.group("y")),
    }


def _normalise_class_dict(class_dict: Optional[Dict[Any, Any]] = None) -> Dict[int, str]:
    if not class_dict:
        return dict(KATHER_INDEX_TO_CLASS)

    normalised: Dict[int, str] = {}
    for key, value in class_dict.items():
        try:
            normalised[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return normalised or dict(KATHER_INDEX_TO_CLASS)


def _probability_for_class(prob_vec: List[float], class_label: str, class_dict: Dict[int, str]) -> float:
    for idx, label in class_dict.items():
        if label == class_label and idx < len(prob_vec):
            return float(prob_vec[idx])
    return 0.0


def _prediction_abnormality_score(prob_vec: List[float], class_dict: Dict[int, str]) -> float:
    tumour = _probability_for_class(prob_vec, "TUM", class_dict)
    stroma = _probability_for_class(prob_vec, "STR", class_dict)
    lymphocyte = _probability_for_class(prob_vec, "LYM", class_dict)
    debris = _probability_for_class(prob_vec, "DEB", class_dict)
    return float(min(1.0, tumour + stroma + lymphocyte + 0.5 * debris))


def _load_kather_predictions(
    predictions_json_path: str,
    class_dict: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    if not os.path.exists(predictions_json_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_json_path}")

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    resolved_class_dict = _normalise_class_dict(
        class_dict or data.get("class_dict") or data.get("label_dict")
    )

    preds = data.get("predictions", [])
    probabilities = data.get("probabilities")
    coordinates = data.get("coordinates")

    # New TIAToolbox WSI raw format: predictions/probabilities/coordinates
    # are parallel arrays. Convert them to the older patch-row shape used by
    # downstream post-processing tools.
    if (
        isinstance(preds, list)
        and preds
        and not isinstance(preds[0], dict)
        and isinstance(probabilities, list)
        and isinstance(coordinates, list)
    ):
        rows = []
        for idx, (pred, prob_vec, coord) in enumerate(zip(preds, probabilities, coordinates)):
            if not isinstance(prob_vec, list) or len(coord) < 2:
                continue

            pred_idx = int(pred)
            label = resolved_class_dict.get(pred_idx, str(pred_idx))
            confidence = float(prob_vec[pred_idx]) if pred_idx < len(prob_vec) else 0.0
            x0 = int(float(coord[0]))
            y0 = int(float(coord[1]))
            x1 = int(float(coord[2])) if len(coord) > 2 else x0
            y1 = int(float(coord[3])) if len(coord) > 3 else y0

            tumour_prob = _probability_for_class(prob_vec, "TUM", resolved_class_dict)
            stroma_prob = _probability_for_class(prob_vec, "STR", resolved_class_dict)
            lymphocyte_prob = _probability_for_class(prob_vec, "LYM", resolved_class_dict)
            abnormality_score = _prediction_abnormality_score(prob_vec, resolved_class_dict)

            row = {
                "filename": f"patch_{idx:06d}",
                "patch_path": "",
                "level": 0,
                "x": x0,
                "y": y0,
                "x1": x1,
                "y1": y1,
                "predicted_class": label,
                "predicted_class_description": KATHER_CLASS_DESCRIPTIONS.get(label, label),
                "class_index": pred_idx,
                "confidence": confidence,
                "tumour_epithelium_probability": tumour_prob,
                "stroma_probability": stroma_prob,
                "lymphocyte_probability": lymphocyte_prob,
                "abnormality_score": abnormality_score,
                "tumour_likelihood_score": tumour_prob,
            }
            for class_idx, class_label in resolved_class_dict.items():
                if class_idx < len(prob_vec):
                    row[f"prob_{class_label}"] = float(prob_vec[class_idx])
            rows.append(row)

        return {
            "model_name": data.get("pretrained_model", data.get("model_name", "resnet18-kather100k")),
            "predictions": rows,
            "class_dict": resolved_class_dict,
            "source_format": "tiatoolbox_raw_wsi",
            "raw": data,
        }

    # Older already-normalised format.
    rows = preds if isinstance(preds, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prob_vec = [
            float(row.get(f"prob_{resolved_class_dict[i]}", 0.0))
            for i in sorted(resolved_class_dict)
        ]
        label = str(row.get("predicted_class", row.get("type", "")))
        if "predicted_class_description" not in row:
            row["predicted_class_description"] = KATHER_CLASS_DESCRIPTIONS.get(label, label)
        if "tumour_epithelium_probability" not in row:
            row["tumour_epithelium_probability"] = _probability_for_class(prob_vec, "TUM", resolved_class_dict)
        if "stroma_probability" not in row:
            row["stroma_probability"] = _probability_for_class(prob_vec, "STR", resolved_class_dict)
        if "lymphocyte_probability" not in row:
            row["lymphocyte_probability"] = _probability_for_class(prob_vec, "LYM", resolved_class_dict)
        if "abnormality_score" not in row:
            row["abnormality_score"] = _prediction_abnormality_score(prob_vec, resolved_class_dict)
        if "tumour_likelihood_score" not in row:
            row["tumour_likelihood_score"] = float(row.get("tumour_epithelium_probability", 0.0))

    return {
        "model_name": data.get("model_name", data.get("pretrained_model", "resnet18-kather100k")),
        "predictions": rows,
        "class_dict": resolved_class_dict,
        "source_format": "normalised_patch_rows",
        "raw": data,
    }


def _choose_device(device: str) -> str:
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    return device


def _load_pretrained_model(model_name: str, device: str):
    import timm
    from tiatoolbox.models.engine.patch_predictor import PatchPredictor

    if model_name in PATCH_PREDICTION_MODEL_CATALOG:
        # hf_model_name = "hf-hub:1aurent/resnet18.tiatoolbox-kather100k"

        # model = timm.create_model(
        #     hf_model_name,
        #     pretrained=True,
        # )

        # model.eval()
        # model.to(device)
        predictor = _create_patch_predictor(PatchPredictor, model_name, batch_size=32)

        return predictor

    valid = ", ".join(sorted(PATCH_PREDICTION_MODEL_CATALOG))
    raise ValueError(f"Unsupported model_name: {model_name}. Valid models: {valid}.")


def _create_patch_predictor(predictor_cls, model_name: str, batch_size: int):
    import inspect

    try:
        params = inspect.signature(predictor_cls).parameters
    except Exception:
        params = {}

    if "pretrained_model" in params:
        return predictor_cls(
            pretrained_model=model_name,
            batch_size=int(batch_size),
        )

    return predictor_cls(
        model=model_name,
        batch_size=int(batch_size),
    )


def _preprocess_patch_for_kather(img_rgb, input_size: int = 224):
    import cv2
    import torch
    import numpy as np

    if img_rgb.shape[0] != input_size or img_rgb.shape[1] != input_size:
        img_rgb = cv2.resize(img_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)

    arr = img_rgb.astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))

    return torch.from_numpy(arr).float()


def _prediction_entropy(probs: List[float]) -> float:
    eps = 1e-12
    vals = [max(float(p), eps) for p in probs]
    total = sum(vals)

    if total <= 0:
        return 0.0

    vals = [v / total for v in vals]
    return float(-sum(v * math.log2(v) for v in vals))


def _call_patch_predictor_wsi_compatible(
    predictor,
    wsi_path: str,
    ioconfig: Optional[Dict[str, Any]],
    save_dir: str,
    selected_device: str,
    class_dict: Dict[int, str],
    input_size: int = 224,
    resolution: float = 0.5,
    units: str = "mpp",
):
    """Call PatchPredictor using the API supported by the installed TIAToolbox.

    TIAToolbox releases differ on both the inference method name
    (run vs predict) and the WSI input keyword (images vs imgs/wsis). This
    wrapper keeps the MCP server compatible without requiring manual edits.
    """
    import inspect
    from pathlib import Path

    method_names = []
    if hasattr(predictor, "predict"):
        method_names.append("predict")
    if hasattr(predictor, "run"):
        method_names.append("run")

    if not method_names:
        raise AttributeError("PatchPredictor has neither predict() nor run().")

    ext = Path(wsi_path).suffix.lower()
    is_wsi = ext in WSI_EXTENSIONS

    input_keywords = ["imgs", "wsis", "images"] if is_wsi else ["imgs", "images", "wsis"]
    base_kwargs = {
        "masks": None,
        "ioconfig": ioconfig,
        "return_probabilities": True,
        "save_dir": save_dir,
        "device": selected_device,
        "class_dict": class_dict,
        "output_type": "annotationstore",
    }
    if is_wsi:
        # TIAToolbox PatchPredictor.predict defaults to mode="patch". Passing
        # a .svs path without mode="wsi" routes into _predict_patch and fails
        # with "Cannot load image data from .svs files".
        base_kwargs.update({
            "mode": "wsi",
            "save_output": True,
            "patch_input_shape": (int(input_size), int(input_size)),
            "stride_shape": (int(input_size), int(input_size)),
            "resolution": float(resolution),
            "units": str(units),
        })
    else:
        base_kwargs["patch_mode"] = False

    errors = []

    for method_name in method_names:
        method = getattr(predictor, method_name)
        try:
            sig = inspect.signature(method)
            params = sig.parameters
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except Exception:
            params = {}
            accepts_var_kwargs = True

        for input_key in input_keywords:
            if params and input_key not in params and not accepts_var_kwargs:
                continue

            input_path = Path(wsi_path) if is_wsi else wsi_path
            kwargs = {input_key: [input_path]}
            for key, value in base_kwargs.items():
                if accepts_var_kwargs or not params or key in params:
                    kwargs[key] = value

            try:
                return method(**kwargs), method_name, input_key
            except (TypeError, ValueError) as exc:
                errors.append(f"{method_name}({input_key}=..., mode={base_kwargs.get('mode')!r}): {exc}")
                continue

    raise TypeError(
        "Could not call PatchPredictor with this TIAToolbox version. Tried:\n- "
        + "\n- ".join(errors)
    )

def _json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _detect_wsi_prediction_output_type(output: Any) -> str:
    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, dict) and "raw" in value:
                return "raw_json"
    return "annotationstore"


def _extract_raw_prediction_paths(output: Any) -> List[str]:
    paths = []
    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, dict) and isinstance(value.get("raw"), str):
                paths.append(value["raw"])
    return paths


def _default_annotationstore_path(
    output_json_path: Optional[str],
    save_dir: str,
    wsi_path: Optional[str] = None,
) -> str:
    if output_json_path:
        base_dir = os.path.dirname(output_json_path) or "."
        stem = (
            os.path.splitext(os.path.basename(wsi_path))[0]
            if wsi_path
            else os.path.splitext(os.path.basename(output_json_path))[0]
        )
        return os.path.join(base_dir, f"{stem}_annotationstore.db")
    return os.path.join(os.path.dirname(save_dir) or ".", "kather_predictions_annotationstore.db")


def convert_kather_raw_json_to_annotationstore(
    raw_json_path: str,
    output_db_path: str,
    class_dict: Dict[int, str],
    wsi_path: Optional[str] = None,
    merge_regions: bool = False,
    smooth_radius: float = 0.0,
) -> Dict[str, Any]:
    from shapely.geometry import MultiPolygon, Polygon, box
    from shapely.ops import unary_union
    from tiatoolbox.annotation.storage import Annotation, SQLiteStore

    if not os.path.exists(raw_json_path):
        raise FileNotFoundError(f"Raw prediction JSON not found: {raw_json_path}")

    ensure_parent_dir(output_db_path)
    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    coordinates = data.get("coordinates", [])
    predictions = data.get("predictions", [])
    probabilities = data.get("probabilities", [])
    model_resolution = float(data.get("resolution", 1.0))
    model_units = str(data.get("units", "baseline")).lower()
    scale_x = 1.0
    scale_y = 1.0

    if wsi_path and model_units == "mpp":
        from tiatoolbox.wsicore.wsireader import WSIReader

        wsi = WSIReader.open(wsi_path)
        mpp = getattr(wsi.info, "mpp", None)
        if mpp is not None:
            scale_x = model_resolution / float(mpp[0])
            scale_y = model_resolution / float(mpp[1])

    if not (
        len(coordinates) == len(predictions) == len(probabilities)
    ):
        raise ValueError(
            "Raw prediction JSON has mismatched coordinates, predictions, "
            "and probabilities lengths."
        )

    annotations = []
    keys = []
    grouped_patches: Dict[str, List[Any]] = {}
    grouped_properties: Dict[str, Dict[str, Any]] = {}
    grouped_confidence: Dict[str, List[float]] = {}

    for idx, (coord, pred, prob_vec) in enumerate(
        zip(coordinates, predictions, probabilities)
    ):
        if len(coord) != 4:
            continue

        x0, y0, x1, y1 = [float(v) for v in coord]
        x0 *= scale_x
        x1 *= scale_x
        y0 *= scale_y
        y1 *= scale_y
        pred = int(pred)
        label = class_dict.get(pred, str(pred))
        confidence = float(prob_vec[pred]) if pred < len(prob_vec) else 0.0

        properties = {
            "type": label,
            "label": label,
            "class_index": pred,
            "confidence": confidence,
            "probability": confidence,
            "source": str(data.get("pretrained_model", "resnet18-kather100k")),
            "resolution": model_resolution,
            "units": str(data.get("units", "mpp")),
            "coordinate_space": "baseline",
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

        for class_index, class_name in class_dict.items():
            if class_index < len(prob_vec):
                properties[f"prob_{class_name}"] = float(prob_vec[class_index])

        patch_geom = box(x0, y0, x1, y1)
        if merge_regions:
            grouped_patches.setdefault(label, []).append(patch_geom)
            grouped_properties.setdefault(label, properties)
            grouped_confidence.setdefault(label, []).append(confidence)
        else:
            annotations.append(Annotation(patch_geom, properties=properties))
            keys.append(f"patch_{idx:06d}")

    if merge_regions:
        for label, patch_geoms in grouped_patches.items():
            merged = unary_union(patch_geoms)
            if smooth_radius > 0:
                merged = merged.buffer(smooth_radius).buffer(-smooth_radius)

            if isinstance(merged, Polygon):
                polygons = [merged]
            elif isinstance(merged, MultiPolygon):
                polygons = list(merged.geoms)
            else:
                polygons = [
                    geom for geom in getattr(merged, "geoms", [])
                    if isinstance(geom, Polygon)
                ]

            properties = dict(grouped_properties[label])
            confidences = grouped_confidence[label]
            properties["confidence"] = float(sum(confidences) / len(confidences))
            properties["probability"] = properties["confidence"]
            properties["region_merged"] = True
            properties["patch_count"] = len(patch_geoms)

            for region_idx, polygon in enumerate(polygons):
                if polygon.is_empty:
                    continue
                annotations.append(Annotation(polygon, properties=properties))
                keys.append(f"region_{label}_{region_idx:05d}")

    store = SQLiteStore(output_db_path)
    try:
        store.append_many(annotations, keys=keys)
        store.commit()
        annotation_count = len(store)
    finally:
        store.close()

    return {
        "annotationstore_path": output_db_path,
        "raw_json_path": raw_json_path,
        "annotation_count": annotation_count,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "merge_regions": merge_regions,
        "smooth_radius": smooth_radius,
    }


def tool_predict_patch_model(
    patch_dir: Optional[str] = None,
    output_json_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    model_name: str = "resnet18-kather100k",
    batch_size: int = 64,
    device: str = "auto",
    input_size: int = 224,
    wsi_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    ioconfig: Optional[Dict[str, Any]] = None,
    output_type: str = "annotationstore",
    patch_mode: bool = False,
) -> str:
    """Run WSI-level patch prediction for TIAViz.

    This follows the TIAToolbox WSI prediction workflow:
    1) instantiate PatchPredictor
    2) call the installed compatible PatchPredictor API with
       patch_mode=False and output_type="annotationstore".

    The output is a TIAViz-compatible AnnotationStore (.db) directory.
    """
    from pathlib import Path
    from tiatoolbox.models.engine.patch_predictor import PatchPredictor

    model_meta = _patch_prediction_model_metadata(model_name)

    # Backwards-compatible alias: older MCP calls passed the WSI file path as patch_dir.
    if not wsi_path and patch_dir:
        wsi_path = patch_dir

    if not isinstance(wsi_path, str) or not wsi_path.strip():
        raise ValueError(
            "predict_patch_model requires a WSI file path via wsi_path "
            "or patch_dir as a backwards-compatible alias."
        )

    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    if patch_mode is not False:
        raise ValueError("For TIAViz WSI prediction, patch_mode must be False.")

    if output_type != "annotationstore":
        raise ValueError("For TIAViz compatibility, output_type must be 'annotationstore'.")

    selected_device = _choose_device(device)

    if not save_dir:
        if output_json_path:
            save_dir = os.path.join(
                os.path.dirname(output_json_path) or ".",
                "wsi_predictions_annotationstore",
            )
        else:
            save_dir = os.path.join(os.getcwd(), "wsi_predictions_annotationstore")

    wsi_ext = Path(wsi_path).suffix.lower()
    is_wsi_input = wsi_ext in WSI_EXTENSIONS

    if is_wsi_input:
        parent_dir = os.path.dirname(save_dir)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if os.path.isdir(save_dir):
            try:
                os.rmdir(save_dir)
            except OSError:
                save_dir = f"{save_dir}_{uuid.uuid4().hex[:8]}"
    else:
        os.makedirs(save_dir, exist_ok=True)

    class_dict = model_meta["class_dict"]

    predictor = _create_patch_predictor(
        PatchPredictor,
        model_name=model_name,
        batch_size=int(batch_size),
    )

    # Supports TIAToolbox variants:
    # - predictor.run(images=...)
    # - predictor.predict(images=...)
    # - predictor.predict(imgs=...)
    # - predictor.predict(wsis=...)
    output, predictor_method, input_keyword = _call_patch_predictor_wsi_compatible(
        predictor=predictor,
        wsi_path=wsi_path,
        ioconfig=ioconfig,
        save_dir=save_dir,
        selected_device=selected_device,
        class_dict=class_dict,
        input_size=int(input_size),
    )
    actual_output_type = _detect_wsi_prediction_output_type(output)
    raw_prediction_paths = _extract_raw_prediction_paths(output)
    annotationstore_result = None

    if raw_prediction_paths:
        annotationstore_path = _default_annotationstore_path(
            output_json_path,
            save_dir,
            wsi_path=wsi_path,
        )
        annotationstore_result = convert_kather_raw_json_to_annotationstore(
            raw_json_path=raw_prediction_paths[0],
            output_db_path=annotationstore_path,
            class_dict=class_dict,
            wsi_path=wsi_path,
        )
        actual_output_type = "annotationstore"

    postprocessing_result = None
    if raw_prediction_paths and model_meta.get("postprocessing") == "kather":
        try:
            postprocessing_result = run_kather_postprocessing_pipeline(
                predictions_json_path=raw_prediction_paths[0],
                output_dir=save_dir,
                class_dict=class_dict,
                abnormality_threshold=0.5,
            )
        except Exception as exc:
            postprocessing_result = {
                "error": str(exc),
                "message": (
                    "Prediction and AnnotationStore outputs were created, "
                    "but automatic post-processing failed."
                ),
            }
    elif raw_prediction_paths:
        postprocessing_result = {
            "message": (
                f"Automatic Kather-specific post-processing skipped for {model_name}; "
                "the prediction output and AnnotationStore were still created."
            )
        }

    slides_dir = os.path.dirname(wsi_path) or "."
    overlay_path = (
        os.path.dirname(annotationstore_result["annotationstore_path"])
        if annotationstore_result
        else save_dir
    )
    tiaviz_command = f'tiatoolbox visualize --slides "{slides_dir}" --overlays "{overlay_path}"'

    result = {
        "mode": f"wsi_{actual_output_type}",
        "model_name": model_name,
        "model_description": model_meta["description"],
        "primary_site": model_meta["primary_site"],
        "model_metadata": _patch_prediction_model_summary(model_name, model_meta),
        "wsi_path": wsi_path,
        "save_dir": save_dir,
        "annotationstore_path": (
            annotationstore_result["annotationstore_path"]
            if annotationstore_result
            else None
        ),
        "raw_prediction_paths": raw_prediction_paths,
        "output_type": actual_output_type,
        "device": selected_device,
        "predictor_method": predictor_method,
        "input_keyword": input_keyword,
        "class_dict": class_dict,
        "annotationstore_conversion": annotationstore_result,
        "postprocessing": postprocessing_result,
        "tiaviz_command": tiaviz_command,
        "tiatoolbox_output": _json_safe(output),
        "clinical_warning": (
            "This is tissue-type classification/model-confidence output, "
            "not a clinical diagnosis."
        ),
    }

    if output_json_path:
        ensure_parent_dir(output_json_path)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    lines = [
        "WSI patch prediction completed successfully.",
        f"Model: {model_name}",
        f"WSI: {wsi_path}",
        f"Prediction output saved to: {save_dir}",
        f"Output type: {actual_output_type}",
        (
            f"AnnotationStore saved to: {annotationstore_result['annotationstore_path']}"
            if annotationstore_result
            else "AnnotationStore conversion: not available"
        ),
        (
            f"Post-processing metrics saved to: {postprocessing_result.get('metrics_json')}"
            if isinstance(postprocessing_result, dict) and postprocessing_result.get("metrics_json")
            else f"Post-processing: {postprocessing_result.get('message', 'not available')}"
            if isinstance(postprocessing_result, dict)
            else "Post-processing: not available"
        ),
        f"Device: {selected_device}",
        f"PatchPredictor API used: {predictor_method}({input_keyword}=...)",
        "",
        "Open in TIAViz with:",
        tiaviz_command,
    ]

    if output_json_path:
        lines.append(f"\nRun summary saved to: {output_json_path}")

    lines.append("\nImportant: this output is not a clinical diagnosis.")
    return "\n".join(lines)


def tool_predict_kather_resnet18(
    patch_dir: Optional[str] = None,
    output_json_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    model_name: str = "resnet18-kather100k",
    batch_size: int = 64,
    device: str = "auto",
    input_size: int = 224,
    wsi_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    ioconfig: Optional[Dict[str, Any]] = None,
    output_type: str = "annotationstore",
    patch_mode: bool = False,
) -> str:
    """Backward-compatible wrapper for the general patch prediction tool."""
    return tool_predict_patch_model(
        patch_dir=patch_dir,
        output_json_path=output_json_path,
        output_csv_path=output_csv_path,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        input_size=input_size,
        wsi_path=wsi_path,
        save_dir=save_dir,
        ioconfig=ioconfig,
        output_type=output_type,
        patch_mode=patch_mode,
    )


KONGNET_PANNUKE_CLASS_DICT = {
    0: "Neoplastic",
    1: "Inflammatory",
    2: "Connective",
    3: "Dead",
    4: "Epithelial",
}

KONGNET_CONIC_CLASS_DICT = {
    0: "Neutrophil",
    1: "Epithelial",
    2: "Lymphocyte",
    3: "Plasma",
    4: "Eosinophil",
    5: "Connective",
}

KONGNET_MIDOG_CLASS_DICT = {
    0: "Mitotic_Figure",
}

KONGNET_MONKEY_CLASS_DICT = {
    0: "Overall_Inflammatory",
    1: "Lymphocyte",
    2: "Monocyte",
}

KONGNET_PUMA_T1_CLASS_DICT = {
    0: "Tumour_Cell",
    1: "Lymphocyte",
    2: "Other_Cell",
}

KONGNET_PUMA_T2_CLASS_DICT = {
    0: "Tumour_Cell",
    1: "Lymphocyte",
    2: "Plasma_Cell",
    3: "Histiocyte",
    4: "Melanophage",
    5: "Neutrophil",
    6: "Stroma_Cell",
    7: "Epithelial_Cell",
    8: "Endothelial_Cell",
    9: "Apoptotic_Cell",
}

KONGNET_MODEL_CATALOG = {
    "KongNet_PanNuke_1": {
        "class_dict": KONGNET_PANNUKE_CLASS_DICT,
        "model_type": "nucleus_detection",
        "target_node": "Nucleus Detection",
        "description": (
            "Nucleus detection model for connective, neoplastic, epithelial, "
            "dead, and inflammatory nuclei across multiple cancers."
        ),
        "primary_site": "multiple",
        "resolution_mpp": None,
        "patch_shape": None,
        "stride_shape": None,
        "limitation": "Research output only; not a clinical diagnosis.",
    },
    "KongNet_CoNIC_1": {
        "class_dict": KONGNET_CONIC_CLASS_DICT,
        "model_type": "nucleus_detection",
        "target_node": "Nucleus Detection",
        "description": (
            "Nuclei detection model for neutrophils, epithelial, lymphocytes, "
            "plasma, eosinophil, and connective nuclei in colorectal cancer. "
            "Does not work for other cancer types."
        ),
        "primary_site": "colorectal",
        "resolution_mpp": 0.5,
        "patch_shape": [512, 512],
        "stride_shape": [512, 512],
        "limitation": "Designed for colorectal cancer; do not use as a general cancer model.",
    },
    "KongNet_Det_MIDOG_1": {
        "class_dict": KONGNET_MIDOG_CLASS_DICT,
        "model_type": "nucleus_detection",
        "target_node": "Nucleus Detection",
        "description": (
            "Nuclei detection model for mitotic nuclei detection across "
            "different cancer types."
        ),
        "primary_site": "multi-site",
        "resolution_mpp": 0.5,
        "patch_shape": [512, 512],
        "stride_shape": [512, 512],
        "limitation": "Specialised mitotic-figure detector; does not classify all nuclei.",
    },
    "KongNet_MONKEY_1": {
        "class_dict": KONGNET_MONKEY_CLASS_DICT,
        "model_type": "nucleus_detection",
        "target_node": "Nucleus Detection",
        "description": (
            "Best model for detecting lymphocytes, monocytes, and inflammatory "
            "cells in kidney biopsy images. Does not work for other cancer types."
        ),
        "primary_site": "kidney",
        "resolution_mpp": 0.5,
        "patch_shape": [512, 512],
        "stride_shape": [512, 512],
        "limitation": "Designed for kidney biopsy images; not intended for other cancer types.",
    },
    "KongNet_PUMA_T1_3": {
        "class_dict": KONGNET_PUMA_T1_CLASS_DICT,
        "model_type": "nucleus_detection",
        "target_node": "Nucleus Detection",
        "description": (
            "Nuclei detection model for detecting Tumor and Lymphocyte nuclei "
            "in melanoma histology images. Does not work for other cancer types."
        ),
        "primary_site": "skin",
        "resolution_mpp": 0.5,
        "patch_shape": [512, 512],
        "stride_shape": [512, 512],
        "limitation": "Designed for melanoma histology images; not intended for other cancer types.",
    },
    "KongNet_PUMA_T2_3": {
        "class_dict": KONGNET_PUMA_T2_CLASS_DICT,
        "model_type": "nucleus_detection",
        "target_node": "Nucleus Detection",
        "description": (
            "Nuclei Detection model for detecting tumor, lymphocytes, plasma, "
            "histiocyte, melanophage, neutrophil, stroma, epithelium, "
            "endothelium, and apoptotic nuclei in melanoma images and metastatic "
            "melanoma for Bone, Gastrointestinal tract, Lung, Liver, brain, "
            "soft tissue, and lymph node cancers. Does not work for other cancer types."
        ),
        "primary_site": "skin",
        "resolution_mpp": 0.5,
        "patch_shape": [512, 512],
        "stride_shape": [512, 512],
        "limitation": (
            "Designed for melanoma and selected metastatic melanoma sites; "
            "not intended for other cancer types."
        ),
    },
}

KONGNET_IMMUNE_CELL_TYPES = {
    "Inflammatory",
    "Overall_Inflammatory",
    "Neutrophil",
    "Lymphocyte",
    "Plasma",
    "Plasma_Cell",
    "Histiocyte",
    "Melanophage",
    "Eosinophil",
    "Monocyte",
}
KONGNET_ALL_CLASS_NAMES = []
for _model_meta in KONGNET_MODEL_CATALOG.values():
    for _class_name in _model_meta["class_dict"].values():
        if _class_name not in KONGNET_ALL_CLASS_NAMES:
            KONGNET_ALL_CLASS_NAMES.append(_class_name)


def _kongnet_model_metadata(model_name: str) -> Dict[str, Any]:
    try:
        return KONGNET_MODEL_CATALOG[model_name]
    except KeyError as exc:
        valid = ", ".join(sorted(KONGNET_MODEL_CATALOG))
        raise ValueError(f"Unsupported KongNet model_name '{model_name}'. Valid models: {valid}.") from exc


def _model_class_mapping(model_meta: Dict[str, Any]) -> Dict[str, int]:
    return {label: int(type_id) for type_id, label in model_meta["class_dict"].items()}


def _kongnet_model_summary(model_name: str, model_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return the public model summary used by MCP clients and saved run JSON."""
    return {
        "model_name": model_name,
        "model_type": model_meta.get("model_type", "nucleus_detection"),
        "target_node": model_meta.get("target_node", "Nucleus Detection"),
        "class_mapping": _model_class_mapping(model_meta),
        "description": model_meta.get("description", ""),
        "primary_site": model_meta.get("primary_site"),
        "input_resolution": (
            {"units": "mpp", "value": model_meta.get("resolution_mpp")}
            if model_meta.get("resolution_mpp") is not None
            else None
        ),
        "patch_shape": model_meta.get("patch_shape"),
        "stride_shape": model_meta.get("stride_shape"),
        "limitation": model_meta.get("limitation"),
    }


NUCLEUS_INSTANCE_SEGMENTATION_MONUSAC_CLASS_DICT = {
    0: "Background",
    1: "Epithelial",
    2: "Lymphocyte",
    3: "Macrophage",
    4: "Neutrophil",
}

NUCLEUS_INSTANCE_SEGMENTATION_PANNUKE_CLASS_DICT = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Non-Neoplastic Epithelial",
}

NUCLEUS_INSTANCE_SEGMENTATION_CONSEP_CLASS_DICT = {
    0: "Background",
    1: "Epithelial",
    2: "Inflammatory",
    3: "Spindle-Shaped",
    4: "Miscellaneous",
}

NUCLEUS_INSTANCE_SEGMENTATION_KUMAR_CLASS_DICT = {}

NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG = {
    "hovernet_fast-monusac": {
        "class_dict": NUCLEUS_INSTANCE_SEGMENTATION_MONUSAC_CLASS_DICT,
        "model_type": "nucleus_instance_segmentation",
        "target_node": "Nucleus Instance Segmentation",
        "description": (
            "Instance Segmentation and detection of Epithelial, Lymphocyte, "
            "Macrophage, and Neutrophil nuclei across different cancer types."
        ),
        "primary_site": "multi-site",
        "resolution_mpp": None,
        "patch_shape": None,
        "stride_shape": None,
        "limitation": "Research output only; not a clinical diagnosis.",
    },
    "hovernet_fast-pannuke": {
        "class_dict": NUCLEUS_INSTANCE_SEGMENTATION_PANNUKE_CLASS_DICT,
        "model_type": "nucleus_instance_segmentation",
        "target_node": "Nucleus Instance Segmentation",
        "description": (
            "Instance Segmentation and detection of Neoplastic, Inflammatory, "
            "Connective, Dead, and Non-Neoplastic Epithelial nuclei across "
            "different cancer types."
        ),
        "primary_site": "multi-site",
        "resolution_mpp": None,
        "patch_shape": None,
        "stride_shape": None,
        "limitation": "Research output only; not a clinical diagnosis.",
    },
    "hovernet_original-consep": {
        "class_dict": NUCLEUS_INSTANCE_SEGMENTATION_CONSEP_CLASS_DICT,
        "model_type": "nucleus_instance_segmentation",
        "target_node": "Nucleus Instance Segmentation",
        "description": (
            "Instance Segmentation and detection of Epithelial, Inflammatory, "
            "and Spindle-Shaped nuclei in colorectal images."
        ),
        "primary_site": "colorectal",
        "resolution_mpp": None,
        "patch_shape": None,
        "stride_shape": None,
        "limitation": "Designed for colorectal images; not intended as a general cancer model.",
    },
    "hovernet_original-kumar": {
        "class_dict": NUCLEUS_INSTANCE_SEGMENTATION_KUMAR_CLASS_DICT,
        "model_type": "nucleus_instance_segmentation",
        "target_node": "Nucleus Instance Segmentation",
        "description": (
            "Instance Segmentation and detection of Epithelial, Inflammatory, "
            "and Spindle-Shaped nuclei across different cancer types."
        ),
        "primary_site": "multi-site",
        "resolution_mpp": 0.5,
        "patch_shape": [512, 512],
        "stride_shape": [512, 512],
        "limitation": (
            "No class mapping was supplied in the model metadata; treat outputs "
            "as instance segmentation unless class labels are present in the AnnotationStore."
        ),
    },
}


def _nucleus_instance_segmentation_model_metadata(model_name: str) -> Dict[str, Any]:
    try:
        return NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG[model_name]
    except KeyError as exc:
        valid = ", ".join(sorted(NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG))
        raise ValueError(
            f"Unsupported nucleus instance segmentation model_name '{model_name}'. "
            f"Valid models: {valid}."
        ) from exc


def _nucleus_instance_segmentation_model_summary(model_name: str, model_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "model_type": model_meta.get("model_type", "nucleus_instance_segmentation"),
        "target_node": model_meta.get("target_node", "Nucleus Instance Segmentation"),
        "class_mapping": _model_class_mapping(model_meta),
        "description": model_meta.get("description", ""),
        "primary_site": model_meta.get("primary_site"),
        "input_resolution": (
            {"units": "mpp", "value": model_meta.get("resolution_mpp")}
            if model_meta.get("resolution_mpp") is not None
            else None
        ),
        "patch_shape": model_meta.get("patch_shape"),
        "stride_shape": model_meta.get("stride_shape"),
        "limitation": model_meta.get("limitation"),
    }


def _infer_nucleus_instance_segmentation_model_name_from_counts(counts: Counter) -> str:
    names = set(counts)
    if names & {"Spindle-Shaped", "Miscellaneous"}:
        return "hovernet_original-consep"
    if names & {"Non-Neoplastic Epithelial", "Neoplastic", "Inflammatory", "Connective", "Dead"}:
        return "hovernet_fast-pannuke"
    if names & {"Macrophage", "Neutrophil"}:
        return "hovernet_fast-monusac"
    return "nucleus_instance_segmentation"


MULTI_TASK_SEGMENTATION_OED_NUCLEAR_CLASS_DICT = {
    0: "Background",
    1: "Other",
    2: "Epithelial",
}

MULTI_TASK_SEGMENTATION_OED_REGION_CLASS_DICT = {
    0: "Background",
    1: "Other Tissue",
    2: "Basal Epithelium",
    3: "Epithelium",
    4: "Keratin",
}

MULTI_TASK_SEGMENTATION_MODEL_CATALOG = {
    "hovernetplus-oed": {
        "nuclear_class_dict": MULTI_TASK_SEGMENTATION_OED_NUCLEAR_CLASS_DICT,
        "output_region_class_dict": MULTI_TASK_SEGMENTATION_OED_REGION_CLASS_DICT,
        "model_type": "multi_task_segmentation",
        "target_node": "Multi-Task Segmentation",
        "description": (
            "Multi-task segmentation model with nuclear classes and output-region "
            "classes for oral epithelial dysplasia-style tissue analysis."
        ),
        "primary_site": "oral",
        "resolution_mpp": None,
        "patch_shape": None,
        "stride_shape": None,
        "limitation": "Research output only; not a clinical diagnosis.",
    },
}


def _class_mapping_from_dict(class_dict: Dict[int, str]) -> Dict[str, int]:
    return {label: int(type_id) for type_id, label in class_dict.items()}


def _multi_task_segmentation_model_metadata(model_name: str) -> Dict[str, Any]:
    try:
        return MULTI_TASK_SEGMENTATION_MODEL_CATALOG[model_name]
    except KeyError as exc:
        valid = ", ".join(sorted(MULTI_TASK_SEGMENTATION_MODEL_CATALOG))
        raise ValueError(
            f"Unsupported multi-task segmentation model_name '{model_name}'. "
            f"Valid models: {valid}."
        ) from exc


def _multi_task_segmentation_model_summary(model_name: str, model_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "model_type": model_meta.get("model_type", "multi_task_segmentation"),
        "target_node": model_meta.get("target_node", "Multi-Task Segmentation"),
        "nuclear_class_mapping": _class_mapping_from_dict(model_meta.get("nuclear_class_dict", {})),
        "output_region_class_mapping": _class_mapping_from_dict(model_meta.get("output_region_class_dict", {})),
        "description": model_meta.get("description", ""),
        "primary_site": model_meta.get("primary_site"),
        "input_resolution": (
            {"units": "mpp", "value": model_meta.get("resolution_mpp")}
            if model_meta.get("resolution_mpp") is not None
            else None
        ),
        "patch_shape": model_meta.get("patch_shape"),
        "stride_shape": model_meta.get("stride_shape"),
        "limitation": model_meta.get("limitation"),
    }


SEMANTIC_SEGMENTATION_BCSS_CLASS_DICT = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflamatory",
    3: "Necrosis",
    4: "Others",
}

SEMANTIC_SEGMENTATION_MODEL_CATALOG = {
    "fcn_resnet50_unet-bcss": {
        "class_dict": SEMANTIC_SEGMENTATION_BCSS_CLASS_DICT,
        "model_type": "semantic_segmentation",
        "target_node": "Semantic Segmentation",
        "description": "Segments tumor, stroma, necrosis, and inflamatory regions in breast cancer images.",
        "primary_site": "breast",
        "resolution_mpp": 0.5,
        "patch_shape": [512, 512],
        "stride_shape": [512, 512],
        "limitation": "Semantic region segmentation only; not a clinical diagnosis.",
    },
}


def _semantic_segmentation_model_metadata(model_name: str) -> Dict[str, Any]:
    try:
        return SEMANTIC_SEGMENTATION_MODEL_CATALOG[model_name]
    except KeyError as exc:
        valid = ", ".join(sorted(SEMANTIC_SEGMENTATION_MODEL_CATALOG))
        raise ValueError(
            f"Unsupported semantic segmentation model_name '{model_name}'. "
            f"Valid models: {valid}."
        ) from exc


def _semantic_segmentation_model_summary(model_name: str, model_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "model_type": model_meta.get("model_type", "semantic_segmentation"),
        "target_node": model_meta.get("target_node", "Semantic Segmentation"),
        "class_mapping": _model_class_mapping(model_meta),
        "description": model_meta.get("description", ""),
        "primary_site": model_meta.get("primary_site"),
        "input_resolution": (
            {"units": "mpp", "value": model_meta.get("resolution_mpp")}
            if model_meta.get("resolution_mpp") is not None
            else None
        ),
        "patch_shape": model_meta.get("patch_shape"),
        "stride_shape": model_meta.get("stride_shape"),
        "limitation": model_meta.get("limitation"),
    }


def _infer_kongnet_model_name_from_counts(counts: Counter) -> str:
    names = set(counts)
    if names & {"Plasma_Cell", "Histiocyte", "Melanophage", "Stroma_Cell", "Epithelial_Cell", "Endothelial_Cell", "Apoptotic_Cell"}:
        return "KongNet_PUMA_T2_3"
    if names & {"Tumour_Cell", "Other_Cell"}:
        return "KongNet_PUMA_T1_3"
    if names & {"Overall_Inflammatory", "Monocyte"}:
        return "KongNet_MONKEY_1"
    if names & {"Mitotic_Figure"}:
        return "KongNet_Det_MIDOG_1"
    if names & {"Neutrophil", "Lymphocyte", "Plasma", "Eosinophil"}:
        return "KongNet_CoNIC_1"
    if names & {"Neoplastic", "Inflammatory", "Dead"}:
        return "KongNet_PanNuke_1"
    return "KongNet"


def _ordered_kongnet_class_names(nuclei: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    present = {str(nucleus.get("type", "Unknown")) for nucleus in nuclei or []}
    ordered = [name for name in KONGNET_ALL_CLASS_NAMES if not present or name in present]
    extras = sorted(name for name in present if name not in ordered)
    return ordered + extras


def _kongnet_immune_fraction(class_counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(class_counts.get(name, 0) for name in KONGNET_IMMUNE_CELL_TYPES) / total


def _kongnet_pair_count(pair_counts: Dict[str, int], first_types: set, second_types: set) -> int:
    total = 0
    for pair_name, count in pair_counts.items():
        parts = str(pair_name).split("--")
        if len(parts) != 2:
            continue
        first, second = parts
        if (first in first_types and second in second_types) or (first in second_types and second in first_types):
            total += int(count)
    return total


def _flatten_annotationstore_paths(output: Any) -> List[str]:
    paths: List[str] = []

    if isinstance(output, dict):
        values = output.values()
    elif isinstance(output, (list, tuple, set)):
        values = output
    else:
        values = [output]

    for value in values:
        if isinstance(value, (list, tuple, set)):
            paths.extend(str(v) for v in value if str(v).lower().endswith(".db"))
        elif str(value).lower().endswith(".db"):
            paths.append(str(value))

    return paths


def _load_kongnet_nucleus_detector():
    try:
        from tiatoolbox.models.engine.nucleus_detector import NucleusDetector

        return NucleusDetector
    except ModuleNotFoundError as exc:
        if exc.name != "tiatoolbox.models.engine.nucleus_detector":
            raise

        import sys
        from importlib.metadata import PackageNotFoundError, version

        try:
            tiatoolbox_version = version("tiatoolbox")
        except PackageNotFoundError:
            tiatoolbox_version = "not installed"

        raise RuntimeError(
            "KongNet nucleus detection requires TIAToolbox's NucleusDetector engine, "
            "but this Python environment does not provide "
            "tiatoolbox.models.engine.nucleus_detector. Upgrade the same "
            "environment used by the MCP server with: "
            'python -m pip install --upgrade "tiatoolbox>=2.0.0". '
            f"Python executable: {sys.executable}. "
            f"Detected TIAToolbox version: {tiatoolbox_version}."
        ) from exc
    except ImportError as exc:
        import sys
        from importlib.metadata import PackageNotFoundError, version

        try:
            tiatoolbox_version = version("tiatoolbox")
        except PackageNotFoundError:
            tiatoolbox_version = "not installed"

        try:
            numpy_version = version("numpy")
        except PackageNotFoundError:
            numpy_version = "not installed"

        raise RuntimeError(
            "TIAToolbox's KongNet nucleus detector could not be imported. "
            "This usually means the active Python environment has an incompatible "
            "compiled dependency, commonly OpenCV built against NumPy 1.x while "
            "NumPy 2.x is installed. Try: "
            'python -m pip install "numpy<2" --force-reinstall, then reinstall '
            "opencv-python/tiatoolbox if needed. "
            f"Python executable: {sys.executable}. "
            f"TIAToolbox version: {tiatoolbox_version}. "
            f"NumPy version: {numpy_version}. "
            f"Original import error: {exc}"
        ) from exc


def _load_nucleus_instance_segmentor():
    try:
        from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

        return NucleusInstanceSegmentor
    except ModuleNotFoundError as exc:
        if exc.name != "tiatoolbox.models.engine.nucleus_instance_segmentor":
            raise

        from importlib.metadata import PackageNotFoundError, version

        try:
            tiatoolbox_version = version("tiatoolbox")
        except PackageNotFoundError:
            tiatoolbox_version = "not installed"

        raise RuntimeError(
            "Nucleus instance segmentation requires TIAToolbox's "
            "NucleusInstanceSegmentor engine, but this Python environment does "
            "not provide tiatoolbox.models.engine.nucleus_instance_segmentor. "
            'Upgrade the MCP environment with: python -m pip install --upgrade "tiatoolbox>=2.0.0". '
            f"Python executable: {sys.executable}. "
            f"Detected TIAToolbox version: {tiatoolbox_version}."
        ) from exc
    except ImportError as exc:
        from importlib.metadata import PackageNotFoundError, version

        try:
            tiatoolbox_version = version("tiatoolbox")
        except PackageNotFoundError:
            tiatoolbox_version = "not installed"

        try:
            numpy_version = version("numpy")
        except PackageNotFoundError:
            numpy_version = "not installed"

        raise RuntimeError(
            "TIAToolbox's nucleus instance segmentor could not be imported. "
            "This usually indicates an incompatible compiled dependency. "
            f"Python executable: {sys.executable}. "
            f"TIAToolbox version: {tiatoolbox_version}. "
            f"NumPy version: {numpy_version}. "
            f"Original import error: {exc}"
        ) from exc


def _load_multi_task_segmentor():
    try:
        from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

        return MultiTaskSegmentor
    except ModuleNotFoundError as exc:
        if exc.name != "tiatoolbox.models.engine.multi_task_segmentor":
            raise

        from importlib.metadata import PackageNotFoundError, version

        try:
            tiatoolbox_version = version("tiatoolbox")
        except PackageNotFoundError:
            tiatoolbox_version = "not installed"

        raise RuntimeError(
            "Multi-task segmentation requires TIAToolbox's MultiTaskSegmentor "
            "engine, but this Python environment does not provide "
            "tiatoolbox.models.engine.multi_task_segmentor. "
            'Upgrade the MCP environment with: python -m pip install --upgrade "tiatoolbox>=2.0.0". '
            f"Python executable: {sys.executable}. "
            f"Detected TIAToolbox version: {tiatoolbox_version}."
        ) from exc
    except ImportError as exc:
        from importlib.metadata import PackageNotFoundError, version

        try:
            tiatoolbox_version = version("tiatoolbox")
        except PackageNotFoundError:
            tiatoolbox_version = "not installed"

        raise RuntimeError(
            "TIAToolbox's MultiTaskSegmentor could not be imported. "
            f"Python executable: {sys.executable}. "
            f"TIAToolbox version: {tiatoolbox_version}. "
            f"Original import error: {exc}"
        ) from exc


def _load_semantic_segmentor():
    try:
        from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

        return SemanticSegmentor
    except ModuleNotFoundError as exc:
        if exc.name != "tiatoolbox.models.engine.semantic_segmentor":
            raise

        from importlib.metadata import PackageNotFoundError, version

        try:
            tiatoolbox_version = version("tiatoolbox")
        except PackageNotFoundError:
            tiatoolbox_version = "not installed"

        raise RuntimeError(
            "Semantic segmentation requires TIAToolbox's SemanticSegmentor "
            "engine, but this Python environment does not provide "
            "tiatoolbox.models.engine.semantic_segmentor. "
            'Upgrade the MCP environment with: python -m pip install --upgrade "tiatoolbox>=2.0.0". '
            f"Python executable: {sys.executable}. "
            f"Detected TIAToolbox version: {tiatoolbox_version}."
        ) from exc
    except ImportError as exc:
        from importlib.metadata import PackageNotFoundError, version

        try:
            tiatoolbox_version = version("tiatoolbox")
        except PackageNotFoundError:
            tiatoolbox_version = "not installed"

        raise RuntimeError(
            "TIAToolbox's SemanticSegmentor could not be imported. "
            f"Python executable: {sys.executable}. "
            f"TIAToolbox version: {tiatoolbox_version}. "
            f"Original import error: {exc}"
        ) from exc


def tool_predict_kongnet_nucleus_detection(
    wsi_path: str,
    output_json_path: Optional[str] = None,
    model_name: str = "KongNet_PanNuke_1",
    batch_size: int = 16,
    device: str = "auto",
    save_dir: Optional[str] = None,
    output_type: str = "annotationstore",
    patch_mode: bool = False,
    auto_get_mask: bool = False,
    num_workers: Optional[int] = None,
    overwrite: bool = True,
) -> str:
    """Run KongNet nucleus detection and save TIAViz overlays.

    Defaults to model_name="KongNet_PanNuke_1", but the same workflow also
    supports any registered KongNet model, including "KongNet_CoNIC_1",
    "KongNet_Det_MIDOG_1", "KongNet_MONKEY_1", "KongNet_PUMA_T1_3",
    and "KongNet_PUMA_T2_3", when explicitly provided.

    The output is a TIAToolbox AnnotationStore (.db) containing one vector
    annotation per detected nucleus, with the class labels defined by the
    selected KongNet model.
    """
    import multiprocessing
    from pathlib import Path

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    NucleusDetector = _load_kongnet_nucleus_detector()
    model_meta = _kongnet_model_metadata(model_name)
    class_dict = model_meta["class_dict"]

    if not isinstance(wsi_path, str) or not wsi_path.strip():
        raise ValueError("predict_kongnet_nucleus_detection requires a WSI file path.")

    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    if patch_mode is not False:
        raise ValueError("For TIAViz WSI nucleus detection, patch_mode must be False.")

    if output_type != "annotationstore":
        raise ValueError("For TIAViz compatibility, output_type must be 'annotationstore'.")

    selected_device = _choose_device(device)
    if isinstance(num_workers, int) and num_workers > 0:
        worker_count = int(num_workers)
    elif sys.platform.startswith("win"):
        # Windows uses the "spawn" multiprocessing strategy. In MCP/background
        # processes, spawning many TIAToolbox workers can fail while unpickling
        # the child process bootstrap state. Use one worker by default and let
        # callers opt into more workers explicitly when their environment is stable.
        worker_count = 1
    else:
        worker_count = multiprocessing.cpu_count()

    if not save_dir:
        if output_json_path:
            save_dir = os.path.join(
                os.path.dirname(output_json_path) or ".",
                "kongnet_nucleus_annotationstore",
            )
        else:
            save_dir = os.path.join(os.getcwd(), "kongnet_nucleus_annotationstore")

    os.makedirs(save_dir, exist_ok=True)

    detector = NucleusDetector(
        model=model_name,
        num_workers=worker_count,
        batch_size=int(batch_size),
        device=selected_device,
        verbose=True,
    )

    output = detector.run(
        images=[Path(wsi_path)],
        masks=None,
        patch_mode=False,
        save_dir=save_dir,
        output_type="annotationstore",
        class_dict=class_dict,
        auto_get_mask=bool(auto_get_mask),
        num_workers=worker_count,
        verbose=True,
        overwrite=bool(overwrite),
    )

    annotationstore_paths = _flatten_annotationstore_paths(output)
    slides_dir = os.path.dirname(wsi_path) or "."
    tiaviz_command = f'tiatoolbox visualize --slides "{slides_dir}" --overlays "{save_dir}"'

    result = {
        "mode": "wsi_nucleus_detection_annotationstore",
        "model_name": model_name,
        "model_description": model_meta["description"],
        "primary_site": model_meta["primary_site"],
        "input_resolution_mpp": model_meta["resolution_mpp"],
        "model_metadata": _kongnet_model_summary(model_name, model_meta),
        "wsi_path": wsi_path,
        "save_dir": save_dir,
        "annotationstore_paths": annotationstore_paths,
        "output_type": output_type,
        "device": selected_device,
        "num_workers": worker_count,
        "batch_size": int(batch_size),
        "patch_mode": False,
        "auto_get_mask": bool(auto_get_mask),
        "class_dict": class_dict,
        "tiaviz_command": tiaviz_command,
        "tiatoolbox_output": _json_safe(output),
        "clinical_warning": (
            "This is model-derived nucleus detection/classification output, "
            "not a clinical diagnosis."
        ),
    }

    if output_json_path:
        ensure_parent_dir(output_json_path)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    lines = [
        "WSI nucleus detection completed successfully.",
        f"Model: {model_name}",
        f"WSI: {wsi_path}",
        f"AnnotationStore output saved to: {save_dir}",
        f"Detected output DBs: {len(annotationstore_paths)}",
        f"Device: {selected_device}",
        f"Workers: {worker_count}",
        "",
        "Open in TIAViz with:",
        tiaviz_command,
    ]

    if annotationstore_paths:
        lines.append("\nAnnotationStore files:")
        lines.extend(annotationstore_paths)

    if output_json_path:
        lines.append(f"\nRun summary saved to: {output_json_path}")

    lines.append("\nImportant: this output is not a clinical diagnosis.")
    return "\n".join(lines)


def tool_predict_nucleus_instance_segmentation(
    wsi_path: str,
    output_json_path: Optional[str] = None,
    model_name: str = "hovernet_fast-monusac",
    batch_size: int = 8,
    device: str = "auto",
    save_dir: Optional[str] = None,
    output_type: str = "annotationstore",
    patch_mode: bool = False,
    auto_get_mask: bool = False,
    num_workers: Optional[int] = None,
    overwrite: bool = True,
) -> str:
    """Run nucleus instance segmentation and save TIAViz overlays.

    Defaults to model_name="hovernet_fast-monusac". The model is selected from
    NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG and run through TIAToolbox's
    NucleusInstanceSegmentor engine.
    """
    import multiprocessing
    from pathlib import Path

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    NucleusInstanceSegmentor = _load_nucleus_instance_segmentor()
    model_meta = _nucleus_instance_segmentation_model_metadata(model_name)
    class_dict = model_meta["class_dict"]

    if not isinstance(wsi_path, str) or not wsi_path.strip():
        raise ValueError("predict_nucleus_instance_segmentation requires a WSI file path.")

    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    if patch_mode is not False:
        raise ValueError("For TIAViz WSI nucleus instance segmentation, patch_mode must be False.")

    if output_type != "annotationstore":
        raise ValueError("For TIAViz compatibility, output_type must be 'annotationstore'.")

    selected_device = _choose_device(device)
    if isinstance(num_workers, int) and num_workers > 0:
        worker_count = int(num_workers)
    elif sys.platform.startswith("win"):
        worker_count = 1
    else:
        worker_count = multiprocessing.cpu_count()

    if not save_dir:
        if output_json_path:
            save_dir = os.path.join(
                os.path.dirname(output_json_path) or ".",
                "nucleus_instance_segmentation_annotationstore",
            )
        else:
            save_dir = os.path.join(os.getcwd(), "nucleus_instance_segmentation_annotationstore")

    os.makedirs(save_dir, exist_ok=True)

    segmentor = NucleusInstanceSegmentor(
        model=model_name,
        num_workers=worker_count,
        batch_size=int(batch_size),
        device=selected_device,
        verbose=True,
    )

    input_resolutions = (
        [{"units": "mpp", "resolution": float(model_meta["resolution_mpp"])}]
        if model_meta.get("resolution_mpp") is not None
        else None
    )

    run_kwargs = {
        "images": [Path(wsi_path)],
        "masks": None,
        "input_resolutions": input_resolutions,
        "patch_input_shape": model_meta.get("patch_shape"),
        "patch_mode": False,
        "save_dir": save_dir,
        "output_type": "annotationstore",
        "auto_get_mask": bool(auto_get_mask),
        "num_workers": worker_count,
        "verbose": True,
        "overwrite": bool(overwrite),
    }
    if class_dict:
        run_kwargs["class_dict"] = class_dict
    output = segmentor.run(**run_kwargs)

    annotationstore_paths = _flatten_annotationstore_paths(output)
    slides_dir = os.path.dirname(wsi_path) or "."
    tiaviz_command = f'tiatoolbox visualize --slides "{slides_dir}" --overlays "{save_dir}"'

    result = {
        "mode": "wsi_nucleus_instance_segmentation_annotationstore",
        "model_name": model_name,
        "model_description": model_meta["description"],
        "primary_site": model_meta["primary_site"],
        "input_resolution_mpp": model_meta["resolution_mpp"],
        "model_metadata": _nucleus_instance_segmentation_model_summary(model_name, model_meta),
        "wsi_path": wsi_path,
        "save_dir": save_dir,
        "annotationstore_paths": annotationstore_paths,
        "output_type": output_type,
        "device": selected_device,
        "num_workers": worker_count,
        "batch_size": int(batch_size),
        "patch_mode": False,
        "auto_get_mask": bool(auto_get_mask),
        "class_dict": class_dict,
        "tiaviz_command": tiaviz_command,
        "tiatoolbox_output": _json_safe(output),
        "clinical_warning": (
            "This is model-derived nucleus instance segmentation output, "
            "not a clinical diagnosis."
        ),
    }

    if output_json_path:
        ensure_parent_dir(output_json_path)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    lines = [
        "WSI nucleus instance segmentation completed successfully.",
        f"Model: {model_name}",
        f"WSI: {wsi_path}",
        f"AnnotationStore output saved to: {save_dir}",
        f"Detected output DBs: {len(annotationstore_paths)}",
        f"Device: {selected_device}",
        f"Workers: {worker_count}",
        "",
        "Open in TIAViz with:",
        tiaviz_command,
    ]

    if annotationstore_paths:
        lines.append("\nAnnotationStore files:")
        lines.extend(annotationstore_paths)

    if output_json_path:
        lines.append(f"\nRun summary saved to: {output_json_path}")

    lines.append("\nImportant: this output is not a clinical diagnosis.")
    return "\n".join(lines)


def tool_predict_multi_task_segmentation(
    wsi_path: str,
    output_json_path: Optional[str] = None,
    model_name: str = "hovernetplus-oed",
    batch_size: int = 8,
    device: str = "auto",
    save_dir: Optional[str] = None,
    output_type: str = "annotationstore",
    patch_mode: bool = False,
    auto_get_mask: bool = False,
    num_workers: Optional[int] = None,
    overwrite: bool = True,
) -> str:
    """Run multi-task segmentation and save TIAViz-compatible outputs where supported."""
    import multiprocessing
    from pathlib import Path

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    MultiTaskSegmentor = _load_multi_task_segmentor()
    model_meta = _multi_task_segmentation_model_metadata(model_name)
    nuclear_class_dict = model_meta.get("nuclear_class_dict", {})

    if not isinstance(wsi_path, str) or not wsi_path.strip():
        raise ValueError("predict_multi_task_segmentation requires a WSI file path.")

    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    if patch_mode is not False:
        raise ValueError("For TIAViz WSI multi-task segmentation, patch_mode must be False.")

    if output_type != "annotationstore":
        raise ValueError("For TIAViz compatibility, output_type must be 'annotationstore'.")

    selected_device = _choose_device(device)
    if isinstance(num_workers, int) and num_workers > 0:
        worker_count = int(num_workers)
    elif sys.platform.startswith("win"):
        worker_count = 1
    else:
        worker_count = multiprocessing.cpu_count()

    if not save_dir:
        if output_json_path:
            save_dir = os.path.join(
                os.path.dirname(output_json_path) or ".",
                "multi_task_segmentation_annotationstore",
            )
        else:
            save_dir = os.path.join(os.getcwd(), "multi_task_segmentation_annotationstore")

    os.makedirs(save_dir, exist_ok=True)

    segmentor = MultiTaskSegmentor(
        model=model_name,
        num_workers=worker_count,
        batch_size=int(batch_size),
        device=selected_device,
        verbose=True,
    )

    input_resolutions = (
        [{"units": "mpp", "resolution": float(model_meta["resolution_mpp"])}]
        if model_meta.get("resolution_mpp") is not None
        else None
    )

    run_kwargs = {
        "images": [Path(wsi_path)],
        "masks": None,
        "input_resolutions": input_resolutions,
        "patch_input_shape": model_meta.get("patch_shape"),
        "patch_mode": False,
        "save_dir": save_dir,
        "output_type": "annotationstore",
        "auto_get_mask": bool(auto_get_mask),
        "num_workers": worker_count,
        "verbose": True,
        "overwrite": bool(overwrite),
    }
    if nuclear_class_dict:
        run_kwargs["class_dict"] = nuclear_class_dict
    output = segmentor.run(**run_kwargs)

    annotationstore_paths = _flatten_annotationstore_paths(output)
    slides_dir = os.path.dirname(wsi_path) or "."
    tiaviz_command = f'tiatoolbox visualize --slides "{slides_dir}" --overlays "{save_dir}"'

    result = {
        "mode": "wsi_multi_task_segmentation_annotationstore",
        "model_name": model_name,
        "model_description": model_meta["description"],
        "primary_site": model_meta["primary_site"],
        "input_resolution_mpp": model_meta["resolution_mpp"],
        "model_metadata": _multi_task_segmentation_model_summary(model_name, model_meta),
        "wsi_path": wsi_path,
        "save_dir": save_dir,
        "annotationstore_paths": annotationstore_paths,
        "output_type": output_type,
        "device": selected_device,
        "num_workers": worker_count,
        "batch_size": int(batch_size),
        "patch_mode": False,
        "auto_get_mask": bool(auto_get_mask),
        "nuclear_class_dict": nuclear_class_dict,
        "output_region_class_dict": model_meta.get("output_region_class_dict", {}),
        "tiaviz_command": tiaviz_command,
        "tiatoolbox_output": _json_safe(output),
        "clinical_warning": (
            "This is model-derived multi-task segmentation output, not a clinical diagnosis."
        ),
    }

    if output_json_path:
        ensure_parent_dir(output_json_path)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    lines = [
        "WSI multi-task segmentation completed successfully.",
        f"Model: {model_name}",
        f"WSI: {wsi_path}",
        f"Output saved to: {save_dir}",
        f"Detected output DBs: {len(annotationstore_paths)}",
        f"Device: {selected_device}",
        f"Workers: {worker_count}",
        "",
        "Open in TIAViz with:",
        tiaviz_command,
    ]

    if annotationstore_paths:
        lines.append("\nAnnotationStore files:")
        lines.extend(annotationstore_paths)

    if output_json_path:
        lines.append(f"\nRun summary saved to: {output_json_path}")

    lines.append("\nImportant: this output is not a clinical diagnosis.")
    return "\n".join(lines)


def tool_predict_semantic_segmentation(
    wsi_path: str,
    output_json_path: Optional[str] = None,
    model_name: str = "fcn_resnet50_unet-bcss",
    batch_size: int = 8,
    device: str = "auto",
    save_dir: Optional[str] = None,
    output_type: str = "annotationstore",
    patch_mode: bool = False,
    auto_get_mask: bool = False,
    num_workers: Optional[int] = None,
    overwrite: bool = True,
) -> str:
    """Run semantic segmentation and save TIAViz-compatible outputs where supported."""
    import multiprocessing
    from pathlib import Path

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    SemanticSegmentor = _load_semantic_segmentor()
    model_meta = _semantic_segmentation_model_metadata(model_name)
    class_dict = model_meta["class_dict"]

    if not isinstance(wsi_path, str) or not wsi_path.strip():
        raise ValueError("predict_semantic_segmentation requires a WSI file path.")

    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    if patch_mode is not False:
        raise ValueError("For TIAViz WSI semantic segmentation, patch_mode must be False.")

    if output_type != "annotationstore":
        raise ValueError("For TIAViz compatibility, output_type must be 'annotationstore'.")

    selected_device = _choose_device(device)
    if isinstance(num_workers, int) and num_workers > 0:
        worker_count = int(num_workers)
    elif sys.platform.startswith("win"):
        worker_count = 1
    else:
        worker_count = multiprocessing.cpu_count()

    if not save_dir:
        if output_json_path:
            save_dir = os.path.join(
                os.path.dirname(output_json_path) or ".",
                "semantic_segmentation_annotationstore",
            )
        else:
            save_dir = os.path.join(os.getcwd(), "semantic_segmentation_annotationstore")

    os.makedirs(save_dir, exist_ok=True)

    segmentor = SemanticSegmentor(
        model=model_name,
        num_workers=worker_count,
        batch_size=int(batch_size),
        device=selected_device,
        verbose=True,
    )

    input_resolutions = (
        [{"units": "mpp", "resolution": float(model_meta["resolution_mpp"])}]
        if model_meta.get("resolution_mpp") is not None
        else None
    )

    output = segmentor.run(
        images=[Path(wsi_path)],
        masks=None,
        input_resolutions=input_resolutions,
        patch_input_shape=model_meta.get("patch_shape"),
        patch_mode=False,
        save_dir=save_dir,
        output_type="annotationstore",
        class_dict=class_dict,
        auto_get_mask=bool(auto_get_mask),
        num_workers=worker_count,
        verbose=True,
        overwrite=bool(overwrite),
    )

    annotationstore_paths = _flatten_annotationstore_paths(output)
    slides_dir = os.path.dirname(wsi_path) or "."
    tiaviz_command = f'tiatoolbox visualize --slides "{slides_dir}" --overlays "{save_dir}"'

    result = {
        "mode": "wsi_semantic_segmentation_annotationstore",
        "model_name": model_name,
        "model_description": model_meta["description"],
        "primary_site": model_meta["primary_site"],
        "input_resolution_mpp": model_meta["resolution_mpp"],
        "model_metadata": _semantic_segmentation_model_summary(model_name, model_meta),
        "wsi_path": wsi_path,
        "save_dir": save_dir,
        "annotationstore_paths": annotationstore_paths,
        "output_type": output_type,
        "device": selected_device,
        "num_workers": worker_count,
        "batch_size": int(batch_size),
        "patch_mode": False,
        "auto_get_mask": bool(auto_get_mask),
        "class_dict": class_dict,
        "tiaviz_command": tiaviz_command,
        "tiatoolbox_output": _json_safe(output),
        "clinical_warning": (
            "This is model-derived semantic segmentation output, not a clinical diagnosis."
        ),
    }

    if output_json_path:
        ensure_parent_dir(output_json_path)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    lines = [
        "WSI semantic segmentation completed successfully.",
        f"Model: {model_name}",
        f"WSI: {wsi_path}",
        f"Output saved to: {save_dir}",
        f"Detected output DBs: {len(annotationstore_paths)}",
        f"Device: {selected_device}",
        f"Workers: {worker_count}",
        "",
        "Open in TIAViz with:",
        tiaviz_command,
    ]

    if annotationstore_paths:
        lines.append("\nAnnotationStore files:")
        lines.extend(annotationstore_paths)

    if output_json_path:
        lines.append(f"\nRun summary saved to: {output_json_path}")

    lines.append("\nImportant: this output is not a clinical diagnosis.")
    return "\n".join(lines)


def _normalise_kongnet_cell_types(cell_types: Optional[List[str]]) -> Optional[List[str]]:
    if not cell_types:
        return None

    canonical = {name.casefold(): name for name in KONGNET_ALL_CLASS_NAMES}
    immune_aliases = {"immune", "inflammatory", "inflammation", "inflammatory cells", "immune cells"}
    normalised = []
    for value in cell_types:
        key = str(value).strip().casefold()
        if key in immune_aliases:
            for immune_type in KONGNET_ALL_CLASS_NAMES:
                if immune_type in KONGNET_IMMUNE_CELL_TYPES and immune_type not in normalised:
                    normalised.append(immune_type)
            continue
        if key not in canonical:
            valid = ", ".join(KONGNET_ALL_CLASS_NAMES)
            raise ValueError(f"Unknown KongNet cell type '{value}'. Valid types: {valid}.")
        if canonical[key] not in normalised:
            normalised.append(canonical[key])
    return normalised


def _resolve_spatial_scale(
    distance_units: str,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
) -> Dict[str, Any]:
    units = str(distance_units).strip().lower()
    if units not in {"microns", "pixels"}:
        raise ValueError("distance_units must be either 'microns' or 'pixels'.")

    if units == "pixels":
        return {"units": units, "x_scale": 1.0, "y_scale": 1.0, "source": "pixels"}

    if mpp is not None:
        value = float(mpp)
        if value <= 0:
            raise ValueError("mpp must be greater than 0.")
        return {"units": units, "x_scale": value, "y_scale": value, "source": "mpp"}

    if not wsi_path or not os.path.exists(wsi_path):
        raise ValueError(
            "Micrometre distances require either a valid wsi_path or a positive mpp value."
        )

    from tiatoolbox.wsicore.wsireader import WSIReader

    reader = WSIReader.open(wsi_path)
    slide_mpp = reader.info.mpp
    try:
        values = list(slide_mpp)
    except TypeError:
        values = [slide_mpp, slide_mpp]

    x_scale = float(values[0])
    y_scale = float(values[1] if len(values) > 1 else values[0])
    if x_scale <= 0 or y_scale <= 0:
        raise ValueError(f"WSI has invalid MPP metadata: {slide_mpp}.")

    return {
        "units": units,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "source": "wsi_metadata",
        "wsi_path": wsi_path,
    }


def _load_kongnet_nuclei(
    annotationstore_path: str,
    cell_types: Optional[List[str]] = None,
    min_probability: float = 0.0,
) -> List[Dict[str, Any]]:
    if not isinstance(annotationstore_path, str) or not os.path.exists(annotationstore_path):
        raise FileNotFoundError(f"KongNet AnnotationStore or nuclei CSV not found: {annotationstore_path}")
    if not 0.0 <= float(min_probability) <= 1.0:
        raise ValueError("min_probability must be between 0 and 1.")

    selected_types = _normalise_kongnet_cell_types(cell_types)
    selected = set(selected_types) if selected_types else None
    id_by_name = {}
    name_by_id = {}
    for metadata in KONGNET_MODEL_CATALOG.values():
        for type_id, name in metadata["class_dict"].items():
            id_by_name.setdefault(name, type_id)
            name_by_id.setdefault(type_id, name)

    nuclei = []
    if os.path.splitext(annotationstore_path)[1].lower() == ".csv":
        with open(annotationstore_path, "r", encoding="utf-8", newline="") as file:
            for row_index, row in enumerate(csv.DictReader(file), start=1):
                cell_type = str(row.get("type") or "Unknown")
                probability = float(row.get("probability") or 1.0)
                if probability < float(min_probability) or (selected is not None and cell_type not in selected):
                    continue
                nuclei.append({
                    "annotation_id": str(row.get("annotation_id") or row_index),
                    "x_px": float(row["x_px"]),
                    "y_px": float(row["y_px"]),
                    "type": cell_type,
                    "type_id": int(row["type_id"]) if row.get("type_id") not in (None, "") else id_by_name.get(cell_type),
                    "probability": probability,
                })
        return nuclei

    from tiatoolbox.annotation.storage import SQLiteStore
    store = SQLiteStore(annotationstore_path)
    try:
        for annotation_id, annotation in store.items():
            properties = dict(annotation.properties or {})
            raw_type = properties.get("type")
            if isinstance(raw_type, int):
                cell_type = name_by_id.get(raw_type, str(raw_type))
            else:
                cell_type = str(raw_type or "Unknown")

            probability = float(properties.get("probability", properties.get("prob", 1.0)))
            if probability < float(min_probability):
                continue
            if selected is not None and cell_type not in selected:
                continue

            centroid = annotation.geometry.centroid
            nuclei.append({
                "annotation_id": str(annotation_id),
                "x_px": float(centroid.x),
                "y_px": float(centroid.y),
                "type": cell_type,
                "type_id": id_by_name.get(cell_type),
                "probability": probability,
            })
    finally:
        store.close()

    return nuclei


def _load_nucleus_instance_segmentation_instances(
    annotationstore_path: str,
    min_probability: float = 0.0,
) -> List[Dict[str, Any]]:
    if not isinstance(annotationstore_path, str) or not os.path.exists(annotationstore_path):
        raise FileNotFoundError(f"Nucleus instance segmentation AnnotationStore or CSV not found: {annotationstore_path}")
    if not 0.0 <= float(min_probability) <= 1.0:
        raise ValueError("min_probability must be between 0 and 1.")

    id_by_name = {}
    name_by_id = {}
    for metadata in NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG.values():
        for type_id, name in metadata["class_dict"].items():
            id_by_name.setdefault(name, type_id)
            name_by_id.setdefault(type_id, name)

    instances = []
    if os.path.splitext(annotationstore_path)[1].lower() == ".csv":
        with open(annotationstore_path, "r", encoding="utf-8", newline="") as file:
            for row_index, row in enumerate(csv.DictReader(file), start=1):
                cell_type = str(row.get("type") or row.get("cell_type") or "Unknown")
                try:
                    probability = float(row.get("probability") or row.get("prob") or 1.0)
                except (TypeError, ValueError):
                    probability = 1.0
                if probability < float(min_probability):
                    continue
                instances.append({
                    "annotation_id": str(row.get("annotation_id") or row_index),
                    "type": cell_type,
                    "type_id": int(row["type_id"]) if row.get("type_id") not in (None, "") else id_by_name.get(cell_type),
                    "probability": probability,
                    "area": float(row["area"]) if row.get("area") not in (None, "") else None,
                })
        return instances

    from tiatoolbox.annotation.storage import SQLiteStore
    store = SQLiteStore(annotationstore_path)
    try:
        for annotation_id, annotation in store.items():
            properties = dict(annotation.properties or {})
            raw_type = properties.get("type", properties.get("label"))
            if isinstance(raw_type, int):
                cell_type = name_by_id.get(raw_type, str(raw_type))
            else:
                cell_type = str(raw_type or "Unknown")

            try:
                probability = float(properties.get("probability", properties.get("prob", 1.0)))
            except (TypeError, ValueError):
                probability = 1.0
            if probability < float(min_probability):
                continue

            instances.append({
                "annotation_id": str(annotation_id),
                "type": cell_type,
                "type_id": id_by_name.get(cell_type),
                "probability": probability,
                "area": float(annotation.geometry.area) if annotation.geometry is not None else None,
            })
    finally:
        store.close()

    return instances


def _nucleus_coordinates(nuclei: List[Dict[str, Any]], scale: Dict[str, Any]):
    import numpy as np

    return np.asarray([
        [nucleus["x_px"] * scale["x_scale"], nucleus["y_px"] * scale["y_scale"]]
        for nucleus in nuclei
    ], dtype=float)


def _write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def tool_export_kongnet_nuclei_to_csv(
    annotationstore_path: str,
    output_csv_path: str,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    min_probability: float = 0.0,
    cell_types: Optional[List[str]] = None,
) -> str:
    """Export KongNet detections with pixel and optional physical coordinates."""
    nuclei = _load_kongnet_nuclei(annotationstore_path, cell_types, min_probability)
    scale = None
    if mpp is not None or wsi_path:
        scale = _resolve_spatial_scale("microns", wsi_path=wsi_path, mpp=mpp)

    ensure_parent_dir(output_csv_path)
    fields = [
        "annotation_id", "x_px", "y_px", "x_um", "y_um",
        "type_id", "type", "probability",
    ]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for nucleus in nuclei:
            writer.writerow({
                **nucleus,
                "x_um": nucleus["x_px"] * scale["x_scale"] if scale else "",
                "y_um": nucleus["y_px"] * scale["y_scale"] if scale else "",
            })

    counts = Counter(nucleus["type"] for nucleus in nuclei)
    return "\n".join([
        "KongNet nuclei exported successfully.",
        f"Nuclei exported: {len(nuclei)}",
        f"Class counts: {dict(counts)}",
        f"Physical coordinates included: {scale is not None}",
        f"CSV: {output_csv_path}",
    ])


def tool_find_cells_within_radius(
    annotationstore_path: str,
    output_csv_path: str,
    radius: float = 50.0, #  default, used only if caller omits radius
    distance_units: str = "microns",
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    source_types: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
    min_probability: float = 0.0,
    output_json_path: Optional[str] = None,
) -> str:
    """Count target nuclei around every selected source nucleus."""
    import numpy as np
    from scipy.spatial import cKDTree

    if float(radius) <= 0:
        raise ValueError("radius must be greater than 0.")
    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    source = _load_kongnet_nuclei(annotationstore_path, source_types, min_probability)
    target = _load_kongnet_nuclei(annotationstore_path, target_types, min_probability)
    if not source or not target:
        raise RuntimeError("No source or target nuclei matched the requested filters.")

    source_coords = _nucleus_coordinates(source, scale)
    target_coords = _nucleus_coordinates(target, scale)
    tree = cKDTree(target_coords)
    rows = []
    total_links = 0
    for nucleus, point in zip(source, source_coords, strict=False):
        candidates = tree.query_ball_point(point, float(radius))
        neighbours = [index for index in candidates if target[index]["annotation_id"] != nucleus["annotation_id"]]
        distances = [float(np.linalg.norm(target_coords[index] - point)) for index in neighbours]
        nearest_index = neighbours[int(np.argmin(distances))] if distances else None
        rows.append({
            "source_id": nucleus["annotation_id"],
            "source_type": nucleus["type"],
            "x_px": nucleus["x_px"],
            "y_px": nucleus["y_px"],
            "neighbour_count": len(neighbours),
            "nearest_target_id": target[nearest_index]["annotation_id"] if nearest_index is not None else "",
            "nearest_target_type": target[nearest_index]["type"] if nearest_index is not None else "",
            "nearest_distance": min(distances) if distances else "",
        })
        total_links += len(neighbours)

    ensure_parent_dir(output_csv_path)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = [row["neighbour_count"] for row in rows]
    summary = {
        "annotationstore_path": annotationstore_path,
        "radius": float(radius),
        "distance_units": scale["units"],
        "scale": scale,
        "source_types": _normalise_kongnet_cell_types(source_types),
        "target_types": _normalise_kongnet_cell_types(target_types),
        "source_nuclei": len(source),
        "target_nuclei": len(target),
        "directed_neighbour_links": total_links,
        "sources_with_neighbours": sum(count > 0 for count in counts),
        "mean_neighbours_per_source": float(np.mean(counts)),
        "median_neighbours_per_source": float(np.median(counts)),
        "output_csv_path": output_csv_path,
    }
    _write_json(output_json_path, summary)
    return "\n".join([
        "Radius neighbourhood analysis completed.",
        f"Sources: {len(source)}; targets: {len(target)}",
        f"Radius: {radius} {scale['units']}",
        f"Mean neighbours per source: {summary['mean_neighbours_per_source']:.3f}",
        f"CSV: {output_csv_path}",
        f"JSON: {output_json_path}" if output_json_path else "JSON summary: not requested",
    ])


def tool_compute_cell_type_cooccurrence(
    annotationstore_path: str,
    output_json_path: str,
    radius: float = 50.0, # ← default, used only if caller omits radius
    distance_units: str = "microns",
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    cell_types: Optional[List[str]] = None,
    min_probability: float = 0.0,
    output_csv_path: Optional[str] = None,
) -> str:
    """Compute undirected cell-type pair counts within a spatial radius."""
    from scipy.spatial import cKDTree

    if float(radius) <= 0:
        raise ValueError("radius must be greater than 0.")
    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    nuclei = _load_kongnet_nuclei(annotationstore_path, cell_types, min_probability)
    if len(nuclei) < 2:
        raise RuntimeError("At least two matching nuclei are required for co-occurrence analysis.")

    class_names = _normalise_kongnet_cell_types(cell_types) or _ordered_kongnet_class_names(nuclei)
    matrix = {source: {target: 0 for target in class_names} for source in class_names}
    coords = _nucleus_coordinates(nuclei, scale)
    pairs = cKDTree(coords).query_pairs(float(radius))
    for first, second in pairs:
        first_type = nuclei[first]["type"]
        second_type = nuclei[second]["type"]
        if first_type not in matrix:
            matrix[first_type] = {target: 0 for target in class_names}
            for source in matrix:
                matrix[source].setdefault(first_type, 0)
            class_names.append(first_type)
        if second_type not in matrix:
            matrix[second_type] = {target: 0 for target in class_names}
            for source in matrix:
                matrix[source].setdefault(second_type, 0)
            class_names.append(second_type)
        matrix[first_type][second_type] += 1
        if first_type != second_type:
            matrix[second_type][first_type] += 1

    class_counts = Counter(nucleus["type"] for nucleus in nuclei)
    epithelial_count = class_counts.get("Epithelial", 0)
    neoplastic_count = class_counts.get("Neoplastic", 0)
    inflammatory_count = sum(class_counts.get(name, 0) for name in KONGNET_IMMUNE_CELL_TYPES)
    summary = {
        "annotationstore_path": annotationstore_path,
        "radius": float(radius),
        "distance_units": scale["units"],
        "scale": scale,
        "nucleus_count": len(nuclei),
        "undirected_pair_count": len(pairs),
        "class_counts": dict(class_counts),
        "cooccurrence_matrix": matrix,
        "inflammatory_to_epithelial_ratio": (
            inflammatory_count / epithelial_count if epithelial_count else None
        ),
        "inflammatory_to_neoplastic_ratio": (
            inflammatory_count / neoplastic_count if neoplastic_count else None
        ),
        "clinical_warning": "Spatial model output for research use; not a clinical diagnosis.",
    }
    _write_json(output_json_path, summary)

    if output_csv_path:
        ensure_parent_dir(output_csv_path)
        with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["source_type", *class_names])
            for source_type in class_names:
                writer.writerow([source_type, *[matrix[source_type][target] for target in class_names]])

    return "\n".join([
        "Cell-type co-occurrence analysis completed.",
        f"Nuclei: {len(nuclei)}",
        f"Pairs within {radius} {scale['units']}: {len(pairs)}",
        f"JSON: {output_json_path}",
        f"CSV: {output_csv_path}" if output_csv_path else "CSV matrix: not requested",
    ])


def tool_compute_nearest_neighbour_features(
    annotationstore_path: str,
    output_csv_path: str,
    distance_units: str = "microns",
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    source_types: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
    min_probability: float = 0.0,
    output_json_path: Optional[str] = None,
) -> str:
    """Find the nearest matching target nucleus for every source nucleus."""
    import numpy as np
    from scipy.spatial import cKDTree

    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    source = _load_kongnet_nuclei(annotationstore_path, source_types, min_probability)
    target = _load_kongnet_nuclei(annotationstore_path, target_types, min_probability)
    if not source or not target:
        raise RuntimeError("No source or target nuclei matched the requested filters.")

    source_coords = _nucleus_coordinates(source, scale)
    target_coords = _nucleus_coordinates(target, scale)
    tree = cKDTree(target_coords)
    query_k = min(8, len(target))
    rows = []
    grouped_distances: Dict[str, List[float]] = {}
    for nucleus, point in zip(source, source_coords, strict=False):
        distances, indices = tree.query(point, k=query_k)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        selected = next(
            (
                (float(distance), int(index))
                for distance, index in zip(distances, indices, strict=False)
                if target[int(index)]["annotation_id"] != nucleus["annotation_id"]
            ),
            None,
        )
        if selected is None:
            continue
        distance, index = selected
        neighbour = target[index]
        pair_name = f"{nucleus['type']}->{neighbour['type']}"
        grouped_distances.setdefault(pair_name, []).append(distance)
        rows.append({
            "source_id": nucleus["annotation_id"],
            "source_type": nucleus["type"],
            "source_x_px": nucleus["x_px"],
            "source_y_px": nucleus["y_px"],
            "neighbour_id": neighbour["annotation_id"],
            "neighbour_type": neighbour["type"],
            "distance": distance,
            "distance_units": scale["units"],
        })

    if not rows:
        raise RuntimeError("No non-self nearest neighbours were found.")
    ensure_parent_dir(output_csv_path)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_distances = [row["distance"] for row in rows]
    summary = {
        "annotationstore_path": annotationstore_path,
        "distance_units": scale["units"],
        "scale": scale,
        "source_types": _normalise_kongnet_cell_types(source_types),
        "target_types": _normalise_kongnet_cell_types(target_types),
        "source_nuclei_with_neighbour": len(rows),
        "mean_distance": float(np.mean(all_distances)),
        "median_distance": float(np.median(all_distances)),
        "pair_summaries": {
            pair: {
                "count": len(values),
                "mean_distance": float(np.mean(values)),
                "median_distance": float(np.median(values)),
            }
            for pair, values in sorted(grouped_distances.items())
        },
        "output_csv_path": output_csv_path,
    }
    _write_json(output_json_path, summary)
    return "\n".join([
        "Nearest-neighbour analysis completed.",
        f"Source nuclei with a neighbour: {len(rows)}",
        f"Mean distance: {summary['mean_distance']:.3f} {scale['units']}",
        f"Median distance: {summary['median_distance']:.3f} {scale['units']}",
        f"CSV: {output_csv_path}",
        f"JSON: {output_json_path}" if output_json_path else "JSON summary: not requested",
    ])


def _kongnet_region_label(class_counts: Counter) -> str:
    """Assign a readable, descriptive label without implying diagnosis."""
    total = sum(class_counts.values()) or 1
    class_names = _ordered_kongnet_class_names([{"type": name} for name in class_counts])
    fractions = {name: class_counts.get(name, 0) / total for name in class_names}
    immune_fraction = _kongnet_immune_fraction(class_counts, total)
    if fractions.get("Neoplastic", 0.0) >= 0.25 and immune_fraction >= 0.20:
        return "mixed tumour-immune region"
    if "Neoplastic" not in fractions and fractions.get("Epithelial", 0.0) >= 0.25 and immune_fraction >= 0.20:
        return "mixed epithelial-immune region"
    dominant = max(fractions, key=fractions.get)
    return {
        "Neoplastic": "high neoplastic-density region",
        "Inflammatory": "high immune-density region",
        "Neutrophil": "high immune-density region",
        "Lymphocyte": "high immune-density region",
        "Plasma": "high immune-density region",
        "Eosinophil": "high immune-density region",
        "Epithelial": "epithelial-rich region",
        "Connective": "connective/stromal-rich region",
        "Dead": "dead-cell-rich region",
    }.get(dominant, "mixed-cell region")


def tool_analyze_kongnet_regions(
    annotationstore_path: str,
    output_json_path: str,
    output_csv_path: Optional[str] = None,
    region_size: float = 500.0,
    neighbourhood_radius: float = 50.0,
    distance_units: str = "microns",
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    min_cells_per_region: int = 10,
    min_probability: float = 0.0,
) -> str:
    """Compute composition and spatial features independently in fixed local ROIs."""
    import numpy as np
    from scipy.spatial import cKDTree

    if region_size <= 0 or neighbourhood_radius <= 0:
        raise ValueError("region_size and neighbourhood_radius must be greater than 0.")
    if min_cells_per_region < 1:
        raise ValueError("min_cells_per_region must be at least 1.")
    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    nuclei = _load_kongnet_nuclei(annotationstore_path, None, min_probability)
    if not nuclei:
        raise RuntimeError("No nuclei matched the requested probability threshold.")
    coords = _nucleus_coordinates(nuclei, scale)
    origin = coords.min(axis=0)
    bins = np.floor((coords - origin) / float(region_size)).astype(int)
    grouped: Dict[tuple, List[int]] = {}
    for index, cell_bin in enumerate(bins):
        grouped.setdefault((int(cell_bin[0]), int(cell_bin[1])), []).append(index)

    class_names = _ordered_kongnet_class_names(nuclei)
    regions = []
    for (grid_x, grid_y), indices in sorted(grouped.items()):
        if len(indices) < min_cells_per_region:
            continue
        region_cells = [nuclei[index] for index in indices]
        region_coords = coords[indices]
        class_counts = Counter(cell["type"] for cell in region_cells)
        tree = cKDTree(region_coords)
        pairs = tree.query_pairs(float(neighbourhood_radius))
        pair_counts = Counter()
        for first, second in pairs:
            pair_name = "--".join(sorted((region_cells[first]["type"], region_cells[second]["type"])))
            pair_counts[pair_name] += 1
        if len(region_cells) > 1:
            nearest_distances = tree.query(region_coords, k=2)[0][:, 1]
            mean_nearest = float(np.mean(nearest_distances))
        else:
            mean_nearest = None
        x_min = float(origin[0] + grid_x * region_size)
        y_min = float(origin[1] + grid_y * region_size)
        area = float(region_size) ** 2
        region = {
            "region_id": f"R{len(regions) + 1}",
            "region_label": _kongnet_region_label(class_counts),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_min + float(region_size),
            "y_max": y_min + float(region_size),
            "distance_units": scale["units"],
            "cell_count": len(region_cells),
            "cell_density_per_square_unit": len(region_cells) / area,
            "class_counts": {name: class_counts.get(name, 0) for name in class_names},
            "class_percentages": {name: class_counts.get(name, 0) / len(region_cells) * 100 for name in class_names},
            "pairs_within_radius": len(pairs),
            "pair_counts": dict(pair_counts),
            "mean_nearest_neighbour_distance": mean_nearest,
        }
        regions.append(region)

    summary = {
        "annotationstore_path": annotationstore_path,
        "method": "fixed non-overlapping grid ROIs",
        "region_size": float(region_size),
        "neighbourhood_radius": float(neighbourhood_radius),
        "distance_units": scale["units"],
        "scale": scale,
        "min_cells_per_region": min_cells_per_region,
        "region_count": len(regions),
        "regions": regions,
        "interpretation_warning": "Region labels describe model-predicted composition and are not diagnoses.",
    }
    _write_json(output_json_path, summary)
    if output_csv_path:
        ensure_parent_dir(output_csv_path)
        fields = [
            "region_id", "region_label", "grid_x", "grid_y", "x_min", "y_min", "x_max", "y_max",
            "distance_units", "cell_count", "cell_density_per_square_unit", "pairs_within_radius",
            "mean_nearest_neighbour_distance",
            *[f"{name}_count" for name in class_names],
            *[f"{name}_percentage" for name in class_names],
        ]
        with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for region in regions:
                row = {key: region.get(key) for key in fields}
                for name in class_names:
                    row[f"{name}_count"] = region["class_counts"][name]
                    row[f"{name}_percentage"] = region["class_percentages"][name]
                writer.writerow(row)
    return "\n".join([
        "KongNet ROI analysis completed.",
        f"Regions retained: {len(regions)}",
        f"Region size: {region_size} {scale['units']}",
        f"Local pair radius: {neighbourhood_radius} {scale['units']}",
        f"JSON: {output_json_path}",
        f"CSV: {output_csv_path}" if output_csv_path else "CSV: not requested",
    ])


KONGNET_REGION_COLOURS = {
    "high neoplastic-density region": "#E53935",
    "high immune-density region": "#1E88E5",
    "epithelial-rich region": "#43A047",
    "mixed tumour-immune region": "#8E24AA",
    "mixed epithelial-immune region": "#6A1B9A",
    "connective/stromal-rich region": "#FB8C00",
    "dead-cell-rich region": "#6D4C41",
    "mixed-cell region": "#757575",
}


def tool_export_kongnet_regions_to_annotationstore(
    regions_json_path: str,
    output_db_path: str,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    overwrite: bool = True,
) -> str:
    """Create TIAViz-loadable ROI rectangles in baseline WSI coordinates."""
    from shapely.geometry import box
    from tiatoolbox.annotation.storage import Annotation, SQLiteStore

    if not os.path.exists(regions_json_path):
        raise FileNotFoundError(f"Regions JSON not found: {regions_json_path}")
    with open(regions_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    regions = data.get("regions", [])
    if not regions:
        raise ValueError("Regions JSON contains no ROI records.")

    units = str(data.get("distance_units", "pixels")).lower()
    stored_scale = data.get("scale") or {}
    if units == "pixels":
        x_scale = y_scale = 1.0
    elif mpp is not None or wsi_path:
        resolved = _resolve_spatial_scale("microns", wsi_path=wsi_path, mpp=mpp)
        x_scale, y_scale = resolved["x_scale"], resolved["y_scale"]
    elif stored_scale.get("x_scale") and stored_scale.get("y_scale"):
        x_scale = float(stored_scale["x_scale"])
        y_scale = float(stored_scale["y_scale"])
    else:
        raise ValueError("Micron-based ROI coordinates require wsi_path, mpp, or scale metadata in the regions JSON.")

    ensure_parent_dir(output_db_path)
    if os.path.exists(output_db_path):
        if not overwrite:
            raise FileExistsError(f"Output AnnotationStore already exists: {output_db_path}")
        os.remove(output_db_path)

    annotations = []
    keys = []
    labels = sorted({str(region.get("region_label", "mixed-cell region")) for region in regions})
    type_ids = {label: index for index, label in enumerate(labels)}
    for index, region in enumerate(regions):
        label = str(region.get("region_label", "mixed-cell region"))
        colour = KONGNET_REGION_COLOURS.get(label, KONGNET_REGION_COLOURS["mixed-cell region"])
        geometry = box(
            float(region["x_min"]) / x_scale,
            float(region["y_min"]) / y_scale,
            float(region["x_max"]) / x_scale,
            float(region["y_max"]) / y_scale,
        )
        properties = {
            "type": label,
            "label": label,
            "type_id": type_ids[label],
            "region_id": region.get("region_id", f"R{index + 1}"),
            "cell_count": int(region.get("cell_count", 0)),
            "cell_density": float(region.get("cell_density_per_square_unit", 0.0)),
            "pairs_within_radius": int(region.get("pairs_within_radius", 0)),
            "mean_nearest_neighbour_distance": region.get("mean_nearest_neighbour_distance"),
            "colour": colour,
            "color": colour,
            "line_color": colour,
            "fill_color": colour,
            "fill_opacity": 0.12,
            "is_roi": True,
            "coordinate_space": "baseline",
            "source": "KongNet fixed ROI analysis",
        }
        for cell_type, percentage in region.get("class_percentages", {}).items():
            properties[f"pct_{cell_type}"] = float(percentage)
        annotations.append(Annotation(geometry, properties=properties))
        keys.append(str(properties["region_id"]))

    store = SQLiteStore(output_db_path)
    try:
        store.append_many(annotations, keys=keys)
        store.commit()
        annotation_count = len(store)
    finally:
        store.close()

    overlays_dir = os.path.dirname(os.path.abspath(output_db_path))
    slides_dir = os.path.dirname(os.path.abspath(wsi_path)) if wsi_path else "<SLIDES_DIRECTORY>"
    tiaviz_command = f'tiatoolbox visualize --slides "{slides_dir}" --overlays "{overlays_dir}"'
    return "\n".join([
        "KongNet ROI boundary overlay created.",
        f"ROI rectangles: {annotation_count}",
        f"AnnotationStore: {output_db_path}",
        "Place the nucleus AnnotationStore in the same overlay directory to view both layers together.",
        "Open in TIAViz with:",
        tiaviz_command,
    ])


def tool_generate_kongnet_region_heatmaps(
    regions_json_path: str,
    output_dir: str,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    overwrite: bool = True,
) -> str:
    """Generate ROI heatmaps as TIAViz AnnotationStores."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize, to_hex
    from shapely.geometry import box
    from tiatoolbox.annotation.storage import Annotation, SQLiteStore

    if not os.path.exists(regions_json_path):
        raise FileNotFoundError(f"Regions JSON not found: {regions_json_path}")
    with open(regions_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    regions = data.get("regions", [])
    if not regions:
        raise ValueError("Regions JSON contains no ROI records.")
    os.makedirs(output_dir, exist_ok=True)

    units = str(data.get("distance_units", "pixels")).lower()
    stored_scale = data.get("scale") or {}
    if units == "pixels":
        x_scale = y_scale = 1.0
    elif mpp is not None or wsi_path:
        scale = _resolve_spatial_scale("microns", wsi_path=wsi_path, mpp=mpp)
        x_scale, y_scale = scale["x_scale"], scale["y_scale"]
    elif stored_scale.get("x_scale") and stored_scale.get("y_scale"):
        x_scale, y_scale = float(stored_scale["x_scale"]), float(stored_scale["y_scale"])
    else:
        raise ValueError("Micron-based heatmaps require wsi_path, mpp, or scale metadata in the regions JSON.")

    def density_value(region: Dict[str, Any]) -> float:
        raw = float(region.get("cell_density_per_square_unit", 0.0))
        return raw * 1_000_000.0 if units == "microns" else raw

    heatmaps = {
        "density": {
            "title": "KongNet Cell Density by ROI",
            "unit": "cells/mm²" if units == "microns" else "cells/pixel²",
            "cmap": "viridis",
            "value": density_value,
        },
        "inflammatory": {
            "title": "KongNet Inflammatory Cell Percentage by ROI",
            "unit": "% inflammatory nuclei",
            "cmap": "Blues",
            "value": lambda region: _kongnet_region_scores(region)["inflammatory_percentage"],
        },
        "tumour_immune_interaction": {
            "title": "KongNet Tumour-Immune Interaction by ROI",
            "unit": "pairs per 100 cells",
            "cmap": "magma",
            "value": lambda region: _kongnet_region_scores(region)["tumour_immune_pairs_per_100_cells"],
        },
    }
    outputs = {}
    for metric, config in heatmaps.items():
        db_path = os.path.join(output_dir, f"kongnet_{metric}_heatmap.db")
        legacy_png_path = os.path.join(output_dir, f"kongnet_{metric}_heatmap.png")
        if os.path.exists(db_path) and not overwrite:
            raise FileExistsError(f"Heatmap output already exists: {db_path}")
        if overwrite and os.path.exists(legacy_png_path):
            os.remove(legacy_png_path)
        values = [float(config["value"](region)) for region in regions]
        value_min, value_max = min(values), max(values)
        norm = Normalize(vmin=value_min, vmax=value_max if value_max > value_min else value_min + 1.0)
        cmap = colormaps[config["cmap"]]

        if os.path.exists(db_path):
            os.remove(db_path)
        annotations, keys = [], []
        for region, value in zip(regions, values, strict=False):
            percentile = norm(value)
            level = "high" if percentile >= 0.67 else "moderate" if percentile >= 0.33 else "low"
            colour = to_hex(cmap(percentile), keep_alpha=False)
            geometry = box(
                float(region["x_min"]) / x_scale,
                float(region["y_min"]) / y_scale,
                float(region["x_max"]) / x_scale,
                float(region["y_max"]) / y_scale,
            )
            label = f"{metric.replace('_', ' ')}: {level}"
            annotations.append(Annotation(geometry, properties={
                "type": label,
                "label": label,
                "region_id": region.get("region_id"),
                "heatmap_metric": metric,
                "heatmap_value": value,
                "heatmap_unit": config["unit"],
                "level": level,
                "color": colour,
                "colour": colour,
                "fill_color": colour,
                "line_color": colour,
                "fill_opacity": 0.35,
                "coordinate_space": "baseline",
                "source": "KongNet fixed ROI heatmap",
            }))
            keys.append(str(region.get("region_id")))
        store = SQLiteStore(db_path)
        try:
            store.append_many(annotations, keys=keys)
            store.commit()
        finally:
            store.close()
        outputs[metric] = {
            "annotationstore_path": db_path,
            "minimum": value_min,
            "maximum": value_max,
            "unit": config["unit"],
        }

    manifest_path = os.path.join(output_dir, "kongnet_heatmaps_manifest.json")
    _write_json(manifest_path, {
        "regions_json_path": regions_json_path,
        "region_count": len(regions),
        "heatmaps": outputs,
        "clinical_warning": "Heatmaps visualize model-derived ROI measurements, not diagnoses.",
    })
    return "\n".join([
        "KongNet ROI heatmaps generated.",
        f"Regions visualized: {len(regions)}",
        "Heatmaps: density, inflammatory percentage, tumour-immune interaction",
        f"TIAViz AnnotationStore outputs: {os.path.abspath(output_dir)}",
        f"Manifest: {manifest_path}",
    ])


def tool_characterize_kongnet_cell_neighbourhoods(
    annotationstore_path: str,
    output_csv_path: str,
    output_json_path: Optional[str] = None,
    radius: float = 50.0,
    distance_units: str = "microns",
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    min_probability: float = 0.0,
    community_count: int = 4,
) -> str:
    """Characterise every nucleus by neighbour type and cluster local profiles."""
    import numpy as np
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if radius <= 0:
        raise ValueError("radius must be greater than 0.")
    if community_count < 1:
        raise ValueError("community_count must be at least 1.")
    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    nuclei = _load_kongnet_nuclei(annotationstore_path, None, min_probability)
    if not nuclei:
        raise RuntimeError("No nuclei matched the requested probability threshold.")
    coords = _nucleus_coordinates(nuclei, scale)
    tree = cKDTree(coords)
    class_names = _ordered_kongnet_class_names(nuclei)
    count_matrix = np.zeros((len(nuclei), len(class_names)), dtype=int)
    rows = []
    for index, (nucleus, point) in enumerate(zip(nuclei, coords, strict=False)):
        neighbour_indices = [candidate for candidate in tree.query_ball_point(point, float(radius)) if candidate != index]
        neighbour_counts = Counter(nuclei[candidate]["type"] for candidate in neighbour_indices)
        count_matrix[index] = [neighbour_counts.get(name, 0) for name in class_names]
        rows.append({
            "annotation_id": nucleus["annotation_id"],
            "cell_type": nucleus["type"],
            "probability": nucleus["probability"],
            "x": float(point[0]),
            "y": float(point[1]),
            "distance_units": scale["units"],
            "radius": float(radius),
            "total_neighbours": len(neighbour_indices),
            **{f"{name}_neighbours": neighbour_counts.get(name, 0) for name in class_names},
        })

    totals = count_matrix.sum(axis=1, keepdims=True)
    proportions = np.divide(count_matrix, totals, out=np.zeros_like(count_matrix, dtype=float), where=totals > 0)
    features = np.column_stack((proportions, np.log1p(totals[:, 0])))
    cluster_count = min(int(community_count), len(nuclei))
    labels = KMeans(n_clusters=cluster_count, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(features))
    communities = []
    for cluster_id in range(cluster_count):
        indices = np.flatnonzero(labels == cluster_id)
        own_counts = Counter(nuclei[index]["type"] for index in indices)
        neighbour_totals = count_matrix[indices].sum(axis=0)
        neighbour_composition = Counter({name: int(neighbour_totals[pos]) for pos, name in enumerate(class_names)})
        combined = Counter(own_counts)
        combined.update(neighbour_composition)
        label = _kongnet_region_label(combined)
        community = {
            "community_id": f"C{cluster_id + 1}",
            "community_label": label,
            "cell_count": len(indices),
            "centroid_x": float(np.mean(coords[indices, 0])),
            "centroid_y": float(np.mean(coords[indices, 1])),
            "own_cell_type_counts": dict(own_counts),
            "mean_neighbour_counts": {
                name: float(np.mean(count_matrix[indices, pos])) for pos, name in enumerate(class_names)
            },
            "mean_total_neighbours": float(np.mean(totals[indices, 0])),
        }
        communities.append(community)
        for index in indices:
            rows[int(index)]["community_id"] = community["community_id"]
            rows[int(index)]["community_label"] = label

    ensure_parent_dir(output_csv_path)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "annotationstore_path": annotationstore_path,
        "radius": float(radius),
        "distance_units": scale["units"],
        "cell_count": len(nuclei),
        "community_count": cluster_count,
        "clustering_features": [*[f"{name}_neighbour_proportion" for name in class_names], "log_total_neighbours"],
        "communities": communities,
        "output_csv_path": output_csv_path,
        "interpretation_warning": "Communities are unsupervised model-derived patterns, not validated biological classes.",
    }
    _write_json(output_json_path, summary)
    return "\n".join([
        "Per-cell KongNet neighbourhood characterisation completed.",
        f"Cells characterised: {len(nuclei)}",
        f"Radius: {radius} {scale['units']}",
        f"Spatial communities: {cluster_count}",
        f"CSV: {output_csv_path}",
        f"JSON summary: {output_json_path}" if output_json_path else "JSON summary: not requested",
    ])

def _estimate_patch_size_from_predictions(preds: List[Dict[str, Any]], fallback: int = 224) -> int:
    xs = sorted({int(p["x"]) for p in preds if int(p.get("x", -1)) >= 0})
    ys = sorted({int(p["y"]) for p in preds if int(p.get("y", -1)) >= 0})

    diffs = []

    for arr in [xs, ys]:
        for a, b in zip(arr, arr[1:]):
            d = b - a
            if d > 0:
                diffs.append(d)

    if not diffs:
        return fallback

    return int(max(1, round(float(sorted(diffs)[len(diffs) // 2]))))


def _count_clusters(preds: List[Dict[str, Any]], cluster_distance: Optional[float] = None) -> int:
    valid = [
        p for p in preds
        if int(p.get("x", -1)) >= 0 and int(p.get("y", -1)) >= 0
    ]

    if not valid:
        return 0

    if cluster_distance is None:
        estimated = _estimate_patch_size_from_predictions(valid, fallback=224)
        cluster_distance = float(estimated * 1.5)

    n = len(valid)
    visited = [False] * n
    coords = [(int(p["x"]), int(p["y"])) for p in valid]

    clusters = 0

    for i in range(n):
        if visited[i]:
            continue

        clusters += 1
        visited[i] = True
        q = deque([i])

        while q:
            cur = q.popleft()
            x1, y1 = coords[cur]

            for j in range(n):
                if visited[j]:
                    continue

                x2, y2 = coords[j]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if dist <= cluster_distance:
                    visited[j] = True
                    q.append(j)

    return clusters


def tool_aggregate_kather_metrics(
    predictions_json_path: str,
    output_metrics_path: Optional[str] = None,
    abnormality_threshold: float = 0.5,
    cluster_distance: Optional[float] = None,
) -> str:
    if not isinstance(predictions_json_path, str) or not os.path.exists(predictions_json_path):
        raise FileNotFoundError("aggregate_kather_metrics requires a valid predictions_json_path.")

    loaded = _load_kather_predictions(predictions_json_path)
    preds = loaded["predictions"]
    if not preds:
        raise RuntimeError("No predictions found in predictions JSON.")

    total = len(preds)
    tissue_preds = [p for p in preds if p.get("predicted_class") != "BACK"]
    tissue_total = len(tissue_preds)

    class_counts = Counter(p["predicted_class"] for p in preds)
    class_percentages = {
        cls: float(100.0 * count / total)
        for cls, count in class_counts.items()
    }

    abnormal_preds = [
        p for p in preds
        if float(p.get("abnormality_score", 0.0)) >= float(abnormality_threshold)
    ]

    tumour_preds = [p for p in preds if p.get("predicted_class") == "TUM"]
    high_tumour_likelihood_preds = [
        p for p in preds
        if float(p.get("tumour_likelihood_score", p.get("tumour_epithelium_probability", 0.0))) >= float(abnormality_threshold)
    ]

    class_distribution = [
        class_counts.get(cls, 0) / total
        for cls in KATHER_ANALYSIS_CLASSES
    ]

    class_entropy = _prediction_entropy(class_distribution)
    max_entropy = math.log2(max(2, len(KATHER_ANALYSIS_CLASSES)))
    normalised_class_entropy = float(class_entropy / max_entropy)

    colour_vectors = [
        KATHER_CLASS_RGB.get(str(p.get("predicted_class")), (0, 0, 0))
        for p in preds
    ]
    channel_variances = []
    for channel in range(3):
        values = [rgb[channel] / 255.0 for rgb in colour_vectors]
        mean = sum(values) / total
        channel_variances.append(sum((v - mean) ** 2 for v in values) / total)
    colour_variance = float(sum(channel_variances) / 3.0)

    abnormality_scores = [float(p.get("abnormality_score", 0.0)) for p in preds]
    mean_abnormality = float(sum(abnormality_scores) / total)
    abnormality_variance = float(
        sum((score - mean_abnormality) ** 2 for score in abnormality_scores) / total
    )
    abnormality_std = math.sqrt(abnormality_variance)
    heterogeneity_index = float(
        (0.45 * normalised_class_entropy)
        + (0.35 * min(1.0, colour_variance * 4.0))
        + (0.20 * min(1.0, abnormality_std * 2.0))
    )

    cluster_count = _count_clusters(
        preds=abnormal_preds,
        cluster_distance=cluster_distance,
    )

    metrics = {
        "source_predictions": predictions_json_path,
        "source_format": loaded["source_format"],
        "model_name": loaded.get("model_name", "resnet18-kather100k"),
        "total_predicted_patches": total,
        "tissue_patch_count_excluding_background": tissue_total,
        "class_counts": dict(class_counts),
        "class_percentages": class_percentages,
        "tumour_epithelium_patch_count": len(tumour_preds),
        "tumour_epithelium_percentage": float(100.0 * len(tumour_preds) / total),
        "tumour_epithelium_percentage_excluding_background": (
            float(100.0 * len([p for p in tissue_preds if p.get("predicted_class") == "TUM"]) / tissue_total)
            if tissue_total else 0.0
        ),
        "high_tumour_likelihood_patch_count": len(high_tumour_likelihood_preds),
        "high_tumour_likelihood_percentage": float(100.0 * len(high_tumour_likelihood_preds) / total),
        "high_abnormality_patch_count": len(abnormal_preds),
        "high_abnormality_percentage": float(100.0 * len(abnormal_preds) / total),
        "high_abnormality_percentage_excluding_background": (
            float(100.0 * len([p for p in abnormal_preds if p.get("predicted_class") != "BACK"]) / tissue_total)
            if tissue_total else 0.0
        ),
        "abnormality_threshold": float(abnormality_threshold),
        "mean_tumour_epithelium_probability": float(
            sum(float(p.get("tumour_epithelium_probability", 0.0)) for p in preds) / total
        ),
        "mean_abnormality_score": mean_abnormality,
        "max_abnormality_score": float(
            max(abnormality_scores)
        ),
        "colour_variance": colour_variance,
        "class_entropy": class_entropy,
        "shannon_entropy": class_entropy,
        "normalised_shannon_entropy": normalised_class_entropy,
        "abnormality_score_std": abnormality_std,
        "heterogeneity_index": heterogeneity_index,
        "cluster_count": int(cluster_count),
        "cluster_distance": cluster_distance,
        "clinical_warning": (
            "These metrics summarise tissue-class model confidence, not clinical diagnosis."
        ),
    }

    if output_metrics_path:
        ensure_parent_dir(output_metrics_path)
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    lines = [
        "Kather prediction metrics aggregated successfully.",
        f"Model: {metrics['model_name']}",
        f"Total predicted patches: {total}",
        f"Tissue patches excluding BACK: {tissue_total}",
        f"TUM patches: {metrics['tumour_epithelium_patch_count']}",
        f"TUM percentage: {metrics['tumour_epithelium_percentage']:.2f}%",
        f"High tumour-likelihood patches: {metrics['high_tumour_likelihood_patch_count']}",
        f"High tumour-likelihood percentage: {metrics['high_tumour_likelihood_percentage']:.2f}%",
        f"High abnormality patches: {metrics['high_abnormality_patch_count']}",
        f"High abnormality percentage: {metrics['high_abnormality_percentage']:.2f}%",
        f"Mean TUM probability: {metrics['mean_tumour_epithelium_probability']:.6f}",
        f"Mean abnormality score: {metrics['mean_abnormality_score']:.6f}",
        f"Max abnormality score: {metrics['max_abnormality_score']:.6f}",
        f"Class entropy: {class_entropy:.6f}",
        f"Colour variance: {colour_variance:.6f}",
        f"Heterogeneity index: {heterogeneity_index:.6f}",
        f"Cluster count: {cluster_count}",
        "",
        "Class percentages:",
    ]

    for cls, pct in sorted(class_percentages.items(), key=lambda x: x[1], reverse=True):
        desc = KATHER_CLASS_DESCRIPTIONS.get(cls, cls)
        lines.append(f"  {cls} ({desc}): {pct:.2f}%")

    if output_metrics_path:
        lines.append(f"\nMetrics saved to: {output_metrics_path}")

    lines.append("")
    lines.append("Important: these are tissue-class/model-confidence metrics, not clinical diagnosis.")

    return "\n".join(lines)


def write_kather_patch_table_csv(
    predictions_json_path: str,
    output_csv_path: str,
    class_dict: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    loaded = _load_kather_predictions(predictions_json_path, class_dict=class_dict)
    preds = loaded["predictions"]
    ensure_parent_dir(output_csv_path)

    preferred_fields = [
        "filename",
        "patch_path",
        "level",
        "x",
        "y",
        "x1",
        "y1",
        "predicted_class",
        "predicted_class_description",
        "class_index",
        "confidence",
        "tumour_epithelium_probability",
        "stroma_probability",
        "lymphocyte_probability",
        "tumour_likelihood_score",
        "abnormality_score",
    ]
    extra_fields = sorted({
        key
        for row in preds
        for key in row.keys()
        if key not in preferred_fields
    })
    fieldnames = preferred_fields + extra_fields

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in preds:
            writer.writerow(row)

    return {
        "path": output_csv_path,
        "patch_count": len(preds),
        "source_format": loaded["source_format"],
    }


def _write_high_abnormality_csv(
    predictions_json_path: str,
    output_csv_path: str,
    abnormality_threshold: float = 0.5,
    top_k: int = 100,
    class_dict: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    loaded = _load_kather_predictions(predictions_json_path, class_dict=class_dict)
    preds = loaded["predictions"]
    high = [
        p for p in preds
        if float(p.get("abnormality_score", 0.0)) >= float(abnormality_threshold)
    ]
    ranked = sorted(high, key=lambda p: float(p.get("abnormality_score", 0.0)), reverse=True)
    if top_k and top_k > 0:
        ranked = ranked[:int(top_k)]

    ensure_parent_dir(output_csv_path)
    fieldnames = [
        "rank",
        "filename",
        "patch_path",
        "x",
        "y",
        "x1",
        "y1",
        "predicted_class",
        "predicted_class_description",
        "confidence",
        "abnormality_score",
        "tumour_likelihood_score",
        "tumour_epithelium_probability",
        "stroma_probability",
        "lymphocyte_probability",
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)

    return {
        "path": output_csv_path,
        "threshold": float(abnormality_threshold),
        "high_abnormality_patch_count": len(high),
        "written_rows": len(ranked),
        "high_abnormality_percentage": float(100.0 * len(high) / len(preds)) if preds else 0.0,
    }


def run_kather_postprocessing_pipeline(
    predictions_json_path: str,
    output_dir: str,
    class_dict: Optional[Dict[Any, Any]] = None,
    abnormality_threshold: float = 0.5,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    patch_table_path = os.path.join(output_dir, "kather_patch_table.csv")
    metrics_path = os.path.join(output_dir, "kather_postprocessing_metrics.json")
    summary_path = os.path.join(output_dir, "kather_postprocessing_summary.txt")
    high_abnormality_path = os.path.join(output_dir, "kather_high_abnormality_patches.csv")
    histogram_path = os.path.join(output_dir, "kather_confidence_histogram.png")

    patch_table = write_kather_patch_table_csv(
        predictions_json_path,
        patch_table_path,
        class_dict=class_dict,
    )

    tool_aggregate_kather_metrics(
        predictions_json_path,
        output_metrics_path=metrics_path,
        abnormality_threshold=abnormality_threshold,
    )

    high_abnormality = _write_high_abnormality_csv(
        predictions_json_path,
        high_abnormality_path,
        abnormality_threshold=abnormality_threshold,
        top_k=100,
        class_dict=class_dict,
    )

    histogram_status = None
    try:
        histogram_status = tool_generate_confidence_histogram(
            predictions_json_path,
            histogram_path,
            bins=20,
        )
    except Exception as exc:
        histogram_status = f"Histogram generation skipped: {exc}"
        histogram_path = None

    summary = tool_summarize_kather_results(
        predictions_json_path=predictions_json_path,
        metrics_json_path=metrics_path,
        output_summary_path=summary_path,
    )

    return {
        "patch_table_csv": patch_table_path,
        "patch_count": patch_table["patch_count"],
        "metrics_json": metrics_path,
        "summary_txt": summary_path,
        "high_abnormality_csv": high_abnormality_path,
        "confidence_histogram_png": histogram_path,
        "abnormality_threshold": float(abnormality_threshold),
        "high_abnormality_patch_count": high_abnormality["high_abnormality_patch_count"],
        "high_abnormality_percentage": high_abnormality["high_abnormality_percentage"],
        "histogram_status": histogram_status,
        "summary_preview": "\n".join(summary.splitlines()[:12]),
    }

def _get_prediction_level_dimensions(wsi, preds: List[Dict[str, Any]]):
    pred_levels = [
        int(p.get("level", 0))
        for p in preds
        if int(p.get("level", 0)) >= 0
    ]

    pred_level = pred_levels[0] if pred_levels else 0

    try:
        level_w, level_h = wsi.info.level_dimensions[pred_level]
    except Exception:
        level_w, level_h = wsi.info.slide_dimensions
        pred_level = 0

    return pred_level, level_w, level_h


def _add_legend_rgb(img_rgb, class_colours):
    import cv2
    import numpy as np

    legend_items = [
        ("ADI", "adipose"),
        ("BACK", "background"),
        ("DEB", "debris"),
        ("LYM", "lymphocytes"),
        ("MUC", "mucus"),
        ("MUS", "muscle"),
        ("NORM", "normal"),
        ("STR", "stroma"),
        ("TUM", "tumour epithelium"),
    ]

    h, w = img_rgb.shape[:2]
    panel_w = 330
    row_h = 28
    panel_h = row_h * len(legend_items) + 20

    x0 = max(10, w - panel_w - 10)
    y0 = 10

    overlay = img_rgb.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + panel_w, y0 + panel_h),
        (255, 255, 255),
        thickness=-1,
    )
    img_rgb = (0.65 * img_rgb + 0.35 * overlay).astype(np.uint8)

    cv2.rectangle(
        img_rgb,
        (x0, y0),
        (x0 + panel_w, y0 + panel_h),
        (0, 0, 0),
        thickness=2,
    )

    cv2.putText(
        img_rgb,
        "Kather100K classes",
        (x0 + 10, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    y = y0 + 45
    for cls, label in legend_items:
        colour = tuple(int(v) for v in class_colours.get(cls, np.array([255, 255, 255])))
        cv2.rectangle(img_rgb, (x0 + 10, y - 14), (x0 + 30, y + 6), colour, thickness=-1)
        cv2.rectangle(img_rgb, (x0 + 10, y - 14), (x0 + 30, y + 6), (0, 0, 0), thickness=1)
        cv2.putText(
            img_rgb,
            f"{cls}: {label}",
            (x0 + 40, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += row_h

    return img_rgb


def tool_generate_kather_overlay(
    wsi_path: str,
    predictions_json_path: str,
    thumbnail_path: str,
    output_overlay_path: str,
    output_heterogeneity_path: Optional[str] = None,
    alpha: float = 0.75,
    patch_size: int = 224,
    min_display_size: int = 8,
    draw_legend: bool = True,
) -> str:
    if not isinstance(wsi_path, str) or not os.path.exists(wsi_path):
        raise FileNotFoundError("generate_kather_overlay requires a valid wsi_path.")

    if not isinstance(predictions_json_path, str) or not os.path.exists(predictions_json_path):
        raise FileNotFoundError("generate_kather_overlay requires a valid predictions_json_path.")

    if not isinstance(thumbnail_path, str) or not os.path.exists(thumbnail_path):
        raise FileNotFoundError("generate_kather_overlay requires a valid thumbnail_path.")

    if not isinstance(output_overlay_path, str) or not output_overlay_path.strip():
        raise ValueError("generate_kather_overlay requires output_overlay_path.")

    ensure_parent_dir(output_overlay_path)

    if output_heterogeneity_path:
        ensure_parent_dir(output_heterogeneity_path)

    import cv2
    import numpy as np
    from tiatoolbox.wsicore.wsireader import WSIReader

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    if not preds:
        raise RuntimeError("No predictions found in predictions JSON.")

    thumb_bgr = cv2.imread(thumbnail_path)
    if thumb_bgr is None:
        raise RuntimeError(f"Could not read thumbnail: {thumbnail_path}")

    thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)

    class_layer = thumb_rgb.copy()
    heterogeneity_layer = thumb_rgb.copy()

    wsi = WSIReader.open(wsi_path)
    pred_level, level_w, level_h = _get_prediction_level_dimensions(wsi, preds)

    thumb_h, thumb_w = thumb_rgb.shape[:2]

    scale_x = thumb_w / float(level_w)
    scale_y = thumb_h / float(level_h)

    class_colours = {
        "ADI": np.array([255, 255, 0], dtype=np.uint8),
        "BACK": np.array([180, 180, 180], dtype=np.uint8),
        "DEB": np.array([120, 80, 40], dtype=np.uint8),
        "LYM": np.array([0, 80, 255], dtype=np.uint8),
        "MUC": np.array([0, 255, 255], dtype=np.uint8),
        "MUS": np.array([255, 128, 0], dtype=np.uint8),
        "NORM": np.array([0, 255, 0], dtype=np.uint8),
        "STR": np.array([255, 0, 255], dtype=np.uint8),
        "TUM": np.array([255, 0, 0], dtype=np.uint8),
    }

    visible_patch_count = 0

    for p in preds:
        x = int(p.get("x", -1))
        y = int(p.get("y", -1))

        if x < 0 or y < 0:
            continue

        predicted_class = p.get("predicted_class", "UNKNOWN")
        abnormality_score = float(p.get("abnormality_score", 0.0))
        confidence = float(p.get("confidence", 0.0))

        tx1 = int(x * scale_x)
        ty1 = int(y * scale_y)

        raw_w = int(patch_size * scale_x)
        raw_h = int(patch_size * scale_y)

        display_w = max(min_display_size, raw_w)
        display_h = max(min_display_size, raw_h)

        tx2 = tx1 + display_w
        ty2 = ty1 + display_h

        tx1 = max(0, min(tx1, thumb_w - 1))
        ty1 = max(0, min(ty1, thumb_h - 1))
        tx2 = max(tx1 + 1, min(tx2, thumb_w))
        ty2 = max(ty1 + 1, min(ty2, thumb_h))

        if tx2 <= tx1 or ty2 <= ty1:
            continue

        visible_patch_count += 1

        colour = class_colours.get(predicted_class, np.array([255, 255, 255], dtype=np.uint8))

        cv2.rectangle(
            class_layer,
            (tx1, ty1),
            (tx2, ty2),
            tuple(int(v) for v in colour),
            thickness=-1,
        )

        if predicted_class in ["TUM", "STR"]:
            cv2.rectangle(
                class_layer,
                (tx1, ty1),
                (tx2, ty2),
                (0, 0, 0),
                thickness=2,
            )

        uncertainty = 1.0 - confidence
        red = int(255 * max(0.0, min(1.0, abnormality_score)))
        green = int(255 * max(0.0, min(1.0, uncertainty)))
        blue = 0

        heter_colour = np.array([red, green, blue], dtype=np.uint8)

        cv2.rectangle(
            heterogeneity_layer,
            (tx1, ty1),
            (tx2, ty2),
            tuple(int(v) for v in heter_colour),
            thickness=-1,
        )

        if abnormality_score >= 0.5:
            cv2.rectangle(
                heterogeneity_layer,
                (tx1, ty1),
                (tx2, ty2),
                (0, 0, 0),
                thickness=2,
            )

    class_overlay_rgb = cv2.addWeighted(
        thumb_rgb,
        1.0 - float(alpha),
        class_layer,
        float(alpha),
        0,
    )

    heterogeneity_rgb = cv2.addWeighted(
        thumb_rgb,
        1.0 - float(alpha),
        heterogeneity_layer,
        float(alpha),
        0,
    )

    if draw_legend:
        class_overlay_rgb = _add_legend_rgb(class_overlay_rgb, class_colours)

        cv2.putText(
            heterogeneity_rgb,
            "Heterogeneity: red = high abnormality, green/yellow = uncertainty",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    class_overlay_bgr = cv2.cvtColor(class_overlay_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(output_overlay_path, class_overlay_bgr)

    if not ok:
        raise IOError(f"Failed to save Kather overlay to: {output_overlay_path}")

    heter_msg = "Heterogeneity overlay: not requested."

    if output_heterogeneity_path:
        heter_bgr = cv2.cvtColor(heterogeneity_rgb, cv2.COLOR_RGB2BGR)
        ok2 = cv2.imwrite(output_heterogeneity_path, heter_bgr)

        if not ok2:
            raise IOError(f"Failed to save heterogeneity overlay to: {output_heterogeneity_path}")

        heter_msg = f"Heterogeneity overlay saved to: {output_heterogeneity_path}"

    lines = [
        "Kather tissue-class overlay generated successfully.",
        f"Class overlay saved to: {output_overlay_path}",
        heter_msg,
        f"Prediction level used for scaling: {pred_level}",
        f"Patches visualised: {visible_patch_count}",
        f"Alpha: {alpha}",
        f"Minimum display patch size: {min_display_size}px",
        "",
        "Overlay interpretation:",
        "Red = TUM/tumour epithelium.",
        "Purple = STR/cancer-associated stroma.",
        "Blue = LYM/lymphocytes.",
        "Heterogeneity overlay: redder patches have higher abnormality score.",
        "This is tissue-type model confidence, not clinical diagnosis.",
    ]

    return "\n".join(lines)


def tool_summarize_kather_results(
    predictions_json_path: str,
    metrics_json_path: str,
    patch_statistics_json_path: Optional[str] = None,
    output_summary_path: Optional[str] = None,
) -> str:
    if not os.path.exists(predictions_json_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_json_path}")

    if not os.path.exists(metrics_json_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_json_path}")

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        predictions_data = json.load(f)

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    patch_stats = None
    if patch_statistics_json_path and os.path.exists(patch_statistics_json_path):
        with open(patch_statistics_json_path, "r", encoding="utf-8") as f:
            patch_stats = json.load(f)

    class_percentages = metrics.get("class_percentages", {})
    dominant_class = max(class_percentages.items(), key=lambda x: x[1])[0] if class_percentages else "unknown"
    dominant_desc = KATHER_CLASS_DESCRIPTIONS.get(dominant_class, dominant_class)

    high_pct = float(metrics.get("high_abnormality_percentage", 0))
    cluster_count = int(metrics.get("cluster_count", 0))

    summary_lines = [
        "Kather100K Post-Processing Summary",
        "==================================",
        "",
        f"Model: {metrics.get('model_name', predictions_data.get('model_name', 'unknown'))}",
        f"Total predicted patches: {metrics.get('total_predicted_patches', predictions_data.get('patch_count', 'unknown'))}",
        "",
        "Dominant tissue class:",
        f"- {dominant_class} ({dominant_desc})",
        f"- Percentage: {class_percentages.get(dominant_class, 0):.2f}%",
        "",
        "Tumour-relevant indicators:",
        f"- TUM / tumour epithelium patches: {metrics.get('tumour_epithelium_patch_count', 0)}",
        f"- TUM percentage: {metrics.get('tumour_epithelium_percentage', 0):.2f}%",
        f"- High abnormality patches: {metrics.get('high_abnormality_patch_count', 0)}",
        f"- High abnormality percentage: {high_pct:.2f}%",
        f"- Mean tumour epithelium probability: {metrics.get('mean_tumour_epithelium_probability', 0):.6f}",
        f"- Mean abnormality score: {metrics.get('mean_abnormality_score', 0):.6f}",
        f"- Max abnormality score: {metrics.get('max_abnormality_score', 0):.6f}",
        "",
        "Spatial interpretation:",
        f"- Cluster count: {cluster_count}",
        f"- Class entropy: {metrics.get('class_entropy', 0):.6f}",
        f"- Normalised Shannon entropy: {metrics.get('normalised_shannon_entropy', 0):.6f}",
        f"- Colour variance: {metrics.get('colour_variance', 0):.6f}",
        f"- Heterogeneity index: {metrics.get('heterogeneity_index', 0):.6f}",
    ]

    if high_pct >= 30:
        summary_lines.append("- Interpretation: high abnormality is relatively widespread across sampled tissue.")
    elif high_pct >= 10:
        summary_lines.append("- Interpretation: moderate abnormality is present in a subset of sampled tissue.")
    else:
        summary_lines.append("- Interpretation: high abnormality is limited within the sampled patches.")

    if cluster_count >= 20:
        summary_lines.append("- Spatial pattern: abnormal patches appear distributed across many small regions.")
    elif cluster_count > 0:
        summary_lines.append("- Spatial pattern: abnormal patches appear concentrated into fewer hotspot regions.")
    else:
        summary_lines.append("- Spatial pattern: no high-abnormality clusters were detected.")

    if patch_stats:
        summary_lines += [
            "",
            "Patch-level heterogeneity:",
            f"- Colour variance: {patch_stats.get('patch_colour_variance', 0):.6f}",
            f"- Grayscale entropy: {patch_stats.get('shannon_entropy_grayscale', 0):.6f}",
            f"- Heterogeneity index: {patch_stats.get('heterogeneity_index', 0):.6f}",
        ]

    summary_lines += [
        "",
        "Class distribution:",
    ]

    for cls, pct in sorted(class_percentages.items(), key=lambda x: x[1], reverse=True):
        desc = KATHER_CLASS_DESCRIPTIONS.get(cls, cls)
        summary_lines.append(f"- {cls} ({desc}): {pct:.2f}%")

    summary_lines += [
        "",
        "Important limitation:",
        "This is tissue-type classification and model-confidence analysis, not clinical diagnosis.",
    ]

    summary = "\n".join(summary_lines)

    if output_summary_path:
        ensure_parent_dir(output_summary_path)
        with open(output_summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

    return summary


def tool_generate_confidence_histogram(
    predictions_json_path: str,
    output_path: str,
    bins: int = 20,
) -> str:
    if not os.path.exists(predictions_json_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_json_path}")

    ensure_parent_dir(output_path)

    import matplotlib.pyplot as plt

    loaded = _load_kather_predictions(predictions_json_path)
    preds = loaded["predictions"]
    if not preds:
        raise RuntimeError("No predictions found.")

    confidences = [float(p.get("confidence", 0.0)) for p in preds]

    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=int(bins))
    plt.xlabel("Prediction confidence")
    plt.ylabel("Patch count")
    plt.title("Kather100K Patch Prediction Confidence Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    low_conf = sum(1 for c in confidences if c < 0.5)
    mid_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
    high_conf = sum(1 for c in confidences if c >= 0.8)

    total = len(confidences)

    return (
        "Confidence histogram generated successfully.\n"
        f"Saved to: {output_path}\n"
        f"Total patches: {total}\n"
        f"Low confidence (<0.5): {low_conf} ({100 * low_conf / total:.2f}%)\n"
        f"Medium confidence (0.5-0.8): {mid_conf} ({100 * mid_conf / total:.2f}%)\n"
        f"High confidence (>=0.8): {high_conf} ({100 * high_conf / total:.2f}%)"
    )


def tool_generate_hotspot_overlay(
    wsi_path: str,
    predictions_json_path: str,
    thumbnail_path: str,
    output_path: str,
    abnormality_threshold: float = 0.5,
    patch_size: int = 224,
    min_display_size: int = 12,
    max_hotspots: int = 10,
) -> str:
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    if not os.path.exists(predictions_json_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_json_path}")

    if not os.path.exists(thumbnail_path):
        raise FileNotFoundError(f"Thumbnail file not found: {thumbnail_path}")

    ensure_parent_dir(output_path)

    import cv2
    import numpy as np
    from tiatoolbox.wsicore.wsireader import WSIReader

    loaded = _load_kather_predictions(predictions_json_path)
    preds = loaded["predictions"]
    if not preds:
        raise RuntimeError("No predictions found.")

    abnormal_preds = [
        p for p in preds
        if float(p.get("abnormality_score", 0.0)) >= float(abnormality_threshold)
        and int(p.get("x", -1)) >= 0
        and int(p.get("y", -1)) >= 0
    ]

    thumb_bgr = cv2.imread(thumbnail_path)
    if thumb_bgr is None:
        raise RuntimeError(f"Could not read thumbnail: {thumbnail_path}")

    thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = thumb_rgb.copy()
    heat_layer = thumb_rgb.copy()

    wsi = WSIReader.open(wsi_path)
    pred_level, level_w, level_h = _get_prediction_level_dimensions(wsi, preds)

    thumb_h, thumb_w = thumb_rgb.shape[:2]

    scale_x = thumb_w / float(level_w)
    scale_y = thumb_h / float(level_h)

    if not abnormal_preds:
        cv2.putText(
            overlay_rgb,
            "No high-abnormality hotspots detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(output_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        return f"No high-abnormality hotspots detected. Output saved to: {output_path}"

    for p in abnormal_preds:
        x = int(p["x"])
        y = int(p["y"])
        score = float(p.get("abnormality_score", 0.0))

        tx1 = int(x * scale_x)
        ty1 = int(y * scale_y)

        display_w = max(min_display_size, int(patch_size * scale_x))
        display_h = max(min_display_size, int(patch_size * scale_y))

        tx2 = min(thumb_w - 1, tx1 + display_w)
        ty2 = min(thumb_h - 1, ty1 + display_h)

        tx1 = max(0, min(tx1, thumb_w - 1))
        ty1 = max(0, min(ty1, thumb_h - 1))

        intensity = int(255 * max(0.0, min(1.0, score)))
        colour = (255, max(0, 180 - intensity // 2), 0)

        cv2.rectangle(
            heat_layer,
            (tx1, ty1),
            (tx2, ty2),
            colour,
            thickness=-1,
        )

    overlay_rgb = cv2.addWeighted(overlay_rgb, 0.55, heat_layer, 0.45, 0)

    cluster_distance = _estimate_patch_size_from_predictions(
        abnormal_preds,
        fallback=patch_size,
    ) * 1.5

    coords = [(int(p["x"]), int(p["y"])) for p in abnormal_preds]
    visited = [False] * len(abnormal_preds)
    clusters = []

    for i in range(len(abnormal_preds)):
        if visited[i]:
            continue

        q = deque([i])
        visited[i] = True
        cluster = []

        while q:
            cur = q.popleft()
            cluster.append(abnormal_preds[cur])
            x1, y1 = coords[cur]

            for j in range(len(abnormal_preds)):
                if visited[j]:
                    continue

                x2, y2 = coords[j]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if dist <= cluster_distance:
                    visited[j] = True
                    q.append(j)

        clusters.append(cluster)

    clusters = sorted(
        clusters,
        key=lambda c: (
            len(c),
            sum(float(p.get("abnormality_score", 0.0)) for p in c) / max(1, len(c))
        ),
        reverse=True,
    )

    for idx, cluster in enumerate(clusters[:max_hotspots]):
        xs = [int(p["x"]) for p in cluster]
        ys = [int(p["y"]) for p in cluster]
        scores = [float(p.get("abnormality_score", 0.0)) for p in cluster]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs) + patch_size
        y2 = max(ys) + patch_size

        tx1 = int(x1 * scale_x)
        ty1 = int(y1 * scale_y)
        tx2 = int(x2 * scale_x)
        ty2 = int(y2 * scale_y)

        tx1 = max(0, min(tx1, thumb_w - 1))
        ty1 = max(0, min(ty1, thumb_h - 1))
        tx2 = max(tx1 + min_display_size, min(tx2, thumb_w - 1))
        ty2 = max(ty1 + min_display_size, min(ty2, thumb_h - 1))

        mean_score = sum(scores) / len(scores)

        cv2.rectangle(
            overlay_rgb,
            (tx1, ty1),
            (tx2, ty2),
            (255, 0, 0),
            thickness=4,
        )

        cv2.rectangle(
            overlay_rgb,
            (tx1, ty1),
            (tx2, ty2),
            (0, 0, 0),
            thickness=1,
        )

        label = f"H{idx + 1}: n={len(cluster)}, score={mean_score:.2f}"
        cv2.putText(
            overlay_rgb,
            label,
            (tx1, max(25, ty1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay_rgb,
            label,
            (tx1, max(25, ty1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    header = (
        f"Hotspots: {len(abnormal_preds)} high-abnormality patches, "
        f"{len(clusters)} clusters, threshold={abnormality_threshold}"
    )

    cv2.putText(
        overlay_rgb,
        header,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay_rgb,
        header,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    ok = cv2.imwrite(output_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Failed to save hotspot overlay to: {output_path}")

    return (
        "Hotspot overlay generated successfully.\n"
        f"Saved to: {output_path}\n"
        f"Prediction level used for scaling: {pred_level}\n"
        f"Abnormality threshold: {abnormality_threshold}\n"
        f"High-abnormality patches: {len(abnormal_preds)}\n"
        f"Detected hotspots/clusters: {len(clusters)}\n"
        f"Displayed top hotspots: {min(max_hotspots, len(clusters))}"
    )


def tool_compare_masked_vs_unmasked_runs(
    masked_metrics_path: str,
    unmasked_metrics_path: str,
    output_path: Optional[str] = None,
) -> str:
    if not os.path.exists(masked_metrics_path):
        raise FileNotFoundError(f"Masked metrics file not found: {masked_metrics_path}")

    if not os.path.exists(unmasked_metrics_path):
        raise FileNotFoundError(f"Unmasked metrics file not found: {unmasked_metrics_path}")

    with open(masked_metrics_path, "r", encoding="utf-8") as f:
        masked = json.load(f)

    with open(unmasked_metrics_path, "r", encoding="utf-8") as f:
        unmasked = json.load(f)

    def get_num(d, key):
        try:
            return float(d.get(key, 0.0))
        except Exception:
            return 0.0

    comparison = {
        "masked_metrics_path": masked_metrics_path,
        "unmasked_metrics_path": unmasked_metrics_path,
        "patch_count_masked": get_num(masked, "total_predicted_patches"),
        "patch_count_unmasked": get_num(unmasked, "total_predicted_patches"),
        "tumour_epithelium_percentage_masked": get_num(masked, "tumour_epithelium_percentage"),
        "tumour_epithelium_percentage_unmasked": get_num(unmasked, "tumour_epithelium_percentage"),
        "high_abnormality_percentage_masked": get_num(masked, "high_abnormality_percentage"),
        "high_abnormality_percentage_unmasked": get_num(unmasked, "high_abnormality_percentage"),
        "mean_abnormality_score_masked": get_num(masked, "mean_abnormality_score"),
        "mean_abnormality_score_unmasked": get_num(unmasked, "mean_abnormality_score"),
        "cluster_count_masked": get_num(masked, "cluster_count"),
        "cluster_count_unmasked": get_num(unmasked, "cluster_count"),
    }

    comparison["delta_tumour_epithelium_percentage"] = (
        comparison["tumour_epithelium_percentage_unmasked"]
        - comparison["tumour_epithelium_percentage_masked"]
    )
    comparison["delta_high_abnormality_percentage"] = (
        comparison["high_abnormality_percentage_unmasked"]
        - comparison["high_abnormality_percentage_masked"]
    )
    comparison["delta_mean_abnormality_score"] = (
        comparison["mean_abnormality_score_unmasked"]
        - comparison["mean_abnormality_score_masked"]
    )
    comparison["delta_cluster_count"] = (
        comparison["cluster_count_unmasked"]
        - comparison["cluster_count_masked"]
    )

    lines = [
        "Masked vs Unmasked Run Comparison",
        "=================================",
        "",
        f"Masked patch count: {comparison['patch_count_masked']:.0f}",
        f"Unmasked patch count: {comparison['patch_count_unmasked']:.0f}",
        "",
        f"Masked TUM percentage: {comparison['tumour_epithelium_percentage_masked']:.2f}%",
        f"Unmasked TUM percentage: {comparison['tumour_epithelium_percentage_unmasked']:.2f}%",
        f"Delta TUM percentage: {comparison['delta_tumour_epithelium_percentage']:.2f}%",
        "",
        f"Masked high-abnormality percentage: {comparison['high_abnormality_percentage_masked']:.2f}%",
        f"Unmasked high-abnormality percentage: {comparison['high_abnormality_percentage_unmasked']:.2f}%",
        f"Delta high-abnormality percentage: {comparison['delta_high_abnormality_percentage']:.2f}%",
        "",
        f"Masked mean abnormality score: {comparison['mean_abnormality_score_masked']:.6f}",
        f"Unmasked mean abnormality score: {comparison['mean_abnormality_score_unmasked']:.6f}",
        f"Delta mean abnormality score: {comparison['delta_mean_abnormality_score']:.6f}",
        "",
        f"Masked cluster count: {comparison['cluster_count_masked']:.0f}",
        f"Unmasked cluster count: {comparison['cluster_count_unmasked']:.0f}",
        f"Delta cluster count: {comparison['delta_cluster_count']:.0f}",
        "",
    ]

    if comparison["high_abnormality_percentage_unmasked"] > comparison["high_abnormality_percentage_masked"]:
        lines.append("Interpretation: the unmasked run produced a higher abnormality percentage, which may indicate that background or non-tissue patches affected prediction behaviour.")
    else:
        lines.append("Interpretation: the masked run produced equal or higher abnormality percentage, suggesting tissue masking did not inflate abnormality estimates.")

    lines.append("")
    lines.append("Important: this comparison reflects model-confidence behaviour, not clinical diagnosis.")

    text = "\n".join(lines)

    if output_path:
        ensure_parent_dir(output_path)
        if output_path.lower().endswith(".json"):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

    return text


def tool_generate_tumour_likelihood_map(
    wsi_path: str,
    predictions_json_path: str,
    thumbnail_path: str,
    output_path: str,
    patch_size: int = 224,
    alpha: float = 0.55,
    blur_kernel: int = 31,
) -> str:
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    if not os.path.exists(predictions_json_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_json_path}")

    if not os.path.exists(thumbnail_path):
        raise FileNotFoundError(f"Thumbnail file not found: {thumbnail_path}")

    ensure_parent_dir(output_path)

    import cv2
    import numpy as np
    from tiatoolbox.wsicore.wsireader import WSIReader

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    if not preds:
        raise RuntimeError("No predictions found.")

    thumb_bgr = cv2.imread(thumbnail_path)
    if thumb_bgr is None:
        raise RuntimeError(f"Could not read thumbnail: {thumbnail_path}")

    thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
    thumb_h, thumb_w = thumb_rgb.shape[:2]

    wsi = WSIReader.open(wsi_path)
    pred_level, level_w, level_h = _get_prediction_level_dimensions(wsi, preds)

    scale_x = thumb_w / float(level_w)
    scale_y = thumb_h / float(level_h)

    score_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    weight_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    valid_count = 0

    for p in preds:
        x = int(p.get("x", -1))
        y = int(p.get("y", -1))

        if x < 0 or y < 0:
            continue

        score = float(p.get("abnormality_score", 0.0))
        score = max(0.0, min(1.0, score))

        tx1 = int(x * scale_x)
        ty1 = int(y * scale_y)
        tx2 = int((x + patch_size) * scale_x)
        ty2 = int((y + patch_size) * scale_y)

        tx1 = max(0, min(tx1, thumb_w - 1))
        ty1 = max(0, min(ty1, thumb_h - 1))
        tx2 = max(tx1 + 1, min(tx2, thumb_w))
        ty2 = max(ty1 + 1, min(ty2, thumb_h))

        score_map[ty1:ty2, tx1:tx2] += score
        weight_map[ty1:ty2, tx1:tx2] += 1.0
        valid_count += 1

    nonzero = weight_map > 0
    likelihood_map = np.zeros_like(score_map)
    likelihood_map[nonzero] = score_map[nonzero] / weight_map[nonzero]

    if blur_kernel and blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        likelihood_map = cv2.GaussianBlur(likelihood_map, (blur_kernel, blur_kernel), 0)

    heat_uint8 = np.clip(likelihood_map * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb = cv2.addWeighted(
        thumb_rgb,
        1.0 - float(alpha),
        heat_rgb,
        float(alpha),
        0,
    )

    cv2.putText(
        overlay_rgb,
        "Tumour-relevant likelihood map: blue=low, red=high",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay_rgb,
        "Tumour-relevant likelihood map: blue=low, red=high",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    ok = cv2.imwrite(output_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Failed to save tumour likelihood map to: {output_path}")

    return (
        "Tumour likelihood map generated successfully.\n"
        f"Saved to: {output_path}\n"
        f"Prediction level used for scaling: {pred_level}\n"
        f"Patches used: {valid_count}\n"
        f"Mean map score: {float(np.mean(likelihood_map[nonzero])) if np.any(nonzero) else 0.0:.6f}\n"
        f"Max map score: {float(np.max(likelihood_map)):.6f}\n"
        "Important: this is tumour-relevant tissue likelihood/model confidence, not clinical diagnosis."
    )


def tool_threshold_sensitivity_analysis(
    predictions_json_path: str,
    output_json_path: str,
    output_csv_path: Optional[str] = None,
    output_plot_path: Optional[str] = None,
    thresholds: Optional[List[float]] = None,
) -> str:
    if not os.path.exists(predictions_json_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_json_path}")

    ensure_parent_dir(output_json_path)

    if output_csv_path:
        ensure_parent_dir(output_csv_path)

    if output_plot_path:
        ensure_parent_dir(output_plot_path)

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    if not preds:
        raise RuntimeError("No predictions found.")

    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    thresholds = [float(t) for t in thresholds]

    total = len(preds)
    results = []

    for threshold in thresholds:
        abnormal = [
            p for p in preds
            if float(p.get("abnormality_score", 0.0)) >= threshold
        ]

        cluster_count = _count_clusters(abnormal)

        results.append({
            "threshold": threshold,
            "high_abnormality_patch_count": len(abnormal),
            "high_abnormality_percentage": float(100.0 * len(abnormal) / total),
            "cluster_count": int(cluster_count),
        })

    output = {
        "source_predictions": predictions_json_path,
        "total_patches": total,
        "threshold_results": results,
        "clinical_warning": "Threshold sensitivity reflects model-confidence behaviour, not clinical diagnosis.",
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if output_csv_path:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "threshold",
                    "high_abnormality_patch_count",
                    "high_abnormality_percentage",
                    "cluster_count",
                ],
            )
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    if output_plot_path:
        import matplotlib.pyplot as plt

        xs = [r["threshold"] for r in results]
        ys_pct = [r["high_abnormality_percentage"] for r in results]
        ys_clusters = [r["cluster_count"] for r in results]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys_pct, marker="o", label="High abnormality %")
        plt.plot(xs, ys_clusters, marker="o", label="Cluster count")
        plt.xlabel("Abnormality threshold")
        plt.ylabel("Value")
        plt.title("Threshold Sensitivity Analysis")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=200)
        plt.close()

    lines = [
        "Threshold sensitivity analysis completed successfully.",
        f"Predictions: {predictions_json_path}",
        f"JSON saved to: {output_json_path}",
    ]

    if output_csv_path:
        lines.append(f"CSV saved to: {output_csv_path}")

    if output_plot_path:
        lines.append(f"Plot saved to: {output_plot_path}")

    lines.append("")
    lines.append("Results:")
    for row in results:
        lines.append(
            f"- Threshold {row['threshold']:.2f}: "
            f"{row['high_abnormality_patch_count']} patches "
            f"({row['high_abnormality_percentage']:.2f}%), "
            f"{row['cluster_count']} clusters"
        )

    lines.append("")
    lines.append("Important: this is model-confidence sensitivity analysis, not clinical diagnosis.")

    return "\n".join(lines)


def tool_extract_top_abnormal_patches(
    predictions_json_path: str,
    output_dir: str,
    top_k: int = 20,
    output_csv_path: Optional[str] = None,
    output_grid_path: Optional[str] = None,
) -> str:
    if not os.path.exists(predictions_json_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_json_path}")

    os.makedirs(output_dir, exist_ok=True)

    if output_csv_path:
        ensure_parent_dir(output_csv_path)

    if output_grid_path:
        ensure_parent_dir(output_grid_path)

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    if not preds:
        raise RuntimeError("No predictions found.")

    ranked = sorted(
        preds,
        key=lambda p: float(p.get("abnormality_score", 0.0)),
        reverse=True,
    )[:int(top_k)]

    rows = []
    copied_paths = []

    for idx, p in enumerate(ranked, start=1):
        src = p.get("patch_path", "")
        score = float(p.get("abnormality_score", 0.0))
        cls = p.get("predicted_class", "UNK")
        confidence = float(p.get("confidence", 0.0))

        safe_name = f"top_{idx:03d}_{cls}_score_{score:.3f}_conf_{confidence:.3f}.png"
        dst = os.path.join(output_dir, safe_name)

        if src and os.path.exists(src):
            shutil.copy2(src, dst)
            copied_paths.append(dst)
        else:
            dst = ""

        rows.append({
            "rank": idx,
            "saved_patch_path": dst,
            "source_patch_path": src,
            "filename": p.get("filename", os.path.basename(src)),
            "predicted_class": cls,
            "confidence": confidence,
            "abnormality_score": score,
            "tumour_epithelium_probability": float(p.get("tumour_epithelium_probability", 0.0)),
            "stroma_probability": float(p.get("stroma_probability", 0.0)),
            "lymphocyte_probability": float(p.get("lymphocyte_probability", 0.0)),
            "x": int(p.get("x", -1)),
            "y": int(p.get("y", -1)),
            "level": int(p.get("level", 0)),
        })

    if output_csv_path:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "saved_patch_path",
                    "source_patch_path",
                    "filename",
                    "predicted_class",
                    "confidence",
                    "abnormality_score",
                    "tumour_epithelium_probability",
                    "stroma_probability",
                    "lymphocyte_probability",
                    "x",
                    "y",
                    "level",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    if output_grid_path and copied_paths:
        import cv2
        import numpy as np

        imgs = []
        for idx, path in enumerate(copied_paths, start=1):
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)

            label = f"#{idx}"
            cv2.putText(
                img,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            imgs.append(img)

        if imgs:
            cols = min(5, len(imgs))
            rows_count = int(math.ceil(len(imgs) / cols))
            blank = np.ones((160, 160, 3), dtype=np.uint8) * 255

            grid_rows = []
            for r in range(rows_count):
                row_imgs = []
                for c in range(cols):
                    idx = r * cols + c
                    row_imgs.append(imgs[idx] if idx < len(imgs) else blank)
                grid_rows.append(np.hstack(row_imgs))

            grid = np.vstack(grid_rows)
            cv2.imwrite(output_grid_path, grid)

    lines = [
        "Top abnormal patches extracted successfully.",
        f"Predictions: {predictions_json_path}",
        f"Output directory: {output_dir}",
        f"Requested top_k: {top_k}",
        f"Patches copied: {len(copied_paths)}",
    ]

    if output_csv_path:
        lines.append(f"CSV saved to: {output_csv_path}")

    if output_grid_path:
        lines.append(f"Grid image saved to: {output_grid_path}")

    lines.append("")
    lines.append("Important: these are the highest model-confidence abnormal patches, not clinical diagnosis.")

    return "\n".join(lines)


def tool_generate_final_ai_report(
    predictions_json_path: str,
    metrics_json_path: str,
    patch_statistics_json_path: Optional[str] = None,
    threshold_sensitivity_json_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
) -> str:
    if not os.path.exists(predictions_json_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_json_path}")

    if not os.path.exists(metrics_json_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_json_path}")

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        predictions_data = json.load(f)

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    patch_stats = None
    if patch_statistics_json_path and os.path.exists(patch_statistics_json_path):
        with open(patch_statistics_json_path, "r", encoding="utf-8") as f:
            patch_stats = json.load(f)

    threshold_data = None
    if threshold_sensitivity_json_path and os.path.exists(threshold_sensitivity_json_path):
        with open(threshold_sensitivity_json_path, "r", encoding="utf-8") as f:
            threshold_data = json.load(f)

    class_percentages = metrics.get("class_percentages", {})
    dominant_class = max(class_percentages.items(), key=lambda x: x[1])[0] if class_percentages else "unknown"
    dominant_desc = KATHER_CLASS_DESCRIPTIONS.get(dominant_class, dominant_class)

    total_patches = metrics.get("total_predicted_patches", predictions_data.get("patch_count", "unknown"))
    tum_pct = float(metrics.get("tumour_epithelium_percentage", 0.0))
    high_pct = float(metrics.get("high_abnormality_percentage", 0.0))
    cluster_count = int(metrics.get("cluster_count", 0))
    class_entropy = float(metrics.get("class_entropy", 0.0))
    mean_abn = float(metrics.get("mean_abnormality_score", 0.0))
    max_abn = float(metrics.get("max_abnormality_score", 0.0))

    if high_pct >= 30:
        abnormality_interpretation = "High-abnormality patches are widespread across the sampled tissue."
    elif high_pct >= 10:
        abnormality_interpretation = "High-abnormality patches are present in a moderate subset of sampled tissue."
    else:
        abnormality_interpretation = "High-abnormality patches are limited within the sampled regions."

    if cluster_count >= 20:
        spatial_interpretation = "The abnormality pattern appears spatially dispersed across multiple small regions."
    elif cluster_count > 0:
        spatial_interpretation = "The abnormality pattern appears concentrated into a smaller number of hotspot regions."
    else:
        spatial_interpretation = "No high-abnormality clusters were detected at the selected threshold."

    report_lines = [
        "Final AI Interpretability Report",
        "================================",
        "",
        "1. System Overview",
        "------------------",
        "This report summarises the outputs of a patch-based histopathology analysis pipeline using ResNet18-Kather100K.",
        "The system classifies extracted tissue patches into Kather100K tissue classes and derives tumour-relevant indicators from the TUM and STR probabilities.",
        "",
        "Important clinical limitation:",
        "This system provides tissue-type classification and model-confidence analysis only. It is not a clinical diagnostic system.",
        "",
        "2. Input and Model Summary",
        "--------------------------",
        f"- Model: {metrics.get('model_name', predictions_data.get('model_name', 'unknown'))}",
        f"- Total predicted patches: {total_patches}",
        f"- Prediction source: {predictions_json_path}",
        "",
        "3. Tissue Class Distribution",
        "----------------------------",
        f"- Dominant tissue class: {dominant_class} ({dominant_desc})",
        f"- Dominant class percentage: {class_percentages.get(dominant_class, 0):.2f}%",
        "",
    ]

    for cls, pct in sorted(class_percentages.items(), key=lambda x: x[1], reverse=True):
        desc = KATHER_CLASS_DESCRIPTIONS.get(cls, cls)
        report_lines.append(f"- {cls} ({desc}): {pct:.2f}%")

    report_lines += [
        "",
        "4. Tumour-Relevant Tissue Likelihood",
        "------------------------------------",
        f"- TUM / tumour epithelium percentage: {tum_pct:.2f}%",
        f"- High abnormality percentage: {high_pct:.2f}%",
        f"- Mean abnormality score: {mean_abn:.6f}",
        f"- Max abnormality score: {max_abn:.6f}",
        f"- Interpretation: {abnormality_interpretation}",
        "",
        "5. Spatial Heterogeneity and Hotspots",
        "-------------------------------------",
        f"- Cluster count: {cluster_count}",
        f"- Class entropy: {class_entropy:.6f}",
        f"- Spatial interpretation: {spatial_interpretation}",
        "",
    ]

    if patch_stats:
        report_lines += [
            "6. Patch-Level Statistical Heterogeneity",
            "----------------------------------------",
            f"- Patch colour variance: {float(patch_stats.get('patch_colour_variance', 0.0)):.6f}",
            f"- Grayscale entropy: {float(patch_stats.get('shannon_entropy_grayscale', 0.0)):.6f}",
            f"- Heterogeneity index: {float(patch_stats.get('heterogeneity_index', 0.0)):.6f}",
            "Interpretation: higher colour variance and entropy suggest greater visual diversity across sampled tissue patches.",
            "",
        ]

    if threshold_data:
        report_lines += [
            "7. Threshold Sensitivity Analysis",
            "---------------------------------",
            "The following results show how abnormality percentage and cluster count change as the abnormality threshold varies.",
            "",
        ]

        for row in threshold_data.get("threshold_results", []):
            report_lines.append(
                f"- Threshold {float(row.get('threshold', 0.0)):.2f}: "
                f"{int(row.get('high_abnormality_patch_count', 0))} high-abnormality patches, "
                f"{float(row.get('high_abnormality_percentage', 0.0)):.2f}%, "
                f"{int(row.get('cluster_count', 0))} clusters"
            )

        report_lines.append("")

    report_lines += [
        "8. Interpretability Outputs",
        "---------------------------",
        "Recommended outputs to inspect alongside this report:",
        "- kather_class_overlay.png: tissue-class overlay",
        "- heterogeneity_overlay.png: abnormality/uncertainty overlay",
        "- tumour_likelihood_map.png: continuous tumour-relevant likelihood map",
        "- hotspot_overlay.png: spatial hotspot visualisation",
        "- top_abnormal_patches/: patches contributing most strongly to abnormality score",
        "- confidence_histogram.png: prediction confidence distribution",
        "",
        "9. Overall Conclusion",
        "---------------------",
        "The MVP demonstrates a transparent and interpretable patch-based workflow that combines pretrained deep learning classification with statistical, spatial, and visual post-processing.",
        "The outputs support analysis of tissue composition, tumour-relevant likelihood, spatial clustering, heterogeneity, and prediction confidence.",
        "",
        "Final caution:",
        "All outputs should be interpreted as model-derived indicators for research and educational purposes, not as clinical diagnoses.",
    ]

    report = "\n".join(report_lines)

    if output_report_path:
        ensure_parent_dir(output_report_path)
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def _kongnet_region_scores(region: Dict[str, Any]) -> Dict[str, float]:
    """Derive size-aware, explicitly defined scores from one ROI record."""
    counts = region.get("class_counts", {})
    percentages = region.get("class_percentages", {})
    pair_counts = region.get("pair_counts", {})
    cell_count = max(int(region.get("cell_count", 0)), 1)
    density = float(region.get("cell_density_per_square_unit", 0.0))
    inflammatory_pct = float(sum(float(percentages.get(name, 0.0)) for name in KONGNET_IMMUNE_CELL_TYPES))
    neoplastic_pct = float(percentages.get("Neoplastic", 0.0))
    connective_pct = float(percentages.get("Connective", 0.0))
    epithelial_pct = float(percentages.get("Epithelial", 0.0))
    immune_count = int(sum(int(counts.get(name, 0)) for name in KONGNET_IMMUNE_CELL_TYPES))
    target_types = {"Neoplastic"} if int(counts.get("Neoplastic", 0)) > 0 else ({"Epithelial"} if int(counts.get("Epithelial", 0)) > 0 else set())
    target_count = int(sum(int(counts.get(name, 0)) for name in target_types))
    interaction_pairs = _kongnet_pair_count(pair_counts, KONGNET_IMMUNE_CELL_TYPES, target_types)
    possible_cross_pairs = immune_count * target_count
    return {
        "inflammatory_percentage": inflammatory_pct,
        "neoplastic_percentage": neoplastic_pct,
        "connective_percentage": connective_pct,
        "epithelial_percentage": epithelial_pct,
        "inflammatory_density": density * inflammatory_pct / 100.0,
        "neoplastic_density": density * neoplastic_pct / 100.0,
        "tumour_immune_pair_count": float(interaction_pairs),
        "tumour_immune_pairs_per_100_cells": interaction_pairs / cell_count * 100.0,
        "tumour_immune_contact_fraction": interaction_pairs / possible_cross_pairs if possible_cross_pairs else 0.0,
        "interaction_target_types": ", ".join(sorted(target_types)) if target_types else "",
    }


def _rank_kongnet_regions(regions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    enriched = []
    for region in regions:
        row = {
            "region_id": region.get("region_id"),
            "region_label": region.get("region_label"),
            "cell_count": int(region.get("cell_count", 0)),
            **_kongnet_region_scores(region),
        }
        enriched.append(row)
    ranking_metrics = {
        "inflammatory": "inflammatory_percentage",
        "neoplastic": "neoplastic_percentage",
        "connective_stromal": "connective_percentage",
        "epithelial": "epithelial_percentage",
        "tumour_immune_interaction": "tumour_immune_pairs_per_100_cells",
        "cell_density": "cell_count",
    }
    return {
        name: sorted(enriched, key=lambda row: (row[metric], row["cell_count"]), reverse=True)
        for name, metric in ranking_metrics.items()
    }


def tool_rank_kongnet_regions(
    regions_json_path: str,
    output_json_path: str,
    output_txt_path: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Rank local ROIs by cell composition and tumour-immune interaction."""
    if not os.path.exists(regions_json_path):
        raise FileNotFoundError(f"Regions JSON not found: {regions_json_path}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    with open(regions_json_path, "r", encoding="utf-8") as file:
        region_data = json.load(file)
    regions = region_data.get("regions", [])
    if not regions:
        raise ValueError("Regions JSON contains no ROI records.")
    rankings = _rank_kongnet_regions(regions)
    payload = {
        "regions_json_path": regions_json_path,
        "region_count": len(regions),
        "top_k": min(top_k, len(regions)),
        "score_definitions": {
            "inflammatory": "Percentage of predicted inflammatory nuclei in the ROI.",
            "neoplastic": "Percentage of predicted neoplastic nuclei in the ROI.",
            "connective_stromal": "Percentage of predicted connective nuclei in the ROI.",
            "epithelial": "Percentage of predicted epithelial nuclei in the ROI.",
            "tumour_immune_interaction": "Immune-neoplastic neighbour pairs within the configured radius per 100 ROI cells; for CoNIC outputs without a neoplastic class, immune-epithelial pairs are used.",
        },
        "rankings": {name: rows[:top_k] for name, rows in rankings.items()},
        "clinical_warning": "Rankings describe model-derived spatial patterns, not diagnoses.",
    }
    _write_json(output_json_path, payload)

    titles = {
        "inflammatory": "Top inflammatory regions",
        "neoplastic": "Top neoplastic regions",
        "connective_stromal": "Top connective/stromal regions",
        "epithelial": "Top epithelial regions",
        "tumour_immune_interaction": "Top tumour-immune interaction regions",
        "cell_density": "Highest cell-density regions",
    }
    value_keys = {
        "inflammatory": "inflammatory_percentage",
        "neoplastic": "neoplastic_percentage",
        "connective_stromal": "connective_percentage",
        "epithelial": "epithelial_percentage",
        "tumour_immune_interaction": "tumour_immune_pairs_per_100_cells",
        "cell_density": "cell_count",
    }
    lines = ["KongNet Region Rankings", "=======================", ""]
    for name, rows in payload["rankings"].items():
        lines.extend([titles[name], "-" * len(titles[name])])
        for index, row in enumerate(rows, start=1):
            value = row[value_keys[name]]
            suffix = "%" if name in {"inflammatory", "neoplastic", "connective_stromal", "epithelial"} else (
                " pairs per 100 cells" if name == "tumour_immune_interaction" else " cells"
            )
            lines.append(f"{index}. {row['region_id']} - {value:.2f}{suffix} ({row['region_label']})")
        lines.append("")
    lines.append("These rankings are model-derived research outputs and are not clinical diagnoses.")
    ranking_text = "\n".join(lines)
    if output_txt_path:
        root, extension = os.path.splitext(output_txt_path)
        output_txt_path = output_txt_path if extension.lower() == ".txt" else f"{root if extension else output_txt_path}.txt"
        ensure_parent_dir(output_txt_path)
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write(ranking_text)
    return ranking_text


def tool_answer_kongnet_spatial_question(
    question: str,
    regions_json_path: str,
    output_txt_path: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Answer common spatial-pathology questions using transparent ROI rankings."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty.")
    if not os.path.exists(regions_json_path):
        raise FileNotFoundError(f"Regions JSON not found: {regions_json_path}")
    with open(regions_json_path, "r", encoding="utf-8") as file:
        regions = json.load(file).get("regions", [])
    if not regions:
        raise ValueError("Regions JSON contains no ROI records.")
    rankings = _rank_kongnet_regions(regions)
    query = question.casefold()
    if any(term in query for term in ("tumour-immune", "tumor-immune", "tumour immune", "tumor immune", "interaction")):
        category, metric, descriptor = "tumour_immune_interaction", "tumour_immune_pairs_per_100_cells", "tumour-immune interaction"
        unit = " inflammatory-neoplastic pairs per 100 cells"
    elif any(term in query for term in ("inflamm", "immune")):
        category, metric, descriptor = "inflammatory", "inflammatory_percentage", "inflammatory infiltration"
        unit = "% inflammatory cells"
    elif any(term in query for term in ("neoplast", "tumour", "tumor")):
        category, metric, descriptor = "neoplastic", "neoplastic_percentage", "neoplastic composition"
        unit = "% neoplastic cells"
    elif any(term in query for term in ("strom", "connective")):
        category, metric, descriptor = "connective_stromal", "connective_percentage", "connective/stromal composition"
        unit = "% connective cells"
    elif "epithelial" in query:
        category, metric, descriptor = "epithelial", "epithelial_percentage", "epithelial composition"
        unit = "% epithelial cells"
    elif any(term in query for term in ("dense", "density", "cell-rich")):
        category, metric, descriptor = "cell_density", "cell_count", "cell abundance"
        unit = " cells"
    else:
        category, metric, descriptor = "tumour_immune_interaction", "tumour_immune_pairs_per_100_cells", "tumour-immune interaction"
        unit = " inflammatory-neoplastic pairs per 100 cells"

    ranked = rankings[category]
    selected = ranked[:min(top_k, len(ranked))]
    maximum = ranked[0][metric] if ranked else 0.0
    lines = [
        f"Question: {question.strip()}",
        f"Answer: regions ranked by {descriptor}",
        "",
    ]
    for index, row in enumerate(selected, start=1):
        relative = row[metric] / maximum if maximum else 0.0
        strength = "High" if relative >= 0.67 else "Moderate" if relative >= 0.33 else "Low"
        lines.append(
            f"{index}. {row['region_id']}: {strength} {descriptor} - "
            f"{row[metric]:.2f}{unit}; {row['region_label']}."
        )
    derivation = (
        "fixed-ROI inflammatory-neoplastic neighbour pairs were normalized per 100 cells and ranked."
        if category == "tumour_immune_interaction"
        else f"fixed-ROI model-predicted {descriptor} measurements were ranked."
    )
    lines.extend([
        "",
        f"How this was derived: {derivation}",
        "Caution: this is an explainable query over model predictions, not a pathological diagnosis or proof of cellular interaction.",
    ])
    answer = "\n".join(lines)
    if output_txt_path:
        root, extension = os.path.splitext(output_txt_path)
        output_txt_path = output_txt_path if extension.lower() == ".txt" else f"{root if extension else output_txt_path}.txt"
        ensure_parent_dir(output_txt_path)
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write(answer)
    return answer


def tool_generate_kongnet_slide_summary(
    nuclei_csv_path: str,
    regions_json_path: str,
    output_txt_path: str,
    output_json_path: Optional[str] = None,
) -> str:
    """Create a concise examiner-friendly whole-slide spatial summary."""
    if not os.path.exists(nuclei_csv_path):
        raise FileNotFoundError(f"Nuclei CSV not found: {nuclei_csv_path}")
    if not os.path.exists(regions_json_path):
        raise FileNotFoundError(f"Regions JSON not found: {regions_json_path}")
    with open(nuclei_csv_path, "r", encoding="utf-8", newline="") as file:
        nuclei = list(csv.DictReader(file))
    with open(regions_json_path, "r", encoding="utf-8") as file:
        regions = json.load(file).get("regions", [])
    if not nuclei or not regions:
        raise ValueError("Slide summary requires nuclei and at least one ROI.")
    counts = Counter(row.get("type") or "Unknown" for row in nuclei)
    rankings = _rank_kongnet_regions(regions)
    total = len(nuclei)
    most_inflammatory = rankings["inflammatory"][0]
    most_neoplastic = rankings["neoplastic"][0]
    most_interactive = rankings["tumour_immune_interaction"][0]
    summary = {
        "total_nuclei": total,
        "class_counts": dict(counts),
        "class_percentages": {name: count / total * 100 for name, count in counts.items()},
        "region_count": len(regions),
        "most_inflammatory_region": most_inflammatory,
        "most_neoplastic_region": most_neoplastic,
        "most_tumour_immune_interactive_region": most_interactive,
        "clinical_warning": "Model-derived research summary; not a clinical diagnosis.",
    }
    lines = [
        "KongNet Slide Summary",
        "=====================",
        "",
        f"Total nuclei: {total:,}",
        f"Analysed ROIs: {len(regions)}",
        "",
        "Predicted cell composition",
        "--------------------------",
    ]
    preferred_order = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    for name in preferred_order + sorted(set(counts) - set(preferred_order)):
        count = counts.get(name, 0)
        lines.append(f"- {name}: {count:,} ({count / total * 100:.2f}%)")
    lines += [
        "",
        "Regional highlights",
        "-------------------",
        f"- Most inflammatory region: {most_inflammatory['region_id']} "
        f"({most_inflammatory['inflammatory_percentage']:.2f}% inflammatory cells)",
        f"- Most tumour-rich region: {most_neoplastic['region_id']} "
        f"({most_neoplastic['neoplastic_percentage']:.2f}% neoplastic cells)",
        f"- Strongest tumour-immune interaction region: {most_interactive['region_id']} "
        f"({most_interactive['tumour_immune_pairs_per_100_cells']:.2f} inflammatory-neoplastic pairs per 100 cells)",
        "",
        "Interpretation",
        "--------------",
        "These highlights identify where model-predicted cell composition and local proximity patterns are strongest on this slide.",
        "They should be verified against the histology and nucleus/ROI overlays and must not be interpreted as a clinical diagnosis.",
    ]
    text_summary = "\n".join(lines)
    root, extension = os.path.splitext(output_txt_path)
    output_txt_path = output_txt_path if extension.lower() == ".txt" else f"{root if extension else output_txt_path}.txt"
    ensure_parent_dir(output_txt_path)
    with open(output_txt_path, "w", encoding="utf-8") as file:
        file.write(text_summary)
    _write_json(output_json_path, summary)
    return text_summary


def tool_generate_kongnet_ai_report(
    nuclei_csv_path: str,
    cooccurrence_json_path: Optional[str] = None,
    neighbourhood_json_path: Optional[str] = None,
    nearest_neighbour_json_path: Optional[str] = None,
    regions_json_path: Optional[str] = None,
    communities_json_path: Optional[str] = None,
    rankings_json_path: Optional[str] = None,
    slide_summary_json_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
) -> str:
    """Generate a research-oriented interpretability report for KongNet outputs."""
    if not os.path.exists(nuclei_csv_path):
        raise FileNotFoundError(f"Nuclei CSV not found: {nuclei_csv_path}")

    if not output_report_path:
        output_report_path = os.path.join(
            os.path.dirname(os.path.abspath(nuclei_csv_path)),
            "kongnet_ai_interpretability_report.txt",
        )
    else:
        report_root, report_extension = os.path.splitext(output_report_path)
        if report_extension.lower() != ".txt":
            output_report_path = f"{report_root if report_extension else output_report_path}.txt"

    with open(nuclei_csv_path, "r", encoding="utf-8", newline="") as file:
        nuclei = list(csv.DictReader(file))
    if not nuclei:
        raise ValueError("The nuclei CSV contains no detections.")

    def load_optional(path: Optional[str], label: str) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    cooccurrence = load_optional(cooccurrence_json_path, "Co-occurrence JSON")
    neighbourhood = load_optional(neighbourhood_json_path, "Neighbourhood JSON")
    nearest = load_optional(nearest_neighbour_json_path, "Nearest-neighbour JSON")
    regions_data = load_optional(regions_json_path, "Regions JSON")
    communities_data = load_optional(communities_json_path, "Communities JSON")
    rankings_data = load_optional(rankings_json_path, "Region rankings JSON")
    slide_summary_data = load_optional(slide_summary_json_path, "Slide summary JSON")

    counts = Counter(row.get("type", "Unknown") or "Unknown" for row in nuclei)
    inferred_model_name = _infer_kongnet_model_name_from_counts(counts)
    probabilities = []
    for row in nuclei:
        try:
            probabilities.append(float(row.get("probability", "")))
        except (TypeError, ValueError):
            pass
    total = len(nuclei)
    dominant_type, dominant_count = counts.most_common(1)[0]
    mean_probability = sum(probabilities) / len(probabilities) if probabilities else None
    low_confidence = sum(value < 0.5 for value in probabilities)

    key_findings = [
        f"KongNet identified {total:,} nuclei. The most common predicted type was "
        f"{dominant_type} ({dominant_count / total * 100:.1f}% of detections)."
    ]
    if probabilities:
        key_findings.append(
            f"The mean model probability was {mean_probability:.3f}; "
            f"{low_confidence / len(probabilities) * 100:.1f}% of scored detections were below 0.50. "
            "This indicates model confidence only, not confirmed accuracy."
        )
    if cooccurrence:
        pair_count = cooccurrence.get("undirected_pair_count")
        radius = cooccurrence.get("radius", "the selected")
        units = cooccurrence.get("distance_units", "distance units")
        key_findings.append(
            f"The spatial analysis found {int(pair_count):,} nearby cell pairs within "
            f"{radius} {units}." if isinstance(pair_count, (int, float)) else
            f"Cell co-occurrence was assessed within {radius} {units}."
        )
    if nearest:
        pair_summaries = nearest.get("pair_summaries", {})
        if pair_summaries:
            strongest = max(pair_summaries.items(), key=lambda item: item[1].get("count", 0))
            key_findings.append(
                f"The most frequently observed nearest-neighbour relationship was "
                f"{strongest[0].replace('->', ' to ')} ({strongest[1].get('count', 0):,} observations)."
            )
    if regions_data:
        key_findings.append(
            f"Local ROI analysis retained {regions_data.get('region_count', 0):,} regions, allowing findings to be compared across the slide rather than reported globally only."
        )
    if communities_data:
        key_findings.append(
            f"Per-cell neighbourhood profiles formed {communities_data.get('community_count', 0)} model-derived spatial communities."
        )
    if rankings_data:
        interaction_rows = rankings_data.get("rankings", {}).get("tumour_immune_interaction", [])
        if interaction_rows:
            row = interaction_rows[0]
            key_findings.append(
                f"{row.get('region_id')} ranked highest for immune-to-epithelial/tumour proximity "
                f"({row.get('tumour_immune_pairs_per_100_cells', 0):.2f} pairs per 100 cells)."
            )

    lines = [
        "KongNet AI Interpretability Report",
        "==================================",
        "",
        "1. Executive Summary",
        "--------------------",
        *[f"- {finding}" for finding in key_findings],
        "- These findings describe model predictions in the analysed region; they are not a diagnosis or evidence of biological causation.",
        "",
        "2. System Overview",
        "------------------",
        f"This report summarises nucleus detection, nucleus-type classification, confidence, and available spatial analyses from {inferred_model_name}.",
        "It describes model-derived patterns and does not establish tissue diagnosis, prognosis, or treatment guidance.",
        "",
        "3. Detection Summary",
        "--------------------",
        f"- Model: {inferred_model_name}",
        f"- Detected nuclei: {total}",
        f"- Dominant predicted cell type: {dominant_type} ({dominant_count / total * 100:.2f}%)",
        f"- Detection source: {nuclei_csv_path}",
        "",
        "4. Predicted Cell-Type Composition",
        "----------------------------------",
    ]
    for cell_type, count in counts.most_common():
        lines.append(f"- {cell_type}: {count} ({count / total * 100:.2f}%)")

    lines += ["", "5. Model Confidence", "-------------------"]
    if probabilities:
        lines += [
            f"- Detections with probability values: {len(probabilities)} of {total}",
            f"- Mean predicted probability: {mean_probability:.4f}",
            f"- Minimum / maximum probability: {min(probabilities):.4f} / {max(probabilities):.4f}",
            f"- Probability below 0.50: {low_confidence} ({low_confidence / len(probabilities) * 100:.2f}%)",
            "Interpretation: probabilities reflect model confidence, not correctness or calibrated clinical certainty.",
        ]
    else:
        lines.append("- No usable probability values were present; confidence could not be summarised.")

    lines += ["", "6. Spatial Cell Relationships", "-----------------------------"]
    if cooccurrence:
        inflammatory_epithelial = cooccurrence.get("inflammatory_to_epithelial_ratio")
        inflammatory_neoplastic = cooccurrence.get("inflammatory_to_neoplastic_ratio")
        lines += [
            f"- Co-occurrence radius: {cooccurrence.get('radius', 'unknown')} {cooccurrence.get('distance_units', '')}".rstrip(),
            f"- Cell pairs within radius: {cooccurrence.get('undirected_pair_count', 'unknown')}",
            f"- Inflammatory-to-epithelial ratio: {float(inflammatory_epithelial):.2f}" if inflammatory_epithelial is not None else "- Inflammatory-to-epithelial ratio: unavailable (no epithelial detections)",
            f"- Inflammatory-to-neoplastic ratio: {float(inflammatory_neoplastic):.2f}" if inflammatory_neoplastic is not None else "- Inflammatory-to-neoplastic ratio: unavailable (no neoplastic detections)",
            "Interpretation: these ratios compare predicted cell counts; a larger ratio means more inflammatory detections relative to the named comparison type.",
        ]
    else:
        lines.append("- Co-occurrence analysis was not supplied.")
    if neighbourhood:
        source_count = neighbourhood.get("source_nuclei")
        sources_with_neighbours = neighbourhood.get("sources_with_neighbours")
        lines += [
            f"- Mean neighbours per source: {float(neighbourhood.get('mean_neighbours_per_source', 0)):.3f}",
            f"- Sources with neighbours: {neighbourhood.get('sources_with_neighbours', 'unknown')} of {neighbourhood.get('source_nuclei', 'unknown')}",
        ]
        if isinstance(source_count, (int, float)) and source_count and isinstance(sources_with_neighbours, (int, float)):
            lines.append(f"Interpretation: {sources_with_neighbours / source_count * 100:.1f}% of source cells had at least one selected target within the chosen radius.")
    else:
        lines.append("- Radius-neighbourhood analysis was not supplied.")
    if nearest:
        lines += [
            f"- Mean nearest-neighbour distance: {float(nearest.get('mean_distance', 0)):.3f} {nearest.get('distance_units', '')}".rstrip(),
            f"- Median nearest-neighbour distance: {float(nearest.get('median_distance', 0)):.3f} {nearest.get('distance_units', '')}".rstrip(),
        ]
        pair_summaries = nearest.get("pair_summaries", {})
        if pair_summaries:
            strongest = max(pair_summaries.items(), key=lambda item: item[1].get("count", 0))
            lines.append(f"- Most frequent nearest-neighbour pairing: {strongest[0]} ({strongest[1].get('count', 0)} observations)")
    else:
        lines.append("- Nearest-neighbour analysis was not supplied.")

    lines += ["", "7. Local ROI Findings", "---------------------"]
    if regions_data and regions_data.get("regions"):
        regions = sorted(regions_data["regions"], key=lambda row: row.get("cell_count", 0), reverse=True)
        lines.append(
            f"The slide was divided into {regions_data.get('region_count', len(regions))} retained "
            f"{regions_data.get('region_size', 'unknown')} {regions_data.get('distance_units', '')} ROIs."
        )
        for region in regions[:10]:
            dominant = max(region.get("class_percentages", {}).items(), key=lambda item: item[1], default=("unknown", 0))
            lines.append(
                f"- {region.get('region_id')}: {region.get('region_label')}; {region.get('cell_count', 0):,} cells; "
                f"dominant type {dominant[0]} ({dominant[1]:.1f}%); "
                f"{region.get('pairs_within_radius', 0):,} local pairs."
            )
        if len(regions) > 10:
            lines.append(f"- The 10 most populated regions are shown above; {len(regions) - 10} additional regions are available in the ROI files.")
        lines.append("Interpretation: differences between ROIs reveal local heterogeneity that a single whole-slide total can hide.")
    else:
        lines.append("- Local ROI analysis was not supplied.")

    lines += ["", "8. Per-Cell Neighbourhood Communities", "-------------------------------------"]
    if communities_data and communities_data.get("communities"):
        lines.append(
            f"Every cell was characterised within {communities_data.get('radius', 'the selected')} "
            f"{communities_data.get('distance_units', '')}, then grouped by neighbour composition."
        )
        for community in sorted(communities_data["communities"], key=lambda row: row.get("cell_count", 0), reverse=True):
            means = community.get("mean_neighbour_counts", {})
            dominant_neighbour = max(means.items(), key=lambda item: item[1], default=("unknown", 0))
            lines.append(
                f"- {community.get('community_id')}: {community.get('community_label')}; "
                f"{community.get('cell_count', 0):,} cells; mean {community.get('mean_total_neighbours', 0):.1f} neighbours; "
                f"largest mean neighbour group {dominant_neighbour[0]} ({dominant_neighbour[1]:.1f} per cell)."
            )
        lines.append("Interpretation: these communities summarize recurring local microenvironments; they are exploratory clusters, not validated biological classes.")
    else:
        lines.append("- Per-cell neighbourhood community analysis was not supplied.")

    lines += ["", "9. Ranked Regional Findings", "----------------------------"]
    if rankings_data:
        ranking_specs = [
            ("inflammatory", "inflammatory_percentage", "% inflammatory", "Most inflammatory"),
            ("neoplastic", "neoplastic_percentage", "% neoplastic", "Most neoplastic"),
            ("tumour_immune_interaction", "tumour_immune_pairs_per_100_cells", "pairs per 100 cells", "Strongest tumour-immune interaction"),
        ]
        for category, metric, unit, title in ranking_specs:
            rows = rankings_data.get("rankings", {}).get(category, [])
            if rows:
                lines.append(f"- {title}: " + ", ".join(
                    f"{row.get('region_id')} ({row.get(metric, 0):.2f} {unit})" for row in rows[:5]
                ))
        lines.append("Interpretation: rankings use normalized regional measurements so large ROIs do not automatically dominate interaction results.")
    else:
        lines.append("- Region rankings were not supplied.")

    lines += ["", "10. Slide-Level Pathology Summary", "---------------------------------"]
    if slide_summary_data:
        inflammatory = slide_summary_data.get("most_inflammatory_region", {})
        neoplastic = slide_summary_data.get("most_neoplastic_region", {})
        interactive = slide_summary_data.get("most_tumour_immune_interactive_region", {})
        lines += [
            f"- Total model-detected nuclei: {slide_summary_data.get('total_nuclei', total):,}",
            f"- Analysed local ROIs: {slide_summary_data.get('region_count', 'unknown')}",
            f"- Most inflammatory ROI: {inflammatory.get('region_id', 'unknown')}",
            f"- Most neoplastic ROI: {neoplastic.get('region_id', 'unknown')}",
            f"- Strongest tumour-immune interaction ROI: {interactive.get('region_id', 'unknown')}",
        ]
    else:
        lines.append("- A separate slide-level summary was not supplied.")

    lines += [
        "",
        "11. Interpretation Guidance",
        "--------------------------",
        "Cell proportions describe the analysed region and are sensitive to tissue sampling, detection errors, class confusion, probability filtering, and slide quality.",
        "Spatial counts and distances describe proximity, not biological interaction or causality. Comparisons across slides require consistent magnification, physical scaling, regions, filters, and analysis parameters.",
        "",
        "12. Recommended Supporting Outputs",
        "---------------------------------",
        "- TIAViz nucleus overlay / AnnotationStore for visual verification",
        "- KongNet nuclei CSV for detection-level audit",
        "- Cell-type co-occurrence matrix",
        "- Radius-neighbourhood and nearest-neighbour tables",
        "- Local ROI CSV/JSON with region-specific composition and spatial features",
        "- Per-cell neighbourhood CSV and spatial-community JSON",
        "- Ranked-region text/JSON outputs",
        "- Density, inflammatory, and tumour-immune TIAViz heatmap AnnotationStores",
        "",
        "13. Overall Conclusion",
        "---------------------",
        f"KongNet detected {total} nuclei, with {dominant_type} as the most frequent predicted type. The supplied spatial analyses should be interpreted alongside the overlay and detection-level data.",
        "Final caution: all findings are model-derived research outputs and are not clinical diagnoses.",
    ]
    report = "\n".join(lines)
    ensure_parent_dir(output_report_path)
    with open(output_report_path, "w", encoding="utf-8") as file:
        file.write(report)
    return report


def tool_generate_nucleus_instance_segmentation_report(
    annotationstore_path: str,
    output_report_path: Optional[str] = None,
    min_probability: float = 0.0,
) -> str:
    """Generate a plain-text interpretability report for nucleus instance segmentation outputs."""
    instances = _load_nucleus_instance_segmentation_instances(annotationstore_path, min_probability)
    if not instances:
        raise ValueError("The nucleus instance segmentation output contains no retained instances.")

    if not output_report_path:
        output_report_path = os.path.join(
            os.path.dirname(os.path.abspath(annotationstore_path)),
            "nucleus_instance_segmentation_report.txt",
        )
    else:
        report_root, report_extension = os.path.splitext(output_report_path)
        if report_extension.lower() != ".txt":
            output_report_path = f"{report_root if report_extension else output_report_path}.txt"

    counts = Counter(instance.get("type", "Unknown") or "Unknown" for instance in instances)
    inferred_model_name = _infer_nucleus_instance_segmentation_model_name_from_counts(counts)
    total = len(instances)
    dominant_type, dominant_count = counts.most_common(1)[0]
    probabilities = [
        float(instance["probability"])
        for instance in instances
        if isinstance(instance.get("probability"), (int, float))
    ]
    areas = [
        float(instance["area"])
        for instance in instances
        if isinstance(instance.get("area"), (int, float)) and float(instance["area"]) >= 0
    ]

    lines = [
        "Nucleus Instance Segmentation Report",
        "====================================",
        "",
        "1. Executive Summary",
        "--------------------",
        f"- Analysed output: {annotationstore_path}",
        f"- Inferred model/output family: {inferred_model_name}",
        f"- Total segmented nucleus instances: {total:,}",
        f"- Dominant predicted class: {dominant_type} ({dominant_count / total * 100:.2f}%)",
        "- These findings are model-derived research outputs and are not a clinical diagnosis.",
        "",
        "2. Output Context",
        "-----------------",
        "- Task: nucleus instance segmentation.",
        "- Meaning: the model predicts individual nucleus objects/boundaries and, where supported, class labels.",
        f"- Minimum probability filter: {float(min_probability):.2f}",
        "",
        "3. Predicted Instance-Class Composition",
        "---------------------------------------",
    ]
    for cell_type, count in counts.most_common():
        lines.append(f"- {cell_type}: {count:,} ({count / total * 100:.2f}%)")

    lines += ["", "4. Confidence Summary", "---------------------"]
    if probabilities:
        low_confidence = sum(value < 0.5 for value in probabilities)
        mean_probability = sum(probabilities) / len(probabilities)
        lines += [
            f"- Instances with probability values: {len(probabilities):,} of {total:,}",
            f"- Mean predicted probability: {mean_probability:.4f}",
            f"- Minimum / maximum probability: {min(probabilities):.4f} / {max(probabilities):.4f}",
            f"- Probability below 0.50: {low_confidence:,} ({low_confidence / len(probabilities) * 100:.2f}%)",
            "Interpretation: probabilities reflect model confidence, not confirmed correctness.",
        ]
    else:
        lines.append("- No usable probability values were present; confidence could not be summarised.")

    lines += ["", "5. Instance Geometry Summary", "----------------------------"]
    if areas:
        mean_area = sum(areas) / len(areas)
        lines += [
            f"- Instances with measurable geometry area: {len(areas):,} of {total:,}",
            f"- Mean instance area in AnnotationStore coordinate units: {mean_area:.2f}",
            f"- Minimum / maximum area: {min(areas):.2f} / {max(areas):.2f}",
            "Interpretation: area values are useful for quality control but depend on coordinate units and output geometry.",
        ]
    else:
        lines.append("- No usable instance geometry areas were available.")

    lines += [
        "",
        "6. Interpretation Guidance",
        "--------------------------",
        "- Class names can overlap across model families, so model selection should use task type, model name, tissue context, and full class set.",
        "- Instance segmentation boundaries should be visually checked in TIAViz against the original histology.",
        "- Counts and percentages describe model predictions in the analysed output, not ground-truth biology.",
        "- This report does not provide diagnosis, grading, prognosis, or treatment advice.",
        "",
        "7. Files",
        "--------",
        f"- Source output: {annotationstore_path}",
        f"- Saved report: {output_report_path}",
    ]

    report_text = "\n".join(lines)
    ensure_parent_dir(output_report_path)
    with open(output_report_path, "w", encoding="utf-8") as file:
        file.write(report_text)
    return report_text


def tool_run_kongnet_spatial_workflow(
    annotationstore_path: str,
    output_dir: str,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    min_probability: float = 0.0,
    neighbourhood_radius: float = 50.0,
    region_size: float = 500.0,
    min_cells_per_region: int = 10,
    community_count: int = 4,
    pathology_question: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """Run the complete KongNet spatial interpretability workflow."""
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty path.")
    if not os.path.exists(annotationstore_path):
        raise FileNotFoundError(f"KongNet AnnotationStore or nuclei CSV not found: {annotationstore_path}")
    if not 0.0 <= float(min_probability) <= 1.0:
        raise ValueError("min_probability must be between 0 and 1.")
    if neighbourhood_radius <= 0 or region_size <= 0:
        raise ValueError("neighbourhood_radius and region_size must be greater than 0.")
    if min_cells_per_region < 1 or community_count < 1:
        raise ValueError("min_cells_per_region and community_count must be at least 1.")

    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "nuclei_csv": os.path.join(output_dir, "kongnet_nuclei.csv"),
        "neighbourhood_csv": os.path.join(output_dir, "radius_neighbourhoods.csv"),
        "neighbourhood_json": os.path.join(output_dir, "radius_neighbourhoods.json"),
        "cooccurrence_csv": os.path.join(output_dir, "cell_type_cooccurrence.csv"),
        "cooccurrence_json": os.path.join(output_dir, "cell_type_cooccurrence.json"),
        "nearest_csv": os.path.join(output_dir, "nearest_neighbours.csv"),
        "nearest_json": os.path.join(output_dir, "nearest_neighbours.json"),
        "regions_csv": os.path.join(output_dir, "kongnet_regions.csv"),
        "regions_json": os.path.join(output_dir, "kongnet_regions.json"),
        "regions_db": os.path.join(output_dir, "kongnet_region_boundaries.db"),
        "cell_neighbourhoods_csv": os.path.join(output_dir, "kongnet_cell_neighbourhoods.csv"),
        "communities_json": os.path.join(output_dir, "kongnet_spatial_communities.json"),
        "rankings_json": os.path.join(output_dir, "kongnet_region_rankings.json"),
        "rankings_txt": os.path.join(output_dir, "kongnet_region_rankings.txt"),
        "slide_summary_json": os.path.join(output_dir, "kongnet_slide_summary.json"),
        "slide_summary_txt": os.path.join(output_dir, "kongnet_slide_summary.txt"),
        "heatmaps_manifest_json": os.path.join(output_dir, "kongnet_heatmaps_manifest.json"),
        "density_heatmap_db": os.path.join(output_dir, "kongnet_density_heatmap.db"),
        "inflammatory_heatmap_db": os.path.join(output_dir, "kongnet_inflammatory_heatmap.db"),
        "interaction_heatmap_db": os.path.join(output_dir, "kongnet_tumour_immune_interaction_heatmap.db"),
        "question_answer_txt": os.path.join(output_dir, "kongnet_spatial_question_answer.txt"),
        "report_txt": os.path.join(output_dir, "kongnet_ai_interpretability_report.txt"),
        "manifest_json": os.path.join(output_dir, "kongnet_spatial_workflow_manifest.json"),
    }
    if not overwrite:
        existing = [path for path in paths.values() if os.path.exists(path)]
        if existing:
            raise FileExistsError("Workflow outputs already exist: " + ", ".join(existing))

    spatial_kwargs = {"wsi_path": wsi_path, "mpp": mpp}
    step_results = {}
    step_results["export_nuclei"] = tool_export_kongnet_nuclei_to_csv(
        annotationstore_path=annotationstore_path,
        output_csv_path=paths["nuclei_csv"],
        min_probability=min_probability,
        **spatial_kwargs,
    )
    step_results["radius_neighbourhoods"] = tool_find_cells_within_radius(
        annotationstore_path=annotationstore_path,
        output_csv_path=paths["neighbourhood_csv"],
        output_json_path=paths["neighbourhood_json"],
        radius=neighbourhood_radius,
        min_probability=min_probability,
        **spatial_kwargs,
    )
    step_results["cooccurrence"] = tool_compute_cell_type_cooccurrence(
        annotationstore_path=annotationstore_path,
        output_json_path=paths["cooccurrence_json"],
        output_csv_path=paths["cooccurrence_csv"],
        radius=neighbourhood_radius,
        min_probability=min_probability,
        **spatial_kwargs,
    )
    step_results["nearest_neighbours"] = tool_compute_nearest_neighbour_features(
        annotationstore_path=annotationstore_path,
        output_csv_path=paths["nearest_csv"],
        output_json_path=paths["nearest_json"],
        min_probability=min_probability,
        **spatial_kwargs,
    )
    step_results["roi_analysis"] = tool_analyze_kongnet_regions(
        annotationstore_path=annotationstore_path,
        output_json_path=paths["regions_json"],
        output_csv_path=paths["regions_csv"],
        region_size=region_size,
        neighbourhood_radius=neighbourhood_radius,
        min_cells_per_region=min_cells_per_region,
        min_probability=min_probability,
        **spatial_kwargs,
    )
    step_results["roi_overlay"] = tool_export_kongnet_regions_to_annotationstore(
        regions_json_path=paths["regions_json"],
        output_db_path=paths["regions_db"],
        wsi_path=wsi_path,
        mpp=mpp,
        overwrite=overwrite,
    )
    step_results["region_rankings"] = tool_rank_kongnet_regions(
        regions_json_path=paths["regions_json"],
        output_json_path=paths["rankings_json"],
        output_txt_path=paths["rankings_txt"],
        top_k=5,
    )
    step_results["region_heatmaps"] = tool_generate_kongnet_region_heatmaps(
        regions_json_path=paths["regions_json"],
        output_dir=output_dir,
        wsi_path=wsi_path,
        mpp=mpp,
        overwrite=overwrite,
    )
    step_results["cell_neighbourhood_communities"] = tool_characterize_kongnet_cell_neighbourhoods(
        annotationstore_path=annotationstore_path,
        output_csv_path=paths["cell_neighbourhoods_csv"],
        output_json_path=paths["communities_json"],
        radius=neighbourhood_radius,
        min_probability=min_probability,
        community_count=community_count,
        **spatial_kwargs,
    )
    step_results["slide_summary"] = tool_generate_kongnet_slide_summary(
        nuclei_csv_path=paths["nuclei_csv"],
        regions_json_path=paths["regions_json"],
        output_txt_path=paths["slide_summary_txt"],
        output_json_path=paths["slide_summary_json"],
    )
    if pathology_question and pathology_question.strip():
        step_results["pathology_question_answer"] = tool_answer_kongnet_spatial_question(
            question=pathology_question,
            regions_json_path=paths["regions_json"],
            output_txt_path=paths["question_answer_txt"],
            top_k=5,
        )
    else:
        paths.pop("question_answer_txt")
    report = tool_generate_kongnet_ai_report(
        nuclei_csv_path=paths["nuclei_csv"],
        cooccurrence_json_path=paths["cooccurrence_json"],
        neighbourhood_json_path=paths["neighbourhood_json"],
        nearest_neighbour_json_path=paths["nearest_json"],
        regions_json_path=paths["regions_json"],
        communities_json_path=paths["communities_json"],
        rankings_json_path=paths["rankings_json"],
        slide_summary_json_path=paths["slide_summary_json"],
        output_report_path=paths["report_txt"],
    )
    step_results["interpretability_report"] = f"Saved plain-text report ({len(report)} characters)."

    manifest = {
        "workflow": "full_kongnet_spatial_workflow",
        "status": "completed",
        "annotationstore_path": annotationstore_path,
        "wsi_path": wsi_path,
        "parameters": {
            "mpp": mpp,
            "min_probability": min_probability,
            "neighbourhood_radius": neighbourhood_radius,
            "region_size": region_size,
            "min_cells_per_region": min_cells_per_region,
            "community_count": community_count,
            "pathology_question": pathology_question,
        },
        "outputs": paths,
        "completed_steps": list(step_results),
        "clinical_warning": "Model-derived research outputs; not a clinical diagnosis.",
    }
    _write_json(paths["manifest_json"], manifest)

    slides_dir = os.path.dirname(os.path.abspath(wsi_path)) if wsi_path else "<SLIDES_DIRECTORY>"
    tiaviz_command = f'tiatoolbox visualize --slides "{slides_dir}" --overlays "{os.path.abspath(output_dir)}"'
    return "\n".join([
        "Full KongNet spatial workflow completed successfully.",
        f"Completed steps: {len(step_results)}",
        f"Output directory: {os.path.abspath(output_dir)}",
        f"Text report: {paths['report_txt']}",
        f"ROI overlay: {paths['regions_db']}",
        f"Workflow manifest: {paths['manifest_json']}",
        "",
        "Open the generated ROI overlay in TIAViz with:",
        tiaviz_command,
        "Place or copy the nucleus AnnotationStore into the same output directory to view both overlay layers together.",
    ])
