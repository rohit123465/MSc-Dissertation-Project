"""
tia_tools.py
------------
MCP-only pathology agent tool logic.


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
from pathlib import Path
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


def tool_validate_qupath_roi_pair(
    image_path: str,
    geojson_path: str,
    output_json_path: str,
    feature_id: Optional[str] = None,
    dimension_tolerance_pixels: float = 2.0,
) -> str:
    """Validate that a QuPath GeoJSON annotation matches an exported ROI image."""
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError('validate_qupath_roi_pair requires a non-empty "image_path".')
    if not isinstance(geojson_path, str) or not geojson_path.strip():
        raise ValueError('validate_qupath_roi_pair requires a non-empty "geojson_path".')
    if not isinstance(output_json_path, str) or not output_json_path.strip():
        raise ValueError('validate_qupath_roi_pair requires a non-empty "output_json_path".')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"ROI image not found: {image_path}")
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"ROI GeoJSON not found: {geojson_path}")
    if float(dimension_tolerance_pixels) < 0:
        raise ValueError("dimension_tolerance_pixels must be non-negative.")

    from shapely import affinity
    from shapely.geometry import shape
    from tiatoolbox.wsicore.wsireader import WSIReader

    with open(geojson_path, "r", encoding="utf-8") as file:
        geojson = json.load(file)
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("QuPath ROI GeoJSON must be a FeatureCollection.")

    candidates = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        if properties.get("objectType") not in {None, "annotation"}:
            continue
        if feature_id and str(feature.get("id")) != str(feature_id):
            continue
        candidates.append(feature)

    if not candidates:
        requested = f' with id "{feature_id}"' if feature_id else ""
        raise ValueError(f"No polygon annotation found in GeoJSON{requested}.")
    if len(candidates) > 1 and not feature_id:
        ids = [str(feature.get("id", "<missing>")) for feature in candidates]
        raise ValueError(
            "GeoJSON contains multiple polygon annotations. Supply feature_id to select one. "
            f"Candidate IDs: {ids}"
        )

    feature = candidates[0]
    geometry = shape(feature["geometry"])
    if geometry.is_empty:
        raise ValueError("Selected ROI geometry is empty.")

    reader = WSIReader.open(image_path)
    image_width, image_height = (int(value) for value in reader.info.slide_dimensions)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("ROI image has invalid dimensions.")

    min_x, min_y, max_x, max_y = (float(value) for value in geometry.bounds)
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError("Selected ROI geometry has a zero-area bounding box.")

    scale_x = bbox_width / float(image_width)
    scale_y = bbox_height / float(image_height)
    inferred_downsample = (scale_x + scale_y) / 2.0
    expected_width = bbox_width / inferred_downsample
    expected_height = bbox_height / inferred_downsample
    width_error = abs(expected_width - image_width)
    height_error = abs(expected_height - image_height)
    anisotropy_fraction = abs(scale_x - scale_y) / max(scale_x, scale_y)

    local_geometry = affinity.translate(geometry, xoff=-min_x, yoff=-min_y)
    local_geometry = affinity.scale(
        local_geometry,
        xfact=1.0 / inferred_downsample,
        yfact=1.0 / inferred_downsample,
        origin=(0.0, 0.0),
    )
    local_min_x, local_min_y, local_max_x, local_max_y = (
        float(value) for value in local_geometry.bounds
    )
    tolerance = float(dimension_tolerance_pixels)
    polygon_inside_image = (
        local_min_x >= -tolerance
        and local_min_y >= -tolerance
        and local_max_x <= image_width + tolerance
        and local_max_y <= image_height + tolerance
    )

    def finite_pair(value: Any) -> Optional[List[float]]:
        try:
            values = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        if len(values) < 2 or not all(math.isfinite(item) for item in values[:2]):
            return None
        return values[:2]

    mpp = finite_pair(getattr(reader.info, "mpp", None))
    objective_power = getattr(reader.info, "objective_power", None)
    try:
        objective_power = float(objective_power) if objective_power is not None else None
    except (TypeError, ValueError):
        objective_power = None

    local_area_pixels = float(local_geometry.area)
    crop_area_pixels = float(image_width * image_height)
    roi_fraction = local_area_pixels / crop_area_pixels if crop_area_pixels else None
    roi_area_mm2 = (
        local_area_pixels * mpp[0] * mpp[1] / 1_000_000.0
        if mpp is not None
        else None
    )

    errors = []
    warnings = []
    if not geometry.is_valid:
        errors.append("GeoJSON polygon geometry is invalid.")
    if anisotropy_fraction > 0.01:
        errors.append(
            "GeoJSON and image imply different X/Y scale factors; they may not be the same ROI export."
        )
    if width_error > tolerance or height_error > tolerance:
        errors.append(
            "GeoJSON bounding box does not match image dimensions within the requested tolerance."
        )
    if not polygon_inside_image:
        errors.append("Localised ROI polygon falls outside the exported image bounds.")
    if mpp is None:
        warnings.append("Image MPP is unavailable; physical ROI area could not be calculated.")
    if getattr(reader.info, "level_count", 1) == 1:
        warnings.append("ROI image has one resolution level; this is normal for a small crop.")
    if roi_fraction is not None and roi_fraction < 0.5:
        warnings.append(
            "Less than half of the rectangular crop lies inside the ROI polygon; polygon filtering is important."
        )

    status = "failed" if errors else ("passed_with_warnings" if warnings else "passed")
    manifest = {
        "validation": "qupath_roi_pair",
        "status": status,
        "image": {
            "path": os.path.abspath(image_path),
            "reader": type(reader).__name__,
            "width": image_width,
            "height": image_height,
            "level_count": int(getattr(reader.info, "level_count", 1)),
            "level_dimensions": [
                [int(item[0]), int(item[1])]
                for item in getattr(reader.info, "level_dimensions", [])
            ],
            "mpp": mpp,
            "objective_power": objective_power,
        },
        "geojson": {
            "path": os.path.abspath(geojson_path),
            "feature_count": len(geojson.get("features", [])),
            "selected_feature_id": feature.get("id"),
            "properties": feature.get("properties") or {},
            "geometry_type": geometry.geom_type,
            "geometry_valid": bool(geometry.is_valid),
            "geometry_validity_reason": (
                "Valid Geometry"
                if geometry.is_valid
                else __import__("shapely.validation", fromlist=["explain_validity"]).explain_validity(geometry)
            ),
            "wsi_bounds": {
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
                "width": bbox_width,
                "height": bbox_height,
            },
        },
        "coordinate_transform": {
            "wsi_origin_x": min_x,
            "wsi_origin_y": min_y,
            "inferred_downsample": inferred_downsample,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "formula_wsi_to_local": (
                "local_x=(wsi_x-wsi_origin_x)/inferred_downsample; "
                "local_y=(wsi_y-wsi_origin_y)/inferred_downsample"
            ),
            "formula_local_to_wsi": (
                "wsi_x=local_x*inferred_downsample+wsi_origin_x; "
                "wsi_y=local_y*inferred_downsample+wsi_origin_y"
            ),
            "local_bounds": {
                "min_x": local_min_x,
                "min_y": local_min_y,
                "max_x": local_max_x,
                "max_y": local_max_y,
            },
        },
        "dimension_check": {
            "tolerance_pixels": tolerance,
            "expected_width_from_geojson": expected_width,
            "expected_height_from_geojson": expected_height,
            "width_error_pixels": width_error,
            "height_error_pixels": height_error,
            "anisotropy_fraction": anisotropy_fraction,
            "polygon_inside_image": polygon_inside_image,
        },
        "area": {
            "roi_area_local_pixels_squared": local_area_pixels,
            "crop_area_pixels_squared": crop_area_pixels,
            "roi_fraction_of_crop": roi_fraction,
            "roi_area_mm_squared": roi_area_mm2,
        },
        "errors": errors,
        "warnings": warnings,
    }
    _write_json(output_json_path, manifest)

    return "\n".join([
        "QuPath ROI pair validation completed.",
        f"Status: {status}",
        f"Image dimensions: {image_width} x {image_height} px",
        f"GeoJSON bounding box: {bbox_width:g} x {bbox_height:g} WSI pixels",
        f"Inferred downsample: {inferred_downsample:.6g}",
        f"WSI origin: ({min_x:g}, {min_y:g})",
        f"ROI fraction of crop: {roi_fraction:.6f}",
        (
            f"ROI area: {roi_area_mm2:.6f} mm^2"
            if roi_area_mm2 is not None
            else "ROI area: unavailable in physical units"
        ),
        f"Errors: {len(errors)}",
        f"Warnings: {len(warnings)}",
        f"Manifest: {output_json_path}",
    ])


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
        # MultiTaskSegmentor.save_predictions expects class dictionaries to be
        # keyed by task name, even when NucleusInstanceSegmentor has only one
        # task. Passing the flat {type_id: label} mapping causes a late
        # KeyError after inference finishes. The engine initializes ``tasks``
        # lazily during run(), so it may still be empty here; the canonical
        # task produced by NucleusInstanceSegmentor is nuclei_segmentation.
        run_kwargs["class_dict"] = {"nuclei_segmentation": class_dict}
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
        # TIAToolbox 2.1.x WSIPatchDataset accepts a string path or WSIReader.
        # pathlib.Path passes our existence check but is rejected by that dataset.
        images=[os.path.abspath(wsi_path)],
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


def _spatial_image_bounds(
    wsi_path: Optional[str],
    scale: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """Return level-0 image bounds expressed in the requested spatial units."""
    if not wsi_path:
        return None
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    from tiatoolbox.wsicore.wsireader import WSIReader

    reader = WSIReader.open(wsi_path)
    width_px, height_px = (int(value) for value in reader.info.slide_dimensions)
    return {
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": float(width_px) * float(scale["x_scale"]),
        "y_max": float(height_px) * float(scale["y_scale"]),
        "width_px": width_px,
        "height_px": height_px,
    }


def _slide_stem_prefix(wsi_path: Optional[str]) -> str:
    """Return the TIAViz slide-matching filename prefix."""
    return f"{Path(wsi_path).stem}_" if wsi_path else ""


def _resolve_dynamic_region_size(
    requested_region_size: Optional[float],
    nuclei_count: int,
    min_cells_per_region: int,
    image_bounds: Optional[Dict[str, float]],
    coords: Any,
    units: str,
) -> tuple[float, Dict[str, Any]]:
    """Resolve an explicit or data-adaptive square grid-cell size."""
    import numpy as np

    if requested_region_size is not None:
        resolved = float(requested_region_size)
        if resolved <= 0:
            raise ValueError("region_size must be greater than 0 or omitted for automatic sizing.")
        return resolved, {
            "mode": "explicit",
            "requested_region_size": resolved,
            "resolved_region_size": resolved,
        }

    if image_bounds:
        width = float(image_bounds["x_max"] - image_bounds["x_min"])
        height = float(image_bounds["y_max"] - image_bounds["y_min"])
    else:
        width = float(np.ptp(coords[:, 0]))
        height = float(np.ptp(coords[:, 1]))
    if width <= 0 or height <= 0:
        raise ValueError("Cannot infer an automatic region size from zero-area spatial bounds.")

    target_cells = max(20, int(min_cells_per_region) * 3)
    target_regions = min(36, max(4, int(math.ceil(nuclei_count / target_cells))))
    raw_size = math.sqrt((width * height) / target_regions)

    if units == "microns":
        rounding_step = 25.0
        minimum_size = min(25.0, min(width, height))
        maximum_size = min(250.0, min(width, height) / 2.0)
    else:
        rounding_step = 32.0
        minimum_size = min(32.0, min(width, height))
        maximum_size = min(1024.0, min(width, height) / 2.0)
    maximum_size = max(minimum_size, maximum_size)
    clamped_size = min(max(raw_size, minimum_size), maximum_size)
    rounded_size = round(clamped_size / rounding_step) * rounding_step
    resolved = min(max(rounded_size, minimum_size), maximum_size)

    return float(resolved), {
        "mode": "automatic",
        "requested_region_size": None,
        "resolved_region_size": float(resolved),
        "spatial_width": width,
        "spatial_height": height,
        "nuclei_count": int(nuclei_count),
        "target_cells_per_region": target_cells,
        "target_candidate_regions": target_regions,
        "unrounded_region_size": raw_size,
        "rounding_step": rounding_step,
        "constraint": "4-36 candidate regions, at least two cells across the shorter image dimension",
    }


def _load_kongnet_nuclei(
    annotationstore_path: str,
    cell_types: Optional[List[str]] = None,
    min_probability: float = 0.0, #means nucleus with low confidence predictions are also selected. 
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
    region_size: Optional[float] = 100.0,
    neighbourhood_radius: float = 50.0,
    distance_units: str = "microns",
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    min_cells_per_region: int = 1,
    min_probability: float = 0.0,
) -> str:
    """Compute composition and spatial features independently in fixed local ROIs."""
    import numpy as np
    from scipy.spatial import cKDTree

    if neighbourhood_radius <= 0:
        raise ValueError("neighbourhood_radius must be greater than 0.")
    if min_cells_per_region < 1:
        raise ValueError("min_cells_per_region must be at least 1.")
    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    nuclei = _load_kongnet_nuclei(annotationstore_path, None, min_probability)
    if not nuclei:
        raise RuntimeError("No nuclei matched the requested probability threshold.")
    coords = _nucleus_coordinates(nuclei, scale)
    image_bounds = _spatial_image_bounds(wsi_path, scale)
    resolved_region_size, region_size_strategy = _resolve_dynamic_region_size(
        region_size,
        len(nuclei),
        min_cells_per_region,
        image_bounds,
        coords,
        scale["units"],
    )
    origin = np.array(
        [image_bounds["x_min"], image_bounds["y_min"]]
        if image_bounds
        else coords.min(axis=0),
        dtype=float,
    )
    bins = np.floor((coords - origin) / resolved_region_size).astype(int)
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
        x_min = float(origin[0] + grid_x * resolved_region_size)
        y_min = float(origin[1] + grid_y * resolved_region_size)
        x_max = x_min + resolved_region_size
        y_max = y_min + resolved_region_size
        if image_bounds:
            x_min = max(x_min, image_bounds["x_min"])
            y_min = max(y_min, image_bounds["y_min"])
            x_max = min(x_max, image_bounds["x_max"])
            y_max = min(y_max, image_bounds["y_max"])
        area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        if area <= 0:
            continue
        region = {
            "region_id": f"R{len(regions) + 1}",
            "region_label": _kongnet_region_label(class_counts),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "area_square_units": area,
            "is_boundary_region": (
                (x_max - x_min) < resolved_region_size
                or (y_max - y_min) < resolved_region_size
            ),
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
        "method": "adaptive non-overlapping grid ROIs" if region_size is None else "fixed non-overlapping grid ROIs",
        "region_size": resolved_region_size,
        "region_size_strategy": region_size_strategy,
        "neighbourhood_radius": float(neighbourhood_radius),
        "distance_units": scale["units"],
        "scale": scale,
        "image_bounds": image_bounds,
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
            "distance_units", "area_square_units", "is_boundary_region", "cell_count",
            "cell_density_per_square_unit", "pairs_within_radius",
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
        f"Region size: {resolved_region_size:g} {scale['units']} ({region_size_strategy['mode']})",
        f"Local pair radius: {neighbourhood_radius} {scale['units']}",
        f"JSON: {output_json_path}",
        f"CSV: {output_csv_path}" if output_csv_path else "CSV: not requested",
    ])


def _pointpats_availability() -> Dict[str, Any]:
    try:
        import pointpats  # type: ignore

        return {
            "available": True,
            "version": getattr(pointpats, "__version__", "unknown"),
            "note": "pointpats is installed; fallback NumPy/SciPy statistics are also reported for transparent MCP output.",
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "note": (
                "pointpats is not installed in this Python environment, so the tool used "
                "transparent NumPy/SciPy point-pattern equivalents."
            ),
            "import_error": f"{type(exc).__name__}: {exc}",
        }


def _point_pattern_label(value: Optional[float], clustered_cutoff: float = 0.9, dispersed_cutoff: float = 1.1) -> str:
    if value is None:
        return "insufficient points"
    if value < clustered_cutoff:
        return "clustered"
    if value > dispersed_cutoff:
        return "dispersed"
    return "approximately random"


def _vmr_label(value: Optional[float]) -> str:
    if value is None:
        return "insufficient points"
    if value > 1.5:
        return "clustered/heterogeneous"
    if value < 0.75:
        return "regular/even"
    return "approximately random"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _roi_moran_feature_value(region: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Extract a numeric ROI-level variable for Moran's I analysis."""
    metric = str(metric_name or "").strip()
    if not metric:
        return None

    direct_metrics = {
        "cell_count",
        "pairs_within_radius",
        "mean_nearest_neighbour_distance",
        "area_square_units",
    }
    if metric in direct_metrics:
        return _safe_float(region.get(metric))
    if metric.endswith("_percentage"):
        class_name = metric[: -len("_percentage")]
        return _safe_float((region.get("class_percentages") or {}).get(class_name))
    if metric.endswith("_count"):
        class_name = metric[: -len("_count")]
        return _safe_float((region.get("class_counts") or {}).get(class_name))
    if metric == "cell_density":
        return _safe_float(region.get("cell_density_per_square_unit"))
    if metric == "interaction_strength":
        cell_count = _safe_float(region.get("cell_count")) or 0.0
        if cell_count <= 0:
            return None
        pair_counts = region.get("pair_counts") or {}
        interaction_pairs = 0.0
        for pair_name, count in pair_counts.items():
            parts = set(str(pair_name).split("--"))
            is_immune = bool(parts & set(KONGNET_IMMUNE_CELL_TYPES))
            is_tumour_or_epithelial = bool(parts & {"Neoplastic", "Epithelial", "Tumour_Cell", "Tumor_Cell"})
            if is_immune and is_tumour_or_epithelial:
                interaction_pairs += float(count or 0)
        return interaction_pairs / cell_count * 100.0

    return _safe_float(region.get(metric))


def _default_moran_roi_metrics(regions: List[Dict[str, Any]]) -> List[str]:
    class_names = []
    for region in regions:
        for name in (region.get("class_counts") or {}).keys():
            if name not in class_names:
                class_names.append(str(name))

    metrics = ["cell_count", "cell_density", "pairs_within_radius", "interaction_strength"]
    for name in class_names:
        metrics.append(f"{name}_percentage")
    return metrics


def _build_roi_pysal_weights(
    regions: List[Dict[str, Any]],
    weights_method: str,
    k_neighbours: int,
    distance_threshold: Optional[float],
):
    """Build ROI neighbour weights using libpysal."""
    import numpy as np
    from libpysal.weights import DistanceBand, KNN, W

    method = str(weights_method or "queen").strip().lower()
    centroids = np.asarray(
        [
            [
                (float(region["x_min"]) + float(region["x_max"])) / 2.0,
                (float(region["y_min"]) + float(region["y_max"])) / 2.0,
            ]
            for region in regions
        ],
        dtype=float,
    )
    ids = [str(region.get("region_id") or index) for index, region in enumerate(regions)]

    if method == "knn":
        k = min(max(1, int(k_neighbours)), max(1, len(regions) - 1))
        weights = KNN.from_array(centroids, k=k, ids=ids)
        return weights, {"method": "libpysal.weights.KNN.from_array", "k_neighbours": k}

    if method == "distance":
        if distance_threshold is None:
            widths = [float(region["x_max"]) - float(region["x_min"]) for region in regions]
            heights = [float(region["y_max"]) - float(region["y_min"]) for region in regions]
            typical_size = max(float(np.median(widths + heights)), 1e-9)
            distance_threshold = typical_size * 1.01
        weights = DistanceBand.from_array(
            centroids,
            threshold=float(distance_threshold),
            binary=True,
            silence_warnings=True,
            ids=ids,
        )
        return weights, {
            "method": "libpysal.weights.DistanceBand.from_array",
            "distance_threshold": float(distance_threshold),
        }

    if method not in {"queen", "rook"}:
        raise ValueError("weights_method must be one of: queen, rook, distance, knn.")

    neighbours: Dict[str, List[str]] = {region_id: [] for region_id in ids}
    weights_by_id: Dict[str, List[float]] = {region_id: [] for region_id in ids}
    for i, region_i in enumerate(regions):
        gx_i = region_i.get("grid_x")
        gy_i = region_i.get("grid_y")
        if gx_i is None or gy_i is None:
            raise ValueError(
                "queen/rook weights require grid_x and grid_y in the regions JSON. "
                "Use weights_method='distance' or 'knn' for non-grid regions."
            )
        for j, region_j in enumerate(regions):
            if i == j:
                continue
            dx = abs(int(region_j.get("grid_x")) - int(gx_i))
            dy = abs(int(region_j.get("grid_y")) - int(gy_i))
            is_neighbour = (dx + dy == 1) if method == "rook" else (max(dx, dy) == 1)
            if is_neighbour:
                neighbours[ids[i]].append(ids[j])
                weights_by_id[ids[i]].append(1.0)

    weights = W(neighbours, weights_by_id, ids=ids, silence_warnings=True)
    return weights, {"method": "libpysal.weights.W", "contiguity": method}


def _moran_interpretation(moran_i: Optional[float], p_value: Optional[float], alpha: float) -> str:
    if moran_i is None or p_value is None:
        return "Moran's I could not be computed for this variable."
    if p_value >= alpha:
        return "No strong evidence of ROI-level spatial autocorrelation."
    if moran_i > 0:
        return "Significant positive spatial autocorrelation: similar ROI values cluster together as hotspots/coldspots."
    if moran_i < 0:
        return "Significant negative spatial autocorrelation: high-value ROIs tend to neighbour low-value ROIs."
    return "Statistically significant but near-zero spatial autocorrelation."


def _entropy_label(value: Optional[float], low_threshold: float, high_threshold: float) -> str:
    if value is None:
        return "unavailable"
    if value < low_threshold:
        return "low diversity / homogeneous"
    if value < high_threshold:
        return "moderate diversity"
    return "high diversity / mixed microenvironment"


def _dominant_region_class(class_counts: Dict[str, Any]) -> Optional[str]:
    numeric_counts = {
        str(name): float(count or 0)
        for name, count in (class_counts or {}).items()
        if _safe_float(count) is not None
    }
    if not numeric_counts:
        return None
    return max(numeric_counts, key=numeric_counts.get)


def _compute_shannon_entropy_from_counts(
    class_counts: Dict[str, Any],
    normalize: bool = True,
    entropy_base: float = math.e,
) -> tuple[Optional[float], Optional[float], int, int]:
    counts = [
        float(count)
        for count in (class_counts or {}).values()
        if _safe_float(count) is not None and float(count) > 0
    ]
    total = sum(counts)
    if total <= 0:
        return None, None, 0, 0

    proportions = [count / total for count in counts]
    raw_entropy = -sum(p * math.log(p, entropy_base) for p in proportions if p > 0)
    present_class_count = len(proportions)
    if not normalize:
        return float(raw_entropy), None, present_class_count, int(total)
    if present_class_count <= 1:
        return 0.0, float(raw_entropy), present_class_count, int(total)
    max_entropy = math.log(present_class_count, entropy_base)
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    return float(normalized_entropy), float(raw_entropy), present_class_count, int(total)


def tool_compute_kongnet_spatial_entropy(
    regions_json_path: str,
    output_json_path: str,
    output_txt_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    normalize: bool = True,
    entropy_base: float = math.e,
    low_threshold: float = 0.40,
    high_threshold: float = 0.70,
    cell_types: Optional[List[str]] = None,
) -> str:
    """Compute Shannon spatial entropy for each KongNet ROI.

    Entropy is computed from the model-predicted class composition inside each
    ROI. Low entropy means one cell class dominates the region. High entropy
    means the ROI contains a more mixed local microenvironment.
    """
    import numpy as np

    if not isinstance(regions_json_path, str) or not regions_json_path.strip():
        raise ValueError('compute_kongnet_spatial_entropy requires a non-empty "regions_json_path".')
    if not os.path.exists(regions_json_path):
        raise FileNotFoundError(f"Regions JSON not found: {regions_json_path}")
    if not isinstance(output_json_path, str) or not output_json_path.strip():
        raise ValueError('compute_kongnet_spatial_entropy requires a non-empty "output_json_path".')
    if float(entropy_base) <= 0.0 or math.isclose(float(entropy_base), 1.0):
        raise ValueError("entropy_base must be positive and not equal to 1.")
    if not 0.0 <= float(low_threshold) <= float(high_threshold) <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= low_threshold <= high_threshold <= 1.")

    with open(regions_json_path, "r", encoding="utf-8") as file:
        regions_payload = json.load(file)
    regions = list(regions_payload.get("regions") or [])
    if not regions:
        raise ValueError("No regions found in the regions JSON.")

    available_cell_types = []
    for region in regions:
        for name in (region.get("class_counts") or {}).keys():
            if name not in available_cell_types:
                available_cell_types.append(name)
    selected_cell_types = None
    if cell_types:
        available_lookup = {str(name).casefold(): str(name) for name in available_cell_types}
        selected_cell_types = []
        unresolved = []
        for requested in cell_types:
            match = available_lookup.get(str(requested).strip().casefold())
            if match:
                if match not in selected_cell_types:
                    selected_cell_types.append(match)
            else:
                unresolved.append(requested)
        if unresolved:
            try:
                expanded = _normalise_kongnet_cell_types(unresolved) or []
            except ValueError:
                expanded = []
            selected_cell_types.extend(
                name for name in expanded
                if name in available_cell_types and name not in selected_cell_types
            )
    if cell_types and not selected_cell_types:
        raise ValueError("cell_types did not contain any recognized KongNet cell classes.")

    entropy_rows = []
    for region in regions:
        all_class_counts = region.get("class_counts") or {}
        class_counts = (
            {name: all_class_counts.get(name, 0) for name in selected_cell_types}
            if selected_cell_types
            else all_class_counts
        )
        entropy, raw_entropy, present_class_count, total_cells = _compute_shannon_entropy_from_counts(
            class_counts,
            normalize=bool(normalize),
            entropy_base=float(entropy_base),
        )
        dominant_class = _dominant_region_class(class_counts)
        dominant_count = _safe_float(class_counts.get(dominant_class)) if dominant_class else None
        dominant_percentage = (
            dominant_count / total_cells * 100.0
            if dominant_count is not None and total_cells > 0
            else None
        )
        entropy_rows.append({
            "region_id": region.get("region_id"),
            "region_label": region.get("region_label"),
            "grid_x": region.get("grid_x"),
            "grid_y": region.get("grid_y"),
            "x_min": region.get("x_min"),
            "y_min": region.get("y_min"),
            "x_max": region.get("x_max"),
            "y_max": region.get("y_max"),
            "cell_count": int(total_cells),
            "dominant_class": dominant_class,
            "dominant_percentage": dominant_percentage,
            "present_class_count": int(present_class_count),
            "shannon_entropy": entropy,
            "raw_shannon_entropy": raw_entropy,
            "entropy_label": _entropy_label(entropy, float(low_threshold), float(high_threshold)),
            "class_counts": class_counts,
            "class_percentages": region.get("class_percentages") or {},
        })

    computed_values = [
        row["shannon_entropy"]
        for row in entropy_rows
        if row.get("shannon_entropy") is not None
    ]
    highest_entropy = sorted(
        [row for row in entropy_rows if row.get("shannon_entropy") is not None],
        key=lambda row: row["shannon_entropy"],
        reverse=True,
    )
    lowest_entropy = sorted(
        [row for row in entropy_rows if row.get("shannon_entropy") is not None],
        key=lambda row: row["shannon_entropy"],
    )
    payload = {
        "analysis": "kongnet_spatial_entropy",
        "regions_json_path": regions_json_path,
        "method": {
            "metric": "Shannon entropy",
            "formula": "H = -sum(p_i * log(p_i))",
            "normalization": "H / log(number of present cell classes)" if normalize else "raw Shannon entropy",
            "meaning": (
                "Low entropy indicates that one predicted cell type dominates an ROI. "
                "High entropy indicates a compositionally mixed ROI, such as a potential "
                "tumour-immune-stromal microenvironment."
            ),
        },
        "parameters": {
            "normalize": bool(normalize),
            "entropy_base": float(entropy_base),
            "low_threshold": float(low_threshold),
            "high_threshold": float(high_threshold),
            "cell_types": selected_cell_types or available_cell_types,
        },
        "region_count": len(regions),
        "computed_region_count": len(computed_values),
        "summary": {
            "mean_entropy": float(np.mean(computed_values)) if computed_values else None,
            "median_entropy": float(np.median(computed_values)) if computed_values else None,
            "minimum_entropy": float(np.min(computed_values)) if computed_values else None,
            "maximum_entropy": float(np.max(computed_values)) if computed_values else None,
            "highest_entropy_regions": highest_entropy[:5],
            "lowest_entropy_regions": lowest_entropy[:5],
        },
        "regions": entropy_rows,
        "interpretation_warning": (
            "Spatial entropy describes ROI composition only. It does not prove cell-cell interaction, "
            "validate model predictions, or provide a clinical diagnosis."
        ),
    }
    _write_json(output_json_path, payload)

    if output_csv_path:
        ensure_parent_dir(output_csv_path)
        fields = [
            "region_id",
            "region_label",
            "grid_x",
            "grid_y",
            "cell_count",
            "dominant_class",
            "dominant_percentage",
            "present_class_count",
            "shannon_entropy",
            "raw_shannon_entropy",
            "entropy_label",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ]
        with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in entropy_rows:
                writer.writerow({field: row.get(field) for field in fields})

    if output_txt_path:
        ensure_parent_dir(output_txt_path)
        lines = [
            "KongNet ROI Spatial Entropy Report",
            "==================================",
            "",
            f"Regions analysed: {len(regions)}",
            f"Entropy metric: {'normalized Shannon entropy' if normalize else 'raw Shannon entropy'}",
            f"Interpretation thresholds: low < {float(low_threshold):.2f}; moderate {float(low_threshold):.2f}-{float(high_threshold):.2f}; high >= {float(high_threshold):.2f}",
            "",
            "What this measures",
            "------------------",
            "Spatial entropy measures how mixed the predicted cell-type composition is inside each ROI.",
            "Low entropy means one cell type dominates the ROI. High entropy means several cell types are present in a more balanced mixture.",
            "",
            "Summary",
            "-------",
        ]
        if computed_values:
            lines.extend([
                f"Mean entropy: {payload['summary']['mean_entropy']:.3f}",
                f"Median entropy: {payload['summary']['median_entropy']:.3f}",
                f"Minimum entropy: {payload['summary']['minimum_entropy']:.3f}",
                f"Maximum entropy: {payload['summary']['maximum_entropy']:.3f}",
            ])
        else:
            lines.append("No entropy values could be computed.")

        lines.extend(["", "Highest entropy regions (most mixed)", "------------------------------------"])
        for index, row in enumerate(highest_entropy[:5], start=1):
            lines.append(
                f"{index}. {row.get('region_id')}: entropy {row.get('shannon_entropy'):.3f}; "
                f"{row.get('entropy_label')}; dominant class {row.get('dominant_class')} "
                f"({(row.get('dominant_percentage') or 0):.1f}%)."
            )

        lines.extend(["", "Lowest entropy regions (most homogeneous)", "------------------------------------------"])
        for index, row in enumerate(lowest_entropy[:5], start=1):
            lines.append(
                f"{index}. {row.get('region_id')}: entropy {row.get('shannon_entropy'):.3f}; "
                f"{row.get('entropy_label')}; dominant class {row.get('dominant_class')} "
                f"({(row.get('dominant_percentage') or 0):.1f}%)."
            )

        lines.extend([
            "",
            "Interpretation note",
            "-------------------",
            payload["interpretation_warning"],
        ])
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    return "\n".join([
        "KongNet spatial entropy analysis completed.",
        f"Regions analysed: {len(regions)}",
        f"Entropy values computed: {len(computed_values)}",
        f"JSON: {output_json_path}",
        f"TXT: {output_txt_path}" if output_txt_path else "TXT: not requested",
        f"CSV: {output_csv_path}" if output_csv_path else "CSV: not requested",
    ])


def _default_cross_g_types(class_names: List[str]) -> tuple[List[str], List[str], str]:
    source_groups = [
        ["Neoplastic"],
        ["Tumour_Cell"],
        ["Tumor_Cell"],
        ["Epithelial"],
        ["Epithelial_Cell"],
        ["Non-Neoplastic Epithelial"],
    ]
    source_types = []
    for group in source_groups:
        source_types = [name for name in group if name in class_names]
        if source_types:
            break
    if not source_types and class_names:
        source_types = [class_names[0]]
    target_types = [name for name in KONGNET_IMMUNE_CELL_TYPES if name in class_names]
    note = (
        "Default source uses neoplastic/tumour classes when available, otherwise epithelial classes. "
        "Default target uses immune/inflammatory classes present in the selected KongNet output."
    )
    return source_types, target_types, note


def _cross_g_curve(
    source_coords,
    source_ids: List[str],
    target_coords,
    target_ids: List[str],
    observation_area: float,
    radii: List[float],
) -> Dict[str, Any]:
    import numpy as np
    from scipy.spatial import cKDTree

    source_count = int(len(source_coords))
    target_count = int(len(target_coords))
    if source_count < 1 or target_count < 1:
        return {
            "status": "skipped",
            "reason": "At least one source and one target cell are required.",
            "source_count": source_count,
            "target_count": target_count,
        }

    tree = cKDTree(target_coords)
    k = min(target_count, 2 if set(source_ids) & set(target_ids) else 1)
    distances_raw, indices_raw = tree.query(source_coords, k=k)
    distances_2d = np.atleast_2d(distances_raw)
    indices_2d = np.atleast_2d(indices_raw)
    if distances_2d.shape[0] != source_count:
        distances_2d = distances_2d.T
        indices_2d = indices_2d.T

    nearest_distances = []
    for row_distances, row_indices, source_id in zip(distances_2d, indices_2d, source_ids, strict=False):
        selected_distance = None
        for distance, target_index in zip(row_distances, row_indices, strict=False):
            if int(target_index) >= target_count:
                continue
            if target_ids[int(target_index)] == source_id:
                continue
            selected_distance = float(distance)
            break
        if selected_distance is not None:
            nearest_distances.append(selected_distance)

    if not nearest_distances:
        return {
            "status": "skipped",
            "reason": "No non-self source-to-target nearest-neighbour distances could be computed.",
            "source_count": source_count,
            "target_count": target_count,
        }

    nearest = np.asarray(nearest_distances, dtype=float)
    area = float(max(observation_area, 1e-9))
    target_density = target_count / area
    curve = []
    for radius in radii:
        empirical = float(np.mean(nearest <= float(radius)))
        theoretical = float(1.0 - math.exp(-target_density * math.pi * float(radius) ** 2))
        curve.append({
            "radius": float(radius),
            "empirical_cross_g": empirical,
            "csr_poisson_expected": theoretical,
            "difference_from_expected": empirical - theoretical,
            "interpretation": (
                "above random expectation / spatial proximity"
                if empirical > theoretical
                else "below random expectation / spatial separation"
                if empirical < theoretical
                else "near random expectation"
            ),
        })

    return {
        "status": "computed",
        "source_count": source_count,
        "target_count": target_count,
        "source_count_with_target_distance": int(len(nearest)),
        "target_density_per_square_unit": float(target_density),
        "mean_nearest_target_distance": float(np.mean(nearest)),
        "median_nearest_target_distance": float(np.median(nearest)),
        "minimum_nearest_target_distance": float(np.min(nearest)),
        "maximum_nearest_target_distance": float(np.max(nearest)),
        "curve": curve,
    }


def tool_compute_kongnet_cross_g_function(
    annotationstore_path: str,
    output_json_path: str,
    output_txt_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    regions_json_path: Optional[str] = None,
    source_types: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
    radii: Optional[List[float]] = None,
    distance_units: str = "microns",
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    min_probability: float = 0.0,
) -> str:
    """Compute empirical cross-G functions between KongNet cell classes.

    Cross-G reports the probability that a source cell has at least one target
    cell within radius r. This is useful for tumour-immune proximity and
    immune-exclusion style questions.
    """
    import numpy as np

    if not isinstance(annotationstore_path, str) or not annotationstore_path.strip():
        raise ValueError('compute_kongnet_cross_g_function requires a non-empty "annotationstore_path".')
    if not isinstance(output_json_path, str) or not output_json_path.strip():
        raise ValueError('compute_kongnet_cross_g_function requires a non-empty "output_json_path".')
    radii = [float(value) for value in (radii or [25.0, 50.0, 100.0]) if float(value) > 0]
    if not radii:
        raise ValueError("At least one positive radius is required.")

    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    nuclei = _load_kongnet_nuclei(annotationstore_path, None, min_probability)
    if not nuclei:
        raise RuntimeError("No nuclei matched the requested probability threshold.")

    coords = _nucleus_coordinates(nuclei, scale)
    classes = np.asarray([cell["type"] for cell in nuclei], dtype=object)
    class_names = _ordered_kongnet_class_names(nuclei)
    default_source_types, default_target_types, default_note = _default_cross_g_types(class_names)
    resolved_source_types = _normalise_kongnet_cell_types(source_types) if source_types else default_source_types
    resolved_target_types = _normalise_kongnet_cell_types(target_types) if target_types else default_target_types
    if not resolved_source_types:
        raise ValueError("No source_types were available. Provide explicit source_types.")
    if not resolved_target_types:
        raise ValueError("No target_types were available. Provide explicit target_types.")

    source_mask = np.isin(classes, resolved_source_types)
    target_mask = np.isin(classes, resolved_target_types)
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    slide_area = float(max((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]), 1e-9))
    source_ids = [nuclei[index]["annotation_id"] for index in np.where(source_mask)[0]]
    target_ids = [nuclei[index]["annotation_id"] for index in np.where(target_mask)[0]]
    whole_slide = _cross_g_curve(
        coords[source_mask],
        source_ids,
        coords[target_mask],
        target_ids,
        slide_area,
        radii,
    )

    per_region = []
    if regions_json_path:
        if not os.path.exists(regions_json_path):
            raise FileNotFoundError(f"regions_json_path not found: {regions_json_path}")
        with open(regions_json_path, "r", encoding="utf-8") as file:
            regions_payload = json.load(file)
        for region in regions_payload.get("regions", []):
            indices = _region_indices_for_bounds(coords, region)
            if not indices:
                continue
            region_coords = coords[indices]
            region_classes = classes[indices]
            region_cells = [nuclei[index] for index in indices]
            region_source_mask = np.isin(region_classes, resolved_source_types)
            region_target_mask = np.isin(region_classes, resolved_target_types)
            region_area = float(max((float(region["x_max"]) - float(region["x_min"])) * (float(region["y_max"]) - float(region["y_min"])), 1e-9))
            region_source_ids = [
                cell["annotation_id"]
                for cell, keep in zip(region_cells, region_source_mask, strict=False)
                if keep
            ]
            region_target_ids = [
                cell["annotation_id"]
                for cell, keep in zip(region_cells, region_target_mask, strict=False)
                if keep
            ]
            per_region.append({
                "region_id": region.get("region_id"),
                "region_label": region.get("region_label"),
                "cell_count": len(indices),
                "x_min": region.get("x_min"),
                "y_min": region.get("y_min"),
                "x_max": region.get("x_max"),
                "y_max": region.get("y_max"),
                "cross_g": _cross_g_curve(
                    region_coords[region_source_mask],
                    region_source_ids,
                    region_coords[region_target_mask],
                    region_target_ids,
                    region_area,
                    radii,
                ),
            })

    payload = {
        "analysis": "kongnet_cross_g_function",
        "annotationstore_path": annotationstore_path,
        "regions_json_path": regions_json_path,
        "method": {
            "metric": "empirical cross-G function",
            "formula": "G_ij(r) = P(nearest target cell of type j is within distance r from a source cell of type i)",
            "csr_reference": "1 - exp(-lambda_j * pi * r^2), using target-cell density in the observation window",
            "meaning": (
                "A higher empirical cross-G at small radii means source cells tend to have target cells nearby. "
                "Values above the CSR reference suggest spatial proximity/enrichment; values below suggest separation."
            ),
            "default_type_note": default_note,
        },
        "parameters": {
            "source_types": resolved_source_types,
            "target_types": resolved_target_types,
            "distance_units": scale["units"],
            "scale": scale,
            "radii": radii,
            "min_probability": float(min_probability),
        },
        "total_nuclei": len(nuclei),
        "class_counts": dict(Counter(classes)),
        "slide_extent": {
            "x_min": float(min_xy[0]),
            "y_min": float(min_xy[1]),
            "x_max": float(max_xy[0]),
            "y_max": float(max_xy[1]),
            "area": slide_area,
        },
        "whole_slide": whole_slide,
        "per_region": per_region,
        "interpretation_warning": (
            "Cross-G is an exploratory distance-based spatial statistic. It does not prove biological interaction, "
            "validate model predictions, or provide a clinical diagnosis. The CSR reference is not edge-corrected."
        ),
    }
    _write_json(output_json_path, payload)

    if output_csv_path:
        ensure_parent_dir(output_csv_path)
        fields = [
            "scope",
            "region_id",
            "region_label",
            "source_types",
            "target_types",
            "source_count",
            "target_count",
            "radius",
            "empirical_cross_g",
            "csr_poisson_expected",
            "difference_from_expected",
            "interpretation",
            "mean_nearest_target_distance",
            "median_nearest_target_distance",
        ]
        with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            def write_curve(scope: str, region_id: str, region_label: str, stats: Dict[str, Any]) -> None:
                if stats.get("status") != "computed":
                    return
                for row in stats.get("curve", []):
                    writer.writerow({
                        "scope": scope,
                        "region_id": region_id,
                        "region_label": region_label,
                        "source_types": ";".join(resolved_source_types),
                        "target_types": ";".join(resolved_target_types),
                        "source_count": stats.get("source_count"),
                        "target_count": stats.get("target_count"),
                        "radius": row.get("radius"),
                        "empirical_cross_g": row.get("empirical_cross_g"),
                        "csr_poisson_expected": row.get("csr_poisson_expected"),
                        "difference_from_expected": row.get("difference_from_expected"),
                        "interpretation": row.get("interpretation"),
                        "mean_nearest_target_distance": stats.get("mean_nearest_target_distance"),
                        "median_nearest_target_distance": stats.get("median_nearest_target_distance"),
                    })

            write_curve("whole_slide", "", "", whole_slide)
            for region in per_region:
                write_curve("region", str(region.get("region_id") or ""), str(region.get("region_label") or ""), region.get("cross_g") or {})

    if output_txt_path:
        ensure_parent_dir(output_txt_path)
        lines = [
            "KongNet Cross-G Tumour-Immune Proximity Report",
            "==============================================",
            "",
            f"Source cell types: {', '.join(resolved_source_types)}",
            f"Target cell types: {', '.join(resolved_target_types)}",
            f"Radii: {', '.join(str(radius) for radius in radii)} {scale['units']}",
            "",
            "What this measures",
            "------------------",
            "Cross-G measures the probability that a source cell has at least one target cell within a given radius.",
            "For tumour-immune analysis, this answers: what fraction of tumour/epithelial cells have an immune cell nearby?",
            "",
            "Whole-slide result",
            "------------------",
        ]
        if whole_slide.get("status") == "computed":
            lines.append(
                f"Source cells: {whole_slide.get('source_count'):,}; target cells: {whole_slide.get('target_count'):,}; "
                f"mean nearest target distance: {whole_slide.get('mean_nearest_target_distance'):.2f} {scale['units']}."
            )
            for row in whole_slide.get("curve", []):
                lines.append(
                    f"- {row['radius']:g} {scale['units']}: G={row['empirical_cross_g']:.3f}; "
                    f"CSR expected={row['csr_poisson_expected']:.3f}; "
                    f"difference={row['difference_from_expected']:.3f} ({row['interpretation']})."
                )
        else:
            lines.append(f"Skipped: {whole_slide.get('reason')}")

        computed_regions = [
            region for region in per_region
            if (region.get("cross_g") or {}).get("status") == "computed"
            and int((region.get("cross_g") or {}).get("source_count") or 0) >= 10
            and int((region.get("cross_g") or {}).get("target_count") or 0) >= 1
        ]
        if computed_regions:
            ranking_radius = max(radii)
            def region_score(region: Dict[str, Any]) -> float:
                for curve_row in (region.get("cross_g") or {}).get("curve", []):
                    if math.isclose(float(curve_row.get("radius")), ranking_radius):
                        return float(curve_row.get("empirical_cross_g", 0.0))
                return 0.0

            top_regions = sorted(computed_regions, key=region_score, reverse=True)[:5]
            lines.extend(["", f"Top regions by cross-G at {ranking_radius:g} {scale['units']}", "----------------------------------------"])
            for index, region in enumerate(top_regions, start=1):
                stats = region.get("cross_g") or {}
                lines.append(
                    f"{index}. {region.get('region_id')}: G={region_score(region):.3f}; "
                    f"{region.get('region_label')}; source cells={stats.get('source_count')}; target cells={stats.get('target_count')}."
                )

        lines.extend([
            "",
            "Interpretation note",
            "-------------------",
            payload["interpretation_warning"],
        ])
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    return "\n".join([
        "KongNet cross-G function analysis completed.",
        f"Source types: {resolved_source_types}",
        f"Target types: {resolved_target_types}",
        f"Radii: {radii} {scale['units']}",
        f"JSON: {output_json_path}",
        f"TXT: {output_txt_path}" if output_txt_path else "TXT: not requested",
        f"CSV: {output_csv_path}" if output_csv_path else "CSV: not requested",
    ])


def tool_compute_kongnet_morans_i(
    regions_json_path: str,
    output_json_path: str,
    output_txt_path: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    weights_method: str = "distance",
    k_neighbours: int = 4,
    distance_threshold: Optional[float] = None,
    permutations: int = 999,
    alpha: float = 0.05,
) -> str:
    """Compute ROI-level Moran's I using PySAL/libpysal/esda.

    Moran's I complements point-level clustering metrics by asking whether
    ROI-level values, such as inflammatory percentage or tumour-immune
    interaction strength, are spatially autocorrelated across neighbouring ROIs.
    """
    import numpy as np
    from esda.moran import Moran

    if not isinstance(regions_json_path, str) or not regions_json_path.strip():
        raise ValueError('compute_kongnet_morans_i requires a non-empty "regions_json_path".')
    if not os.path.exists(regions_json_path):
        raise FileNotFoundError(f"Regions JSON not found: {regions_json_path}")
    if not isinstance(output_json_path, str) or not output_json_path.strip():
        raise ValueError('compute_kongnet_morans_i requires a non-empty "output_json_path".')
    if int(permutations) < 0:
        raise ValueError("permutations must be non-negative.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    with open(regions_json_path, "r", encoding="utf-8") as file:
        regions_payload = json.load(file)
    regions = list(regions_payload.get("regions") or [])
    if len(regions) < 3:
        raise ValueError("Moran's I requires at least 3 retained ROIs.")

    requested_metrics = metrics or _default_moran_roi_metrics(regions)
    weights, weights_details = _build_roi_pysal_weights(
        regions,
        weights_method=weights_method,
        k_neighbours=k_neighbours,
        distance_threshold=distance_threshold,
    )
    weights.transform = "R"

    results = {}
    for metric in requested_metrics:
        values = [_roi_moran_feature_value(region, metric) for region in regions]
        valid_pairs = [
            (region, value)
            for region, value in zip(regions, values)
            if value is not None
        ]
        if len(valid_pairs) < 3:
            results[metric] = {
                "status": "skipped",
                "reason": "Fewer than 3 ROIs had a numeric value for this metric.",
            }
            continue

        valid_region_ids = {str(region.get("region_id") or index) for index, (region, _) in enumerate(valid_pairs)}
        if len(valid_pairs) != len(regions):
            sub_weights = weights.subset(list(valid_region_ids))
        else:
            sub_weights = weights
        if any(len(sub_weights.neighbors.get(region_id, [])) == 0 for region_id in sub_weights.id_order):
            results[metric] = {
                "status": "skipped",
                "reason": "At least one ROI had no spatial neighbours under the selected weight rule.",
            }
            continue

        y = np.asarray([value for _, value in valid_pairs], dtype=float)
        if float(np.var(y)) <= 0.0:
            results[metric] = {
                "status": "skipped",
                "reason": "Metric has zero variance across ROIs.",
                "roi_count": int(len(y)),
            }
            continue

        moran = Moran(y, sub_weights, permutations=int(permutations))
        p_value = _safe_float(getattr(moran, "p_sim", None))
        if p_value is None:
            p_value = _safe_float(getattr(moran, "p_norm", None))
        moran_i = _safe_float(getattr(moran, "I", None))
        results[metric] = {
            "status": "computed",
            "source": "esda.moran.Moran",
            "roi_count": int(len(y)),
            "moran_i": moran_i,
            "expected_i": _safe_float(getattr(moran, "EI", None)),
            "z_score": _safe_float(getattr(moran, "z_sim", None)) or _safe_float(getattr(moran, "z_norm", None)),
            "p_value": p_value,
            "p_value_source": "permutation p-value (p_sim)" if int(permutations) > 0 else "normal approximation (p_norm)",
            "permutations": int(permutations),
            "mean": float(np.mean(y)),
            "standard_deviation": float(np.std(y)),
            "minimum": float(np.min(y)),
            "maximum": float(np.max(y)),
            "interpretation": _moran_interpretation(moran_i, p_value, float(alpha)),
        }

    payload = {
        "analysis": "kongnet_roi_morans_i",
        "regions_json_path": regions_json_path,
        "method": {
            "library": "PySAL ecosystem",
            "weights": weights_details,
            "statistic": "esda.moran.Moran",
            "meaning": (
                "Moran's I measures whether similar ROI-level values occur near each other. "
                "Positive significant values indicate hotspot/coldspot organisation; negative significant "
                "values indicate neighbouring dissimilarity."
            ),
        },
        "parameters": {
            "metrics": requested_metrics,
            "weights_method": str(weights_method),
            "k_neighbours": int(k_neighbours),
            "distance_threshold": distance_threshold,
            "permutations": int(permutations),
            "alpha": float(alpha),
        },
        "region_count": len(regions),
        "results": results,
        "interpretation_warning": (
            "Moran's I is an exploratory ROI-level spatial autocorrelation statistic. "
            "It does not validate model predictions or prove biological causality."
        ),
    }
    _write_json(output_json_path, payload)

    if output_txt_path:
        ensure_parent_dir(output_txt_path)
        lines = [
            "KongNet ROI Moran's I Spatial Autocorrelation",
            "=============================================",
            "",
            f"Regions analysed: {len(regions)}",
            f"Statistic: esda.moran.Moran",
            f"Weights: {weights_details}",
            f"Permutations: {int(permutations)}",
            f"Significance threshold: p < {float(alpha):g}",
            "",
            "Interpretation guide",
            "--------------------",
            "- Moran's I > 0 with significant p-value: high-value ROIs cluster near high-value ROIs and low-value ROIs cluster near low-value ROIs.",
            "- Moran's I < 0 with significant p-value: high-value ROIs tend to sit beside low-value ROIs.",
            "- Non-significant p-value: no strong evidence of ROI-level spatial autocorrelation.",
            "",
            "Results",
            "-------",
        ]
        for metric, result in results.items():
            if result.get("status") != "computed":
                lines.append(f"- {metric}: skipped ({result.get('reason')})")
                continue
            p_value = result.get("p_value")
            p_text = "unavailable" if p_value is None else ("<0.001" if p_value < 0.001 else f"{p_value:.4f}")
            lines.append(
                f"- {metric}: Moran's I {result.get('moran_i'):.3f}; "
                f"p {p_text}; {result.get('interpretation')}"
            )
        lines.extend([
            "",
            "Important note",
            "--------------",
            payload["interpretation_warning"],
        ])
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    computed = sum(1 for result in results.values() if result.get("status") == "computed")
    return "\n".join([
        "KongNet ROI Moran's I analysis completed.",
        f"Regions analysed: {len(regions)}",
        f"Metrics computed: {computed}/{len(results)}",
        f"Weights method: {weights_details}",
        f"JSON: {output_json_path}",
        f"TXT: {output_txt_path}" if output_txt_path else "TXT: not requested",
    ])


LOCAL_MORAN_CLUSTER_COLOURS = {
    "high-high": "#D73027",
    "low-low": "#4575B4",
    "high-low": "#FDAE61",
    "low-high": "#74ADD1",
    "not significant": "#BDBDBD",
}


def _export_local_moran_annotationstore(
    regions_payload: Dict[str, Any],
    valid_pairs,
    rows: List[Dict[str, Any]],
    output_db_path: str,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    overwrite: bool = True,
) -> int:
    """Write Local Moran ROI rectangles as a TIAViz-compatible AnnotationStore."""
    from shapely.geometry import box
    from tiatoolbox.annotation.storage import Annotation, SQLiteStore

    units = str(regions_payload.get("distance_units", "pixels")).lower()
    stored_scale = regions_payload.get("scale") or {}
    if units == "pixels":
        x_scale = y_scale = 1.0
    elif mpp is not None or wsi_path:
        resolved = _resolve_spatial_scale("microns", wsi_path=wsi_path, mpp=mpp)
        x_scale, y_scale = resolved["x_scale"], resolved["y_scale"]
    elif stored_scale.get("x_scale") and stored_scale.get("y_scale"):
        x_scale = float(stored_scale["x_scale"])
        y_scale = float(stored_scale["y_scale"])
    else:
        raise ValueError(
            "Micron-based Local Moran ROI coordinates require wsi_path, mpp, "
            "or scale metadata in the regions JSON."
        )

    # AnnotationStore geometries use baseline pixels.  Grid ROIs can extend
    # past the right/bottom edge when the slide size is not an exact multiple
    # of the requested ROI size, so clip them to the real image dimensions.
    slide_bounds = None
    if wsi_path:
        try:
            from tiatoolbox.wsicore.wsireader import WSIReader

            reader = WSIReader.open(wsi_path)
            width_px, height_px = (
                int(value) for value in reader.info.slide_dimensions
            )
        except Exception:
            # Reading TIFF metadata directly is a lightweight fallback for
            # environments where TIAToolbox's optional zarr stack is absent or
            # incompatible. No pixel data are loaded here.
            from tifffile import TiffFile

            with TiffFile(wsi_path) as tif:
                axes = tif.series[0].axes
                shape = tif.series[0].shape
                width_px = int(shape[axes.index("X")])
                height_px = int(shape[axes.index("Y")])
        slide_bounds = box(0.0, 0.0, float(width_px), float(height_px))

    ensure_parent_dir(output_db_path)
    if os.path.exists(output_db_path):
        if not overwrite:
            raise FileExistsError(f"Output AnnotationStore already exists: {output_db_path}")
        os.remove(output_db_path)

    annotations, keys = [], []
    type_ids = {
        "high-high": 0,
        "low-low": 1,
        "high-low": 2,
        "low-high": 3,
        "not significant": 4,
    }
    for (region, _), row in zip(valid_pairs, rows, strict=True):
        cluster_label = str(row["cluster_label"])
        colour = LOCAL_MORAN_CLUSTER_COLOURS.get(cluster_label, "#BDBDBD")
        geometry = box(
            float(region["x_min"]) / x_scale,
            float(region["y_min"]) / y_scale,
            float(region["x_max"]) / x_scale,
            float(region["y_max"]) / y_scale,
        )
        was_clipped = False
        if slide_bounds is not None:
            unclipped_geometry = geometry
            geometry = geometry.intersection(slide_bounds)
            was_clipped = not geometry.equals(unclipped_geometry)
            if geometry.is_empty:
                continue
        properties = {
            "type": cluster_label,
            "label": cluster_label,
            "type_id": type_ids.get(cluster_label, type_ids["not significant"]),
            "region_id": row["region_id"],
            "region_label": row.get("region_label"),
            "metric": row["metric"],
            "feature_value": row["feature_value"],
            "standardized_feature_value": row["standardized_feature_value"],
            "spatial_lag": row["spatial_lag"],
            "local_moran_i": row["local_moran_i"],
            "p_value": row["p_value"],
            "quadrant": row["quadrant"],
            "raw_cluster_label": row["raw_cluster_label"],
            "cluster_label": cluster_label,
            "significant": row["significant"],
            "neighbour_count": row["neighbour_count"],
            "neighbour_ids": ";".join(row["neighbour_ids"]),
            "colour": colour,
            "color": colour,
            "line_color": colour,
            "fill_color": colour,
            "fill_opacity": 0.42 if row["significant"] else 0.12,
            "is_roi": True,
            "coordinate_space": "baseline",
            "clipped_to_slide_bounds": was_clipped,
            "source": "KongNet Local Moran's I",
        }
        annotations.append(Annotation(geometry, properties=properties))
        keys.append(str(row["region_id"]))

    store = SQLiteStore(output_db_path)
    try:
        store.append_many(annotations, keys=keys)
        store.commit()
        return len(store)
    finally:
        store.close()


def tool_compute_kongnet_local_morans_i(
    regions_json_path: str,
    output_json_path: str,
    metric: str,
    output_csv_path: Optional[str] = None,
    output_txt_path: Optional[str] = None,
    output_annotationstore_path: Optional[str] = None,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    overwrite: bool = True,
    weights_method: str = "distance",
    k_neighbours: int = 4,
    distance_threshold: Optional[float] = None,
    permutations: int = 999,
    alpha: float = 0.05,
    seed: int = 42,
) -> str:
    """Identify ROI-level clusters and spatial outliers using Local Moran's I.

    The statistic is calculated for one numeric feature per ROI. Distance weights
    use ROI-centroid distances, matching the global KongNet Moran's I tool.
    """
    import numpy as np
    from esda.moran import Moran_Local
    from libpysal.weights import lag_spatial

    if not isinstance(regions_json_path, str) or not regions_json_path.strip():
        raise ValueError('compute_kongnet_local_morans_i requires a non-empty "regions_json_path".')
    if not os.path.exists(regions_json_path):
        raise FileNotFoundError(f"Regions JSON not found: {regions_json_path}")
    if not isinstance(output_json_path, str) or not output_json_path.strip():
        raise ValueError('compute_kongnet_local_morans_i requires a non-empty "output_json_path".')
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError('compute_kongnet_local_morans_i requires a non-empty "metric".')
    if int(permutations) < 0:
        raise ValueError("permutations must be non-negative.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    with open(regions_json_path, "r", encoding="utf-8") as file:
        regions_payload = json.load(file)
    all_regions = list(regions_payload.get("regions") or [])
    valid_pairs = [
        (region, _roi_moran_feature_value(region, metric))
        for region in all_regions
    ]
    valid_pairs = [(region, value) for region, value in valid_pairs if value is not None]
    if len(valid_pairs) < 3:
        raise ValueError("Local Moran's I requires at least 3 ROIs with numeric feature values.")

    regions = [region for region, _ in valid_pairs]
    values = np.asarray([value for _, value in valid_pairs], dtype=float)
    if float(np.var(values)) <= 0.0:
        raise ValueError("Local Moran's I cannot be computed because the selected metric has zero variance.")

    weights, weights_details = _build_roi_pysal_weights(
        regions,
        weights_method=weights_method,
        k_neighbours=k_neighbours,
        distance_threshold=distance_threshold,
    )
    isolated_ids = [
        region_id
        for region_id in weights.id_order
        if len(weights.neighbors.get(region_id, [])) == 0
    ]
    if isolated_ids:
        raise ValueError(
            "Local Moran's I requires every retained ROI to have at least one neighbour. "
            f"Isolated ROI IDs: {isolated_ids}"
        )

    weights.transform = "R"
    np.random.seed(int(seed))
    local = Moran_Local(
        values,
        weights,
        permutations=int(permutations),
        seed=int(seed),
    )

    standardized_values = np.asarray(local.z, dtype=float)
    spatial_lags = np.asarray(lag_spatial(weights, standardized_values), dtype=float)
    quadrant_labels = {
        1: "high-high",
        2: "low-high",
        3: "low-low",
        4: "high-low",
    }
    rows = []
    for index, (region, feature_value) in enumerate(valid_pairs):
        region_id = str(region.get("region_id") or index)
        quadrant = int(local.q[index])
        p_value = _safe_float(local.p_sim[index]) if int(permutations) > 0 else None
        significant = p_value is not None and p_value < float(alpha)
        raw_cluster_label = quadrant_labels.get(quadrant, "unclassified")
        cluster_label = raw_cluster_label if significant else "not significant"
        centroid_x = (float(region["x_min"]) + float(region["x_max"])) / 2.0
        centroid_y = (float(region["y_min"]) + float(region["y_max"])) / 2.0
        rows.append({
            "region_id": region_id,
            "region_label": region.get("region_label"),
            "metric": metric,
            "feature_value": float(feature_value),
            "standardized_feature_value": float(standardized_values[index]),
            "spatial_lag": float(spatial_lags[index]),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "local_moran_i": float(local.Is[index]),
            "p_value": p_value,
            "p_value_source": "conditional permutation p-value (p_sim)" if int(permutations) > 0 else None,
            "quadrant": quadrant,
            "raw_cluster_label": raw_cluster_label,
            "cluster_label": cluster_label,
            "significant": bool(significant),
            "neighbour_count": len(weights.neighbors.get(region_id, [])),
            "neighbour_ids": list(weights.neighbors.get(region_id, [])),
        })

    cluster_counts = dict(Counter(row["cluster_label"] for row in rows))
    significant_positive = [
        row for row in rows
        if row["significant"] and row["local_moran_i"] > 0
        and row["raw_cluster_label"] in {"high-high", "low-low"}
    ]
    resolved_annotationstore_path = output_annotationstore_path or (
        os.path.splitext(output_json_path)[0] + "_overlay.db"
    )
    annotation_count = _export_local_moran_annotationstore(
        regions_payload=regions_payload,
        valid_pairs=valid_pairs,
        rows=rows,
        output_db_path=resolved_annotationstore_path,
        wsi_path=wsi_path,
        mpp=mpp,
        overwrite=overwrite,
    )

    payload = {
        "analysis": "kongnet_roi_local_morans_i",
        "regions_json_path": regions_json_path,
        "method": {
            "library": "PySAL ecosystem",
            "weights": weights_details,
            "weights_transform": "row-standardized",
            "statistic": "esda.moran.Moran_Local",
            "quadrants": quadrant_labels,
        },
        "parameters": {
            "metric": metric,
            "weights_method": str(weights_method),
            "k_neighbours": int(k_neighbours),
            "distance_threshold": distance_threshold,
            "permutations": int(permutations),
            "alpha": float(alpha),
            "seed": int(seed),
        },
        "region_count": len(rows),
        "annotationstore_path": resolved_annotationstore_path,
        "annotation_count": annotation_count,
        "visualization_legend": LOCAL_MORAN_CLUSTER_COLOURS,
        "cluster_counts": cluster_counts,
        "significant_positive_cluster_roi_ids": [row["region_id"] for row in significant_positive],
        "regions": rows,
        "interpretation_warning": (
            "Local Moran's I is exploratory and involves multiple ROI-level tests. "
            "Permutation significance does not prove biological causality; consider a multiple-testing correction."
        ),
    }
    _write_json(output_json_path, payload)

    if output_csv_path:
        ensure_parent_dir(output_csv_path)
        fieldnames = [
            "region_id", "region_label", "metric", "feature_value",
            "standardized_feature_value", "spatial_lag", "centroid_x", "centroid_y",
            "local_moran_i", "p_value", "quadrant", "raw_cluster_label",
            "cluster_label", "significant", "neighbour_count", "neighbour_ids",
        ]
        with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                output_row = {key: row.get(key) for key in fieldnames}
                output_row["neighbour_ids"] = ";".join(row["neighbour_ids"])
                writer.writerow(output_row)

    if output_txt_path:
        ensure_parent_dir(output_txt_path)
        lines = [
            "KongNet ROI Local Moran's I",
            "===========================",
            "",
            f"Metric: {metric}",
            f"Regions analysed: {len(rows)}",
            f"Weights: {weights_details}",
            "Weights transform: row-standardized",
            f"Permutations: {int(permutations)}",
            f"Significance threshold: p < {float(alpha):g}",
            "",
            "Significant positive local clusters",
            "-----------------------------------",
        ]
        if significant_positive:
            for row in sorted(significant_positive, key=lambda item: item["p_value"]):
                lines.append(
                    f"- {row['region_id']}: {row['raw_cluster_label']}; "
                    f"Local I={row['local_moran_i']:.4f}; p={row['p_value']:.4f}; "
                    f"{metric}={row['feature_value']:.4f}"
                )
        else:
            lines.append("- None at the selected alpha level.")
        lines.extend(["", payload["interpretation_warning"]])
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    return "\n".join([
        "KongNet ROI Local Moran's I analysis completed.",
        f"Metric: {metric}",
        f"Regions analysed: {len(rows)}",
        f"Significant positive cluster ROIs: {[row['region_id'] for row in significant_positive]}",
        f"JSON: {output_json_path}",
        f"CSV: {output_csv_path}" if output_csv_path else "CSV: not requested",
        f"TXT: {output_txt_path}" if output_txt_path else "TXT: not requested",
        f"TIAViz AnnotationStore: {resolved_annotationstore_path}",
    ])


def _pointpats_standard_point_pattern_metrics(
    coords,
    radii: List[float],
    quadrat_grid_size: int,
) -> Dict[str, Any]:
    """Compute standard pointpats metrics when the library is available.

    The rest of the tool keeps transparent NumPy/SciPy values for reporting and
    fallback. This block records the direct pointpats outputs so the workflow
    genuinely integrates the library where its standard APIs fit the problem.
    """
    import numpy as np
    import warnings

    result: Dict[str, Any] = {
        "available": False,
        "computed": False,
        "metrics": {},
    }
    try:
        import pointpats  # type: ignore
        from pointpats import PointPattern, QStatistic  # type: ignore
    except Exception as exc:
        result.update({
            "error": f"{type(exc).__name__}: {exc}",
        })
        return result

    result["available"] = True
    result["version"] = getattr(pointpats, "__version__", "unknown")

    if len(coords) < 2:
        result["error"] = "At least two points are required for pointpats point-pattern metrics."
        return result

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pattern = PointPattern(np.asarray(coords, dtype=float))
            metrics: Dict[str, Any] = {
                "point_count": int(pattern.n),
                "minimum_bounding_box": [float(value) for value in pattern.mbb],
                "minimum_bounding_box_area": float(pattern.mbb_area),
                "lambda_mbb": float(pattern.lambda_mbb),
                "mean_nearest_neighbour_distance": float(pattern.mean_nnd),
                "minimum_nearest_neighbour_distance": float(pattern.min_nnd),
                "maximum_nearest_neighbour_distance": float(pattern.max_nnd),
                "method": "pointpats.PointPattern",
            }

            try:
                quadrat = QStatistic(
                    pattern,
                    nx=int(quadrat_grid_size),
                    ny=int(quadrat_grid_size),
                )
                metrics["quadrat"] = {
                    "chi2": float(quadrat.chi2),
                    "chi2_pvalue": float(quadrat.chi2_pvalue),
                    "degrees_of_freedom": int(quadrat.df),
                    "grid_size": int(quadrat_grid_size),
                    "method": "pointpats.QStatistic",
                    "interpretation": (
                        "low p-value suggests counts differ across quadrats more than expected "
                        "under complete spatial randomness"
                    ),
                }
            except Exception as exc:
                metrics["quadrat"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "method": "pointpats.QStatistic",
                }

            try:
                support, l_values = pointpats.l(
                    np.asarray(coords, dtype=float),
                    support=np.asarray(radii, dtype=float),
                    edge_correction=None,
                )
                metrics["l_function"] = [
                    {
                        "radius": float(radius),
                        "pointpats_l": float(l_value),
                        "pointpats_l_minus_r": float(l_value - radius),
                        "method": "pointpats.l",
                        "edge_correction": None,
                    }
                    for radius, l_value in zip(support, l_values, strict=False)
                ]
            except Exception as exc:
                metrics["l_function"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "method": "pointpats.l",
                }

        result["computed"] = True
        result["metrics"] = metrics
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _summarise_unmarked_point_pattern(
    coords,
    area: float,
    radii: List[float],
    quadrat_grid_size: int,
    min_points_per_pattern: int,
) -> Dict[str, Any]:
    import numpy as np
    from scipy.spatial import cKDTree

    n_points = int(len(coords))
    if n_points < int(min_points_per_pattern) or area <= 0:
        return {
            "point_count": n_points,
            "status": "insufficient points",
            "minimum_required": int(min_points_per_pattern),
            "pointpats_standard_metrics": _pointpats_standard_point_pattern_metrics(
                coords,
                radii,
                quadrat_grid_size,
            ) if n_points >= 2 else {"available": _pointpats_availability()["available"], "computed": False},
        }

    pointpats_metrics = _pointpats_standard_point_pattern_metrics(
        coords,
        radii,
        quadrat_grid_size,
    )
    density = float(n_points / area)
    tree = cKDTree(coords)
    nearest = tree.query(coords, k=2)[0][:, 1] if n_points > 1 else np.asarray([], dtype=float)
    pointpats_nn = (
        pointpats_metrics.get("metrics", {})
        .get("mean_nearest_neighbour_distance")
        if pointpats_metrics.get("computed")
        else None
    )
    observed_mean_nn = (
        float(pointpats_nn)
        if pointpats_nn is not None
        else float(np.mean(nearest)) if nearest.size else None
    )
    expected_csr_mean_nn = float(0.5 / math.sqrt(density)) if density > 0 else None
    nearest_neighbour_index = (
        float(observed_mean_nn / expected_csr_mean_nn)
        if observed_mean_nn is not None and expected_csr_mean_nn and expected_csr_mean_nn > 0
        else None
    )

    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-9)
    grid_size = max(1, int(quadrat_grid_size))
    x_edges = np.linspace(min_xy[0], max_xy[0] + 1e-9, grid_size + 1)
    y_edges = np.linspace(min_xy[1], max_xy[1] + 1e-9, grid_size + 1)
    quadrat_counts, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_edges, y_edges])
    quadrat_values = quadrat_counts.ravel()
    quadrat_mean = float(np.mean(quadrat_values))
    quadrat_variance = float(np.var(quadrat_values, ddof=1)) if len(quadrat_values) > 1 else 0.0
    quadrat_vmr = float(quadrat_variance / quadrat_mean) if quadrat_mean > 0 else None

    pointpats_l_lookup = {}
    pointpats_l_values = pointpats_metrics.get("metrics", {}).get("l_function")
    if isinstance(pointpats_l_values, list):
        for row in pointpats_l_values:
            try:
                pointpats_l_lookup[float(row["radius"])] = row
            except (KeyError, TypeError, ValueError):
                continue

    ripley = []
    for radius in radii:
        r = float(radius)
        if r <= 0:
            continue
        neighbour_counts = np.asarray([len(indices) - 1 for indices in tree.query_ball_point(coords, r)], dtype=float)
        total_neighbour_links = float(np.sum(neighbour_counts))
        k_value = float(area * total_neighbour_links / (n_points * (n_points - 1))) if n_points > 1 else None
        custom_l_value = float(math.sqrt(k_value / math.pi)) if k_value is not None and k_value >= 0 else None
        custom_l_minus_r = float(custom_l_value - r) if custom_l_value is not None else None
        pointpats_row = pointpats_l_lookup.get(r)
        pointpats_l_value = _safe_float(pointpats_row.get("pointpats_l")) if pointpats_row else None
        pointpats_l_minus_r = _safe_float(pointpats_row.get("pointpats_l_minus_r")) if pointpats_row else None
        l_minus_r = pointpats_l_minus_r if pointpats_l_minus_r is not None else custom_l_minus_r
        ripley.append({
            "radius": r,
            "mean_neighbours_per_point": float(np.mean(neighbour_counts)) if neighbour_counts.size else 0.0,
            "ripley_k_no_edge_correction": k_value,
            "pointpats_l_no_edge_correction": pointpats_l_value,
            "pointpats_l_minus_r_no_edge_correction": pointpats_l_minus_r,
            "custom_l_no_edge_correction": custom_l_value,
            "custom_l_minus_r_no_edge_correction": custom_l_minus_r,
            "ripley_l_minus_r_no_edge_correction": l_minus_r,
            "ripley_l_source": "pointpats.l" if pointpats_l_minus_r is not None else "custom scipy.cKDTree fallback",
            "interpretation": (
                "clustered at this scale" if l_minus_r is not None and l_minus_r > 0
                else "not clustered at this scale" if l_minus_r is not None
                else "insufficient points"
            ),
        })

    return {
        "point_count": n_points,
        "status": "computed",
        "density_per_square_unit": density,
        "mean_nearest_neighbour_distance": observed_mean_nn,
        "mean_nearest_neighbour_distance_source": (
            "pointpats.PointPattern.mean_nnd" if pointpats_nn is not None else "scipy.cKDTree fallback"
        ),
        "expected_csr_mean_nearest_neighbour_distance": expected_csr_mean_nn,
        "nearest_neighbour_index": nearest_neighbour_index,
        "nearest_neighbour_interpretation": _point_pattern_label(nearest_neighbour_index),
        "quadrat_grid_size": grid_size,
        "quadrat_count_mean": quadrat_mean,
        "quadrat_count_variance": quadrat_variance,
        "quadrat_variance_to_mean_ratio": quadrat_vmr,
        "quadrat_interpretation": _vmr_label(quadrat_vmr),
        "ripley_l_by_radius_no_edge_correction": ripley,
        "pointpats_standard_metrics": pointpats_metrics,
        "extent_width": float(span[0]),
        "extent_height": float(span[1]),
    }


def _summarise_cross_type_proximity(source_coords, target_coords, area: float, radii: List[float]) -> Dict[str, Any]:
    import numpy as np
    from scipy.spatial import cKDTree

    source_count = int(len(source_coords))
    target_count = int(len(target_coords))
    if source_count < 1 or target_count < 1 or area <= 0:
        return {
            "source_count": source_count,
            "target_count": target_count,
            "status": "insufficient source or target points",
        }

    target_tree = cKDTree(target_coords)
    target_density = float(target_count / area)
    by_radius = []
    for radius in radii:
        r = float(radius)
        if r <= 0:
            continue
        counts = np.asarray(
            [len(target_tree.query_ball_point(point, r)) for point in source_coords],
            dtype=float,
        )
        observed = float(np.mean(counts)) if counts.size else 0.0
        expected = float(target_density * math.pi * r * r)
        ratio = float(observed / expected) if expected > 0 else None
        by_radius.append({
            "radius": r,
            "observed_mean_target_neighbours_per_source": observed,
            "expected_mean_under_csr": expected,
            "observed_to_expected_ratio": ratio,
            "interpretation": (
                "more cross-type proximity than expected from density alone"
                if ratio is not None and ratio > 1.25
                else "less cross-type proximity than expected from density alone"
                if ratio is not None and ratio < 0.75
                else "close to density-based expectation"
                if ratio is not None
                else "insufficient points"
            ),
        })

    return {
        "source_count": source_count,
        "target_count": target_count,
        "status": "computed",
        "target_density_per_square_unit": target_density,
        "by_radius": by_radius,
    }


def _region_indices_for_bounds(coords, region: Dict[str, Any]) -> List[int]:
    indices = []
    x_min = float(region["x_min"])
    y_min = float(region["y_min"])
    x_max = float(region["x_max"])
    y_max = float(region["y_max"])
    for index, point in enumerate(coords):
        x, y = float(point[0]), float(point[1])
        if x_min <= x < x_max and y_min <= y < y_max:
            indices.append(index)
    return indices


def tool_compute_kongnet_point_pattern_statistics(
    annotationstore_path: str,
    output_json_path: str,
    output_txt_path: Optional[str] = None,
    regions_json_path: Optional[str] = None,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    distance_units: str = "microns",
    radii: Optional[List[float]] = None,
    quadrat_grid_size: int = 4,
    min_points_per_pattern: int = 1,
    min_probability: float = 0.0,
    cell_types: Optional[List[str]] = None,
    source_types: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
) -> str:
    """Compute point-pattern statistics for KongNet nucleus coordinates.

    The output is designed to complement the existing ROI counts by reporting
    nearest-neighbour clustering, quadrat heterogeneity, Ripley-style
    multi-radius clustering evidence, and tumour/epithelial-immune proximity.
    """
    import numpy as np

    if quadrat_grid_size < 1:
        raise ValueError("quadrat_grid_size must be at least 1.")
    if min_points_per_pattern < 1:
        raise ValueError("min_points_per_pattern must be at least 1.")

    radii = [float(value) for value in (radii or [25.0, 50.0, 100.0]) if float(value) > 0]
    if not radii:
        raise ValueError("At least one positive radius is required.")

    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    nuclei = _load_kongnet_nuclei(annotationstore_path, None, min_probability)
    if not nuclei:
        raise RuntimeError("No nuclei matched the requested probability threshold.")

    coords = _nucleus_coordinates(nuclei, scale)
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    slide_area = float(max((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]), 1e-9))
    available_class_names = _ordered_kongnet_class_names(nuclei)
    class_names = (
        [name for name in _normalise_kongnet_cell_types(cell_types) if name in available_class_names]
        if cell_types
        else available_class_names
    )
    if not class_names:
        raise ValueError("No valid cell_types were selected for point-pattern statistics.")
    classes = np.asarray([cell["type"] for cell in nuclei], dtype=object)

    whole_slide_by_class = {}
    for class_name in class_names:
        class_coords = coords[classes == class_name]
        whole_slide_by_class[class_name] = _summarise_unmarked_point_pattern(
            class_coords,
            slide_area,
            radii,
            quadrat_grid_size,
            min_points_per_pattern,
        )

    immune_types = (
        [name for name in _normalise_kongnet_cell_types(target_types) if name in available_class_names]
        if target_types
        else [name for name in KONGNET_IMMUNE_CELL_TYPES if name in available_class_names]
    )
    tumour_source_types = (
        [name for name in _normalise_kongnet_cell_types(source_types) if name in available_class_names]
        if source_types
        else [name for name in ["Neoplastic"] if name in available_class_names]
    )
    if not source_types and not tumour_source_types and "Epithelial" in available_class_names:
        tumour_source_types = ["Epithelial"]
    if not tumour_source_types or not immune_types:
        raise ValueError("Point-pattern source_types and target_types must resolve to valid KongNet cell classes.")
    source_mask = np.isin(classes, tumour_source_types)
    target_mask = np.isin(classes, immune_types)
    cross_type_summary = {
        "source_types": tumour_source_types,
        "target_types": immune_types,
        "note": (
            "Uses neoplastic cells as tumour source when available; falls back to epithelial cells "
            "for models such as CoNIC that do not contain a neoplastic class."
        ),
        "statistics": _summarise_cross_type_proximity(coords[source_mask], coords[target_mask], slide_area, radii),
    }

    regions_payload = None
    per_region = []
    if regions_json_path:
        if not os.path.exists(regions_json_path):
            raise FileNotFoundError(f"regions_json_path not found: {regions_json_path}")
        with open(regions_json_path, "r", encoding="utf-8") as file:
            regions_payload = json.load(file)
        for region in regions_payload.get("regions", []):
            indices = _region_indices_for_bounds(coords, region)
            region_coords = coords[indices]
            region_classes = classes[indices]
            region_area = float(max((float(region["x_max"]) - float(region["x_min"])) * (float(region["y_max"]) - float(region["y_min"])), 1e-9))
            region_by_class = {}
            for class_name in class_names:
                region_by_class[class_name] = _summarise_unmarked_point_pattern(
                    region_coords[region_classes == class_name],
                    region_area,
                    radii,
                    quadrat_grid_size,
                    min_points_per_pattern,
                )
            region_source_mask = np.isin(region_classes, tumour_source_types)
            region_target_mask = np.isin(region_classes, immune_types)
            per_region.append({
                "region_id": region.get("region_id"),
                "region_label": region.get("region_label"),
                "cell_count": len(indices),
                "x_min": region.get("x_min"),
                "y_min": region.get("y_min"),
                "x_max": region.get("x_max"),
                "y_max": region.get("y_max"),
                "by_class": region_by_class,
                "tumour_or_epithelial_immune_cross_type": _summarise_cross_type_proximity(
                    region_coords[region_source_mask],
                    region_coords[region_target_mask],
                    region_area,
                    radii,
                ),
            })

    payload = {
        "analysis": "kongnet_point_pattern_statistics",
        "annotationstore_path": annotationstore_path,
        "regions_json_path": regions_json_path,
        "pointpats": _pointpats_availability(),
        "method": {
            "nearest_neighbour_index": "Observed mean nearest-neighbour distance from pointpats.PointPattern.mean_nnd when available, divided by CSR expectation 0.5/sqrt(lambda). Values below 1 suggest clustering.",
            "pointpats_quadrat": "pointpats.QStatistic is used to compute standard quadrat chi-square and p-value when available.",
            "quadrat_vmr": "Transparent custom variance-to-mean ratio of counts across a local grid. Values above 1 indicate heterogeneous/clumped counts.",
            "ripley_l_minus_r": "pointpats.l is used for L(r)-r when available; positive values suggest clustering at that radius. No edge correction is applied.",
            "cross_type_ratio": "Custom pathology-specific metric: observed mean immune neighbours per tumour/epithelial source divided by density-based CSR expectation.",
        },
        "parameters": {
            "distance_units": scale["units"],
            "scale": scale,
            "radii": radii,
            "quadrat_grid_size": int(quadrat_grid_size),
            "min_points_per_pattern": int(min_points_per_pattern),
            "min_probability": float(min_probability),
            "cell_types": class_names,
            "source_types": tumour_source_types,
            "target_types": immune_types,
        },
        "slide_extent": {
            "x_min": float(min_xy[0]),
            "y_min": float(min_xy[1]),
            "x_max": float(max_xy[0]),
            "y_max": float(max_xy[1]),
            "area": slide_area,
        },
        "total_nuclei": len(nuclei),
        "class_counts": dict(Counter(classes)),
        "whole_slide_by_class": whole_slide_by_class,
        "tumour_or_epithelial_immune_cross_type": cross_type_summary,
        "per_region": per_region,
        "interpretation_warning": (
            "These are model-derived exploratory spatial statistics. They do not prove biological interaction "
            "or provide a clinical diagnosis. Ripley-style values are reported without edge correction."
        ),
    }
    _write_json(output_json_path, payload)

    if output_txt_path:
        ensure_parent_dir(output_txt_path)
        lines = [
            "KongNet Point-Pattern Spatial Statistics",
            "========================================",
            "",
            f"Total nuclei analysed: {len(nuclei):,}",
            f"Distance units: {scale['units']}",
            f"Radii: {', '.join(str(r) for r in radii)} {scale['units']}",
            f"pointpats available: {payload['pointpats']['available']}",
            f"pointpats note: {payload['pointpats']['note']}",
            "",
            "Whole-slide class organisation",
            "------------------------------",
        ]
        for class_name, stats in whole_slide_by_class.items():
            if stats.get("status") != "computed":
                lines.append(f"- {class_name}: {stats.get('point_count', 0)} points; insufficient points for stable statistics.")
                continue
            nni = _safe_float(stats.get("nearest_neighbour_index"))
            vmr = _safe_float(stats.get("quadrat_variance_to_mean_ratio"))
            lines.append(
                f"- {class_name}: {stats.get('point_count', 0):,} points; "
                f"NNI {nni:.3f} ({stats.get('nearest_neighbour_interpretation')})" if nni is not None
                else f"- {class_name}: {stats.get('point_count', 0):,} points; NNI unavailable"
            )
            lines.append(
                f"  Quadrat VMR: {vmr:.3f} ({stats.get('quadrat_interpretation')})"
                if vmr is not None else "  Quadrat VMR: unavailable"
            )
        lines.extend([
            "",
            "Tumour/epithelial-to-immune proximity",
            "-------------------------------------",
            f"Source types: {', '.join(tumour_source_types) if tumour_source_types else 'none'}",
            f"Target immune types: {', '.join(immune_types) if immune_types else 'none'}",
        ])
        cross_stats = cross_type_summary["statistics"]
        if cross_stats.get("status") == "computed":
            for row in cross_stats.get("by_radius", []):
                ratio = _safe_float(row.get("observed_to_expected_ratio"))
                lines.append(
                    f"- {row['radius']} {scale['units']}: observed/expected ratio "
                    f"{ratio:.3f} - {row.get('interpretation')}" if ratio is not None
                    else f"- {row['radius']} {scale['units']}: ratio unavailable"
                )
        else:
            lines.append(f"- {cross_stats.get('status')}")

        if per_region:
            lines.extend(["", "Top ROI-level signals", "---------------------"])
            region_rows = []
            radius_key = radii[min(1, len(radii) - 1)]
            for region in per_region:
                best_class = None
                best_nni = None
                for class_name, stats in region.get("by_class", {}).items():
                    nni = _safe_float(stats.get("nearest_neighbour_index"))
                    if nni is not None and (best_nni is None or nni < best_nni):
                        best_nni = nni
                        best_class = class_name
                ratio = None
                for row in region.get("tumour_or_epithelial_immune_cross_type", {}).get("by_radius", []):
                    if float(row.get("radius", -1)) == float(radius_key):
                        ratio = _safe_float(row.get("observed_to_expected_ratio"))
                        break
                region_rows.append((region.get("region_id"), region.get("region_label"), best_class, best_nni, ratio))
            for region_id, label, best_class, best_nni, ratio in sorted(region_rows, key=lambda row: (row[3] if row[3] is not None else 999.0))[:5]:
                lines.append(
                    f"- {region_id} ({label}): strongest class clustering = {best_class or 'unavailable'} "
                    f"with NNI {best_nni:.3f}" if best_nni is not None
                    else f"- {region_id} ({label}): insufficient class-specific points"
                )
                if ratio is not None:
                    lines.append(f"  Tumour/epithelial-immune observed/expected ratio at {radius_key} {scale['units']}: {ratio:.3f}")
        lines.extend([
            "",
            "Important interpretation warning:",
            "These statistics describe spatial organisation of model-predicted nuclei. They are exploratory, sensitive to ROI size/radius/model errors, and are not a clinical diagnosis.",
        ])
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))

    return "\n".join([
        "KongNet point-pattern statistics completed.",
        f"Nuclei analysed: {len(nuclei)}",
        f"Classes analysed: {', '.join(class_names)}",
        f"pointpats available: {payload['pointpats']['available']}",
        f"JSON: {output_json_path}",
        f"Text report: {output_txt_path}" if output_txt_path else "Text report: not requested",
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
    if len(regions) < 2:
        raise ValueError(
            "At least two retained local regions are required for a meaningful heatmap. "
            "Use a smaller region_size or min_cells_per_region, then rerun ROI analysis."
        )
    os.makedirs(output_dir, exist_ok=True)
    slide_prefix = _slide_stem_prefix(wsi_path)

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
        db_path = os.path.join(output_dir, f"{slide_prefix}kongnet_{metric}_heatmap.db")
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
                metric: value,
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


def tool_generate_kongnet_point_pattern_overlays(
    point_pattern_json_path: str,
    output_dir: str,
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    overwrite: bool = True,
) -> str:
    """Generate TIAViz ROI overlays for point-pattern clustering statistics."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize, to_hex
    from shapely.geometry import box
    from tiatoolbox.annotation.storage import Annotation, SQLiteStore

    if not os.path.exists(point_pattern_json_path):
        raise FileNotFoundError(f"Point-pattern JSON not found: {point_pattern_json_path}")
    with open(point_pattern_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    per_region = data.get("per_region", [])
    if not per_region:
        raise ValueError(
            "Point-pattern JSON contains no per-region records. Re-run "
            "compute_kongnet_point_pattern_statistics with regions_json_path supplied."
        )
    if len(per_region) < 2:
        raise ValueError(
            "At least two retained local regions are required for meaningful point-pattern overlays."
        )
    os.makedirs(output_dir, exist_ok=True)
    slide_prefix = _slide_stem_prefix(wsi_path)

    parameters = data.get("parameters", {})
    units = str(parameters.get("distance_units", "pixels")).lower()
    stored_scale = parameters.get("scale") or {}
    if units == "pixels":
        x_scale = y_scale = 1.0
    elif mpp is not None or wsi_path:
        scale = _resolve_spatial_scale("microns", wsi_path=wsi_path, mpp=mpp)
        x_scale, y_scale = scale["x_scale"], scale["y_scale"]
    elif stored_scale.get("x_scale") and stored_scale.get("y_scale"):
        x_scale, y_scale = float(stored_scale["x_scale"]), float(stored_scale["y_scale"])
    else:
        raise ValueError("Micron-based point-pattern overlays require wsi_path, mpp, or scale metadata.")

    radii = [float(value) for value in parameters.get("radii", [])]
    preferred_radius = radii[min(1, len(radii) - 1)] if radii else None

    def best_class_by_nni(region: Dict[str, Any]) -> Dict[str, Any]:
        best = {
            "class_name": None,
            "point_count": 0,
            "nearest_neighbour_index": None,
            "quadrat_vmr": None,
            "ripley_strength": None,
        }
        for class_name, stats in (region.get("by_class") or {}).items():
            if stats.get("status") != "computed":
                continue
            nni = _safe_float(stats.get("nearest_neighbour_index"))
            if nni is None:
                continue
            current_best = _safe_float(best.get("nearest_neighbour_index"))
            if current_best is None or nni < current_best:
                ripley_values = [
                    _safe_float(row.get("ripley_l_minus_r_no_edge_correction"))
                    for row in stats.get("ripley_l_by_radius_no_edge_correction", [])
                ]
                ripley_values = [value for value in ripley_values if value is not None]
                best = {
                    "class_name": class_name,
                    "point_count": int(stats.get("point_count", 0)),
                    "nearest_neighbour_index": nni,
                    "quadrat_vmr": _safe_float(stats.get("quadrat_variance_to_mean_ratio")),
                    "ripley_strength": max(ripley_values) if ripley_values else None,
                }
        return best

    def max_quadrat_vmr(region: Dict[str, Any]) -> Optional[float]:
        values = []
        for stats in (region.get("by_class") or {}).values():
            if stats.get("status") == "computed":
                value = _safe_float(stats.get("quadrat_variance_to_mean_ratio"))
                if value is not None:
                    values.append(value)
        return max(values) if values else None

    def max_ripley_strength(region: Dict[str, Any]) -> Optional[float]:
        values = []
        for stats in (region.get("by_class") or {}).values():
            if stats.get("status") != "computed":
                continue
            for row in stats.get("ripley_l_by_radius_no_edge_correction", []):
                value = _safe_float(row.get("ripley_l_minus_r_no_edge_correction"))
                if value is not None:
                    values.append(value)
        return max(values) if values else None

    def nni_value(region: Dict[str, Any]) -> Optional[float]:
        return _safe_float(best_class_by_nni(region).get("nearest_neighbour_index"))

    def clustered_cell_value(region: Dict[str, Any]) -> Optional[float]:
        # Higher value means stronger visible clustering. NNI below 1 implies clustering,
        # so convert it to a positive "clustering strength" for intuitive colour scaling.
        nni = nni_value(region)
        if nni is None:
            return None
        return max(0.0, 1.0 - nni)

    overlays = {
        "clustered_cell_roi": {
            "title": "Dominant Clustered Cell-Type ROI Overlay",
            "unit": "1 - lowest class-specific NNI",
            "cmap": "plasma",
            "value": clustered_cell_value,
            "label": lambda region, value: (
                f"clustered {best_class_by_nni(region).get('class_name') or 'cell'} ROI"
                if value is not None else "insufficient point-pattern data"
            ),
        },
        "nni_heatmap": {
            "title": "Lowest Class-Specific Nearest-Neighbour Index by ROI",
            "unit": "NNI; lower means more clustered",
            "cmap": "viridis_r",
            "value": nni_value,
            "label": lambda region, value: (
                "high clustering / low NNI" if value is not None and value < 0.7
                else "moderate clustering / NNI" if value is not None and value < 0.9
                else "low clustering / NNI" if value is not None
                else "insufficient point-pattern data"
            ),
        },
        "quadrat_vmr_heatmap": {
            "title": "Maximum Quadrat Variance-to-Mean Ratio by ROI",
            "unit": "VMR; higher means more hotspot-like",
            "cmap": "magma",
            "value": max_quadrat_vmr,
            "label": lambda region, value: (
                "high quadrat heterogeneity" if value is not None and value > 1.5
                else "moderate quadrat heterogeneity" if value is not None and value > 0.75
                else "low quadrat heterogeneity" if value is not None
                else "insufficient point-pattern data"
            ),
        },
        "ripley_clustering_strength": {
            "title": "Maximum Ripley-Style Clustering Strength by ROI",
            "unit": "max L(r)-r; positive means clustered",
            "cmap": "inferno",
            "value": max_ripley_strength,
            "label": lambda region, value: (
                "high Ripley-style clustering" if value is not None and value > 0
                else "no positive Ripley-style clustering" if value is not None
                else "insufficient point-pattern data"
            ),
        },
    }

    class_colours = {
        "Neoplastic": "#E53935",
        "Inflammatory": "#1E88E5",
        "Connective": "#FB8C00",
        "Epithelial": "#43A047",
        "Lymphocyte": "#3949AB",
        "Neutrophil": "#00ACC1",
        "Plasma": "#5E35B1",
        "Eosinophil": "#D81B60",
        "Unknown": "#757575",
    }

    outputs = {}
    for metric, config in overlays.items():
        db_path = os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_{metric}.db")
        if os.path.exists(db_path) and not overwrite:
            raise FileExistsError(f"Point-pattern overlay already exists: {db_path}")
        if os.path.exists(db_path):
            os.remove(db_path)

        raw_values = [config["value"](region) for region in per_region]
        numeric_values = [float(value) for value in raw_values if value is not None]
        value_min = min(numeric_values) if numeric_values else 0.0
        value_max = max(numeric_values) if numeric_values else 1.0
        if value_max <= value_min:
            value_max = value_min + 1.0
        norm = Normalize(vmin=value_min, vmax=value_max)
        cmap = colormaps[config["cmap"]]

        annotations, keys = [], []
        for index, (region, value) in enumerate(zip(per_region, raw_values, strict=False)):
            best = best_class_by_nni(region)
            dominant_clustered_class = best.get("class_name") or "Unknown"
            if metric == "clustered_cell_roi":
                colour = class_colours.get(str(dominant_clustered_class), class_colours["Unknown"])
                fill_opacity = 0.30 if value is not None else 0.08
            else:
                colour = to_hex(cmap(norm(float(value))) if value is not None else (0.7, 0.7, 0.7, 1.0), keep_alpha=False)
                fill_opacity = 0.35 if value is not None else 0.08

            label = config["label"](region, value)
            geometry = box(
                float(region["x_min"]) / x_scale,
                float(region["y_min"]) / y_scale,
                float(region["x_max"]) / x_scale,
                float(region["y_max"]) / y_scale,
            )
            properties = {
                "type": label,
                "label": label,
                "region_id": region.get("region_id", f"R{index + 1}"),
                "point_pattern_metric": metric,
                "point_pattern_value": float(value) if value is not None else None,
                "point_pattern_unit": config["unit"],
                "dominant_clustered_class": dominant_clustered_class,
                "dominant_clustered_class_nni": best.get("nearest_neighbour_index"),
                "dominant_clustered_class_quadrat_vmr": best.get("quadrat_vmr"),
                "dominant_clustered_class_ripley_strength": best.get("ripley_strength"),
                "preferred_radius": preferred_radius,
                "cell_count": int(region.get("cell_count", 0)),
                "color": colour,
                "colour": colour,
                "fill_color": colour,
                "line_color": colour,
                "fill_opacity": fill_opacity,
                "coordinate_space": "baseline",
                "source": "KongNet point-pattern spatial statistics",
            }
            annotations.append(Annotation(geometry, properties=properties))
            keys.append(f"{metric}_{properties['region_id']}")

        store = SQLiteStore(db_path)
        try:
            store.append_many(annotations, keys=keys)
            store.commit()
            annotation_count = len(store)
        finally:
            store.close()

        outputs[metric] = {
            "annotationstore_path": db_path,
            "region_count": annotation_count,
            "minimum": value_min,
            "maximum": value_max,
            "unit": config["unit"],
            "title": config["title"],
        }

    manifest_path = os.path.join(output_dir, "kongnet_point_pattern_overlays_manifest.json")
    _write_json(manifest_path, {
        "point_pattern_json_path": point_pattern_json_path,
        "region_count": len(per_region),
        "distance_units": units,
        "radii": radii,
        "overlays": outputs,
        "interpretation": {
            "clustered_cell_roi": "Categorical ROI overlay coloured by the cell type with the lowest class-specific nearest-neighbour index.",
            "nni_heatmap": "Lower NNI values indicate stronger local clustering.",
            "quadrat_vmr_heatmap": "Higher VMR values indicate more uneven, hotspot-like counts across quadrats.",
            "ripley_clustering_strength": "Positive L(r)-r values indicate clustering at at least one analysed radius.",
        },
        "clinical_warning": "Point-pattern overlays visualize model-derived exploratory spatial statistics, not diagnoses.",
    })

    overlays_dir = os.path.abspath(output_dir)
    slides_dir = os.path.dirname(os.path.abspath(wsi_path)) if wsi_path else "<SLIDES_DIRECTORY>"
    tiaviz_command = f'tiatoolbox visualize --slides "{slides_dir}" --overlays "{overlays_dir}"'
    return "\n".join([
        "KongNet point-pattern visual overlays generated.",
        f"Regions visualized: {len(per_region)}",
        "Overlays: clustered-cell ROI, NNI heatmap, quadrat VMR heatmap, Ripley clustering-strength",
        f"Output directory: {overlays_dir}",
        f"Manifest: {manifest_path}",
        "Open in TIAViz with:",
        tiaviz_command,
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
    point_pattern_json_path: Optional[str] = None,
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
    point_pattern_data = load_optional(point_pattern_json_path, "Point-pattern statistics JSON")

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

    lines += ["", "6b. Point-Pattern Spatial Statistics", "------------------------------------"]
    if point_pattern_data:
        pointpats_info = point_pattern_data.get("pointpats", {})
        lines.append(
            f"- pointpats available in this environment: {bool(pointpats_info.get('available'))}"
        )
        lines.append(
            "- Purpose: these statistics test whether each predicted cell population appears clustered, dispersed, or close to random spatial organisation."
        )
        by_class = point_pattern_data.get("whole_slide_by_class", {})
        for class_name, stats in by_class.items():
            if stats.get("status") != "computed":
                lines.append(f"- {class_name}: insufficient points for stable point-pattern statistics.")
                continue
            nni = _safe_float(stats.get("nearest_neighbour_index"))
            vmr = _safe_float(stats.get("quadrat_variance_to_mean_ratio"))
            nni_text = f"{nni:.3f}" if nni is not None else "unavailable"
            vmr_text = f"{vmr:.3f}" if vmr is not None else "unavailable"
            lines.append(
                f"- {class_name}: NNI {nni_text} ({stats.get('nearest_neighbour_interpretation')}); "
                f"quadrat VMR {vmr_text} ({stats.get('quadrat_interpretation')})."
            )
        cross_stats = point_pattern_data.get("tumour_or_epithelial_immune_cross_type", {}).get("statistics", {})
        if cross_stats.get("status") == "computed":
            best_ratio = None
            best_radius = None
            for row in cross_stats.get("by_radius", []):
                ratio = _safe_float(row.get("observed_to_expected_ratio"))
                if ratio is not None and (best_ratio is None or ratio > best_ratio):
                    best_ratio = ratio
                    best_radius = row.get("radius")
            if best_ratio is not None:
                lines.append(
                    f"- Strongest tumour/epithelial-to-immune proximity ratio: {best_ratio:.3f} "
                    f"at radius {best_radius} {point_pattern_data.get('parameters', {}).get('distance_units', '')}."
                )
        lines.append(
            "Interpretation: values below 1 for nearest-neighbour index suggest clustering; values above 1 for quadrat VMR suggest uneven hotspot-like organisation."
        )
    else:
        lines.append("- Point-pattern statistics were not supplied.")

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
    point_pattern_radii: Optional[List[float]] = None,
    cooccurrence_cell_types: Optional[List[str]] = None,
    neighbourhood_source_types: Optional[List[str]] = None,
    neighbourhood_target_types: Optional[List[str]] = None,
    point_pattern_cell_types: Optional[List[str]] = None,
    point_pattern_source_types: Optional[List[str]] = None,
    point_pattern_target_types: Optional[List[str]] = None,
    region_size: Optional[float] = 100.0,
    min_cells_per_region: int = 1,
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
    if neighbourhood_radius <= 0:
        raise ValueError("neighbourhood_radius must be greater than 0.")
    if point_pattern_radii is None:
        point_pattern_radii = [25.0, neighbourhood_radius, neighbourhood_radius * 2.0]
    point_pattern_radii = [float(radius) for radius in point_pattern_radii]
    if not point_pattern_radii or any(radius <= 0 for radius in point_pattern_radii):
        raise ValueError("point_pattern_radii must contain at least one positive radius.")
    if region_size is not None and float(region_size) <= 0:
        raise ValueError("region_size must be greater than 0 or omitted for automatic sizing.")
    if min_cells_per_region < 1 or community_count < 1:
        raise ValueError("min_cells_per_region and community_count must be at least 1.")

    os.makedirs(output_dir, exist_ok=True)
    slide_prefix = _slide_stem_prefix(wsi_path)
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
        "point_pattern_json": os.path.join(output_dir, "kongnet_point_pattern_statistics.json"),
        "point_pattern_txt": os.path.join(output_dir, "kongnet_point_pattern_statistics.txt"),
        "regions_db": os.path.join(output_dir, f"{slide_prefix}kongnet_region_boundaries.db"),
        "cell_neighbourhoods_csv": os.path.join(output_dir, "kongnet_cell_neighbourhoods.csv"),
        "communities_json": os.path.join(output_dir, "kongnet_spatial_communities.json"),
        "rankings_json": os.path.join(output_dir, "kongnet_region_rankings.json"),
        "rankings_txt": os.path.join(output_dir, "kongnet_region_rankings.txt"),
        "slide_summary_json": os.path.join(output_dir, "kongnet_slide_summary.json"),
        "slide_summary_txt": os.path.join(output_dir, "kongnet_slide_summary.txt"),
        "heatmaps_manifest_json": os.path.join(output_dir, "kongnet_heatmaps_manifest.json"),
        "density_heatmap_db": os.path.join(output_dir, f"{slide_prefix}kongnet_density_heatmap.db"),
        "inflammatory_heatmap_db": os.path.join(output_dir, f"{slide_prefix}kongnet_inflammatory_heatmap.db"),
        "interaction_heatmap_db": os.path.join(output_dir, f"{slide_prefix}kongnet_tumour_immune_interaction_heatmap.db"),
        "point_pattern_overlays_manifest_json": os.path.join(output_dir, "kongnet_point_pattern_overlays_manifest.json"),
        "point_pattern_clustered_cell_roi_db": os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_clustered_cell_roi.db"),
        "point_pattern_nni_heatmap_db": os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_nni_heatmap.db"),
        "point_pattern_quadrat_vmr_heatmap_db": os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_quadrat_vmr_heatmap.db"),
        "point_pattern_ripley_clustering_strength_db": os.path.join(output_dir, f"{slide_prefix}kongnet_point_pattern_ripley_clustering_strength.db"),
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
        source_types=neighbourhood_source_types,
        target_types=neighbourhood_target_types,
        **spatial_kwargs,
    )
    step_results["cooccurrence"] = tool_compute_cell_type_cooccurrence(
        annotationstore_path=annotationstore_path,
        output_json_path=paths["cooccurrence_json"],
        output_csv_path=paths["cooccurrence_csv"],
        radius=neighbourhood_radius,
        min_probability=min_probability,
        cell_types=cooccurrence_cell_types,
        **spatial_kwargs,
    )
    step_results["nearest_neighbours"] = tool_compute_nearest_neighbour_features(
        annotationstore_path=annotationstore_path,
        output_csv_path=paths["nearest_csv"],
        output_json_path=paths["nearest_json"],
        min_probability=min_probability,
        source_types=neighbourhood_source_types,
        target_types=neighbourhood_target_types,
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
    step_results["point_pattern_statistics"] = tool_compute_kongnet_point_pattern_statistics(
        annotationstore_path=annotationstore_path,
        output_json_path=paths["point_pattern_json"],
        output_txt_path=paths["point_pattern_txt"],
        regions_json_path=paths["regions_json"],
        radii=point_pattern_radii,
        min_probability=min_probability,
        cell_types=point_pattern_cell_types,
        source_types=point_pattern_source_types,
        target_types=point_pattern_target_types,
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
    step_results["point_pattern_overlays"] = tool_generate_kongnet_point_pattern_overlays(
        point_pattern_json_path=paths["point_pattern_json"],
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
        point_pattern_json_path=paths["point_pattern_json"],
        output_report_path=paths["report_txt"],
    )
    step_results["interpretability_report"] = f"Saved plain-text report ({len(report)} characters)."

    with open(paths["regions_json"], "r", encoding="utf-8") as regions_file:
        resolved_regions = json.load(regions_file)
    manifest = {
        "workflow": "full_kongnet_spatial_workflow",
        "status": "completed",
        "annotationstore_path": annotationstore_path,
        "wsi_path": wsi_path,
        "parameters": {
            "mpp": mpp,
            "min_probability": min_probability,
            "neighbourhood_radius": neighbourhood_radius,
            "point_pattern_radii": point_pattern_radii,
            "cooccurrence_cell_types": cooccurrence_cell_types,
            "neighbourhood_source_types": neighbourhood_source_types,
            "neighbourhood_target_types": neighbourhood_target_types,
            "point_pattern_cell_types": point_pattern_cell_types,
            "point_pattern_source_types": point_pattern_source_types,
            "point_pattern_target_types": point_pattern_target_types,
            "region_size_requested": region_size if region_size is not None else "auto",
            "region_size_resolved": resolved_regions.get("region_size"),
            "region_size_strategy": resolved_regions.get("region_size_strategy"),
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


# -----------------------------------------------------------------------------
# Cross-model common spatial representation
# -----------------------------------------------------------------------------

COMMON_SPATIAL_MODEL_FAMILIES = {
    "kongnet": "object",
    "hovernet": "object",
    "hovernetplus": "object_and_region",
    "semantic_segmentation": "region",
    "patch_classification": "region",
}

COMMON_SPATIAL_STATISTIC_ALIASES = {
    "entropy": "spatial_entropy",
    "shannon_entropy": "spatial_entropy",
    "roi_entropy": "spatial_entropy",
    "moran": "morans_i",
    "moran_i": "morans_i",
    "point_pattern": "point_pattern_statistics",
    "point_pattern_analysis": "point_pattern_statistics",
    "ripley_k": "ripley",
    "ripley_l": "ripley",
    "cross_g_function": "cross_g",
    "cell_type_cooccurrence": "cooccurrence",
    "cells_within_radius": "radius_neighbourhood",
    "neighbourhood": "radius_neighbourhood",
    "nearest_neighbor": "nearest_neighbour",
}

COMMON_IMPLEMENTED_STATISTICS = [
    "morans_i", "spatial_entropy", "nni", "ripley", "quadrat_vmr",
    "point_pattern_statistics", "cross_g", "nearest_neighbour",
    "cooccurrence", "radius_neighbourhood",
]


def _common_model_class_dict(model_family: str, model_name: str) -> Dict[int, str]:
    if model_family == "kongnet":
        return dict(KONGNET_MODEL_CATALOG.get(model_name, {}).get("class_dict", {}))
    if model_family == "hovernet":
        return dict(NUCLEUS_INSTANCE_SEGMENTATION_MODEL_CATALOG.get(model_name, {}).get("class_dict", {}))
    if model_family == "semantic_segmentation":
        return dict(SEMANTIC_SEGMENTATION_MODEL_CATALOG.get(model_name, {}).get("class_dict", {}))
    if model_family == "patch_classification":
        return dict(PATCH_PREDICTION_MODEL_CATALOG.get(model_name, {}).get("class_dict", {}))
    if model_family == "hovernetplus":
        metadata = MULTI_TASK_SEGMENTATION_MODEL_CATALOG.get(model_name, {})
        return dict(metadata.get("nuclear_class_dict", {}))
    return {}


def _annotation_class_and_probability(properties: Dict[str, Any], class_dict: Dict[int, str]) -> tuple[str, float]:
    raw_class = properties.get("class_name", properties.get("type", properties.get("label", "Unknown")))
    class_name = class_dict.get(raw_class, str(raw_class)) if isinstance(raw_class, int) else str(raw_class)
    try:
        probability = float(properties.get("probability", properties.get("prob", properties.get("score", 1.0))))
    except (TypeError, ValueError):
        probability = 1.0
    return class_name, probability


def tool_build_common_spatial_features(
    source_path: str,
    model_family: str,
    model_name: str,
    output_dir: str,
    region_size: float = 256.0,
    min_probability: float = 0.0,
    distance_units: str = "pixels",
    wsi_path: Optional[str] = None,
    mpp: Optional[float] = None,
    min_recommended_rois: int = 9,
) -> str:
    """Convert model-specific AnnotationStore/CSV outputs into common spatial tables.

    Object models are aggregated by centroid. Region, semantic, and patch models
    are aggregated by geometry intersection area when an AnnotationStore is used.
    The resulting JSON deliberately mirrors the existing KongNet ROI structure so
    generalized ROI statistics can consume it without model-specific branches.
    """
    from collections import defaultdict
    from shapely.geometry import box

    family = str(model_family or "").strip().lower()
    if family not in COMMON_SPATIAL_MODEL_FAMILIES:
        raise ValueError(f"model_family must be one of: {sorted(COMMON_SPATIAL_MODEL_FAMILIES)}")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Model output not found: {source_path}")
    if float(region_size) <= 0:
        raise ValueError("region_size must be positive.")
    if not 0.0 <= float(min_probability) <= 1.0:
        raise ValueError("min_probability must be between 0 and 1.")
    if int(min_recommended_rois) < 3:
        raise ValueError("min_recommended_rois must be at least 3.")

    scale = _resolve_spatial_scale(distance_units, wsi_path=wsi_path, mpp=mpp)
    class_dict = _common_model_class_dict(family, model_name)
    os.makedirs(output_dir, exist_ok=True)
    objects_path = os.path.join(output_dir, "spatial_objects.csv")
    features_path = os.path.join(output_dir, "spatial_roi_features.csv")
    common_json_path = os.path.join(output_dir, "common_spatial_features.json")
    capabilities_path = os.path.join(output_dir, "spatial_capabilities.json")

    records = []
    is_csv = os.path.splitext(source_path)[1].lower() == ".csv"
    if is_csv:
        with open(source_path, "r", encoding="utf-8", newline="") as file:
            for index, row in enumerate(csv.DictReader(file), start=1):
                probability = float(row.get("probability") or row.get("prob") or row.get("score") or 1.0)
                if probability < float(min_probability):
                    continue
                x_px = float(row.get("x_px", row.get("x", row.get("centroid_x"))))
                y_px = float(row.get("y_px", row.get("y", row.get("centroid_y"))))
                raw_csv_class = row.get("class_name") or row.get("type") or row.get("label") or "Unknown"
                try:
                    numeric_csv_class = int(raw_csv_class)
                except (TypeError, ValueError):
                    numeric_csv_class = None
                records.append({
                    "object_id": str(row.get("object_id") or row.get("annotation_id") or index),
                    "class_name": class_dict.get(numeric_csv_class, str(raw_csv_class)) if numeric_csv_class is not None else str(raw_csv_class),
                    "probability": probability,
                    "x_px": x_px,
                    "y_px": y_px,
                    "area_px2": float(row.get("area") or 0.0),
                    "geometry": None,
                })
    else:
        from tiatoolbox.annotation.storage import SQLiteStore
        store = SQLiteStore(source_path)
        try:
            for annotation_id, annotation in store.items():
                properties = dict(annotation.properties or {})
                class_name, probability = _annotation_class_and_probability(properties, class_dict)
                if probability < float(min_probability) or annotation.geometry is None:
                    continue
                centroid = annotation.geometry.centroid
                records.append({
                    "object_id": str(annotation_id),
                    "class_name": class_name,
                    "probability": probability,
                    "x_px": float(centroid.x),
                    "y_px": float(centroid.y),
                    "area_px2": float(annotation.geometry.area),
                    "geometry": annotation.geometry,
                })
        finally:
            store.close()

    if not records:
        raise ValueError("No annotations remained after filtering.")

    for record in records:
        record["x"] = record["x_px"] * scale["x_scale"]
        record["y"] = record["y_px"] * scale["y_scale"]
        record["area"] = record["area_px2"] * scale["x_scale"] * scale["y_scale"]

    min_x = min(record["x"] for record in records)
    min_y = min(record["y"] for record in records)
    max_x = max(record["x"] for record in records)
    max_y = max(record["y"] for record in records)
    size = float(region_size)
    aggregation_mode = "centroid_count" if COMMON_SPATIAL_MODEL_FAMILIES[family] == "object" else "geometry_area"

    region_data: Dict[tuple[int, int], Dict[str, Any]] = defaultdict(
        lambda: {"class_counts": defaultdict(float), "class_areas": defaultdict(float), "probabilities": defaultdict(list)}
    )
    for record in records:
        gx = int(math.floor((record["x"] - min_x) / size))
        gy = int(math.floor((record["y"] - min_y) / size))
        if aggregation_mode == "centroid_count" or record["geometry"] is None:
            bucket = region_data[(gx, gy)]
            bucket["class_counts"][record["class_name"]] += 1.0
            bucket["class_areas"][record["class_name"]] += record["area"]
            bucket["probabilities"][record["class_name"]].append(record["probability"])
            continue

        geometry = record["geometry"]
        scaled_bounds = (
            geometry.bounds[0] * scale["x_scale"], geometry.bounds[1] * scale["y_scale"],
            geometry.bounds[2] * scale["x_scale"], geometry.bounds[3] * scale["y_scale"],
        )
        gx0 = int(math.floor((scaled_bounds[0] - min_x) / size)); gx1 = int(math.floor((scaled_bounds[2] - min_x) / size))
        gy0 = int(math.floor((scaled_bounds[1] - min_y) / size)); gy1 = int(math.floor((scaled_bounds[3] - min_y) / size))
        for cell_x in range(gx0, gx1 + 1):
            for cell_y in range(gy0, gy1 + 1):
                pixel_box = box(
                    (min_x + cell_x * size) / scale["x_scale"],
                    (min_y + cell_y * size) / scale["y_scale"],
                    (min_x + (cell_x + 1) * size) / scale["x_scale"],
                    (min_y + (cell_y + 1) * size) / scale["y_scale"],
                )
                intersection_area = float(geometry.intersection(pixel_box).area) * scale["x_scale"] * scale["y_scale"]
                if intersection_area <= 0:
                    continue
                bucket = region_data[(cell_x, cell_y)]
                bucket["class_counts"][record["class_name"]] += 1.0
                bucket["class_areas"][record["class_name"]] += intersection_area
                bucket["probabilities"][record["class_name"]].append(record["probability"])

    object_fields = ["object_id", "model_family", "model_name", "class_name", "probability", "x", "y", "area", "distance_units"]
    with open(objects_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=object_fields); writer.writeheader()
        for record in records:
            writer.writerow({key: value for key, value in {
                "object_id": record["object_id"], "model_family": family, "model_name": model_name,
                "class_name": record["class_name"], "probability": record["probability"],
                "x": record["x"], "y": record["y"], "area": record["area"], "distance_units": scale["units"],
            }.items()})

    all_classes = sorted({record["class_name"] for record in records})
    regions = []
    long_rows = []
    for index, ((gx, gy), data) in enumerate(sorted(region_data.items()), start=1):
        x_min = min_x + gx * size; y_min = min_y + gy * size; area = size * size
        counts = {name: float(data["class_counts"].get(name, 0.0)) for name in all_classes}
        areas = {name: float(data["class_areas"].get(name, 0.0)) for name in all_classes}
        total_count = sum(counts.values()); total_covered_area = sum(areas.values())
        feature_values = {
            "object_count": total_count,
            "object_density": total_count / area,
            "covered_area": total_covered_area,
            "covered_area_percentage": total_covered_area / area * 100.0,
        }
        class_percentages = {}
        for class_name in sorted(counts):
            denominator = total_covered_area if aggregation_mode == "geometry_area" else total_count
            numerator = areas[class_name] if aggregation_mode == "geometry_area" else counts[class_name]
            class_percentages[class_name] = numerator / denominator * 100.0 if denominator > 0 else 0.0
            feature_values[f"{class_name}_count"] = counts[class_name]
            feature_values[f"{class_name}_percentage"] = class_percentages[class_name]
            probs = data["probabilities"][class_name]
            feature_values[f"{class_name}_mean_probability"] = sum(probs) / len(probs) if probs else None
        region = {
            "region_id": f"R{index}", "grid_x": gx, "grid_y": gy,
            "x_min": x_min, "y_min": y_min, "x_max": x_min + size, "y_max": y_min + size,
            "area_square_units": area, "cell_count": total_count, "cell_density_per_square_unit": total_count / area,
            "class_counts": counts, "class_percentages": class_percentages, **feature_values,
        }
        regions.append(region)
        for feature_name, feature_value in feature_values.items():
            long_rows.append({
                "region_id": region["region_id"], "model_family": family, "model_name": model_name,
                "grid_x": gx, "grid_y": gy, "x_min": x_min, "y_min": y_min,
                "x_max": x_min + size, "y_max": y_min + size, "area": area,
                "feature_name": feature_name, "feature_value": feature_value,
            })

    with open(features_path, "w", encoding="utf-8", newline="") as file:
        fields = ["region_id", "model_family", "model_name", "grid_x", "grid_y", "x_min", "y_min", "x_max", "y_max", "area", "feature_name", "feature_value"]
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(long_rows)

    object_statistics = ["nni", "ripley", "quadrat_vmr", "point_pattern_statistics", "cross_g", "nearest_neighbour", "cooccurrence", "radius_neighbourhood"]
    roi_statistics = ["morans_i", "spatial_entropy", "hotspot_ranking"]
    data_compatible = (object_statistics if family in {"kongnet", "hovernet", "hovernetplus"} else []) + roi_statistics
    warnings = []
    if len(regions) < int(min_recommended_rois):
        warnings.append(
            f"Only {len(regions)} ROIs were generated; at least {int(min_recommended_rois)} are recommended for ROI-level spatial inference. Reduce region_size and rebuild."
        )
    capabilities = {
        "model_family": family, "model_name": model_name, "source_path": source_path,
        "representation": COMMON_SPATIAL_MODEL_FAMILIES[family],
        "has_individual_objects": family in {"kongnet", "hovernet", "hovernetplus"},
        "has_object_coordinates": family in {"kongnet", "hovernet", "hovernetplus"},
        "has_class_labels": any(record["class_name"] != "Unknown" for record in records),
        "has_roi_features": bool(regions), "has_numeric_roi_features": bool(long_rows),
        "has_physical_scale": scale["units"] == "microns", "distance_units": scale["units"],
        "available_classes": sorted({record["class_name"] for record in records}),
        "available_roi_features": sorted({row["feature_name"] for row in long_rows}),
        "roi_count": len(regions), "minimum_recommended_rois": int(min_recommended_rois),
        "warnings": warnings,
        "data_compatible_statistics": data_compatible,
        "implemented_statistics": [name for name in data_compatible if name in COMMON_IMPLEMENTED_STATISTICS],
        "compatible_statistics": data_compatible,
    }
    payload = {
        "format": "common_spatial_features_v1", "source_path": source_path, "model_family": family,
        "model_name": model_name, "aggregation_mode": aggregation_mode, "region_size": size,
        "distance_units": scale["units"], "scale": scale, "regions": regions,
        "outputs": {"objects_csv": objects_path, "roi_features_csv": features_path, "capabilities_json": capabilities_path},
    }
    _write_json(common_json_path, payload); _write_json(capabilities_path, capabilities)
    return "\n".join([
        "Common spatial feature conversion completed.", f"Model: {model_name} ({family})",
        f"Annotations: {len(records)}", f"ROIs: {len(regions)}", f"Common JSON: {common_json_path}",
        f"Objects CSV: {objects_path}", f"ROI features CSV: {features_path}", f"Capabilities: {capabilities_path}",
    ])


def tool_validate_spatial_capabilities(
    capabilities_json_path: str,
    statistic: str,
    feature_name: Optional[str] = None,
    source_types: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
) -> str:
    """Validate that adapted model output can support a requested statistic."""
    if not os.path.exists(capabilities_json_path):
        raise FileNotFoundError(f"Capabilities JSON not found: {capabilities_json_path}")
    with open(capabilities_json_path, "r", encoding="utf-8") as file:
        capabilities = json.load(file)
    requested_raw = str(statistic).strip().lower()
    requested = COMMON_SPATIAL_STATISTIC_ALIASES.get(requested_raw, requested_raw)
    errors = []
    if requested not in set(capabilities.get("data_compatible_statistics") or capabilities.get("compatible_statistics") or []):
        errors.append(f"Statistic '{requested}' is incompatible with this representation.")
    elif requested not in set(capabilities.get("implemented_statistics") or []):
        errors.append(f"Statistic '{requested}' is data-compatible but no generalized execution tool is implemented.")
    if feature_name and feature_name not in set(capabilities.get("available_roi_features") or []):
        errors.append(f"ROI feature '{feature_name}' is unavailable.")
    classes = set(capabilities.get("available_classes") or [])
    for label, values in (("source", source_types), ("target", target_types)):
        missing = sorted(set(values or []) - classes)
        if missing:
            errors.append(f"Unknown {label} classes: {missing}.")
    result = {
        "valid": not errors, "requested_statistic": requested_raw, "statistic": requested, "feature_name": feature_name,
        "errors": errors, "model_family": capabilities.get("model_family"),
        "model_name": capabilities.get("model_name"),
    }
    return json.dumps(result, indent=2)


def tool_compute_common_roi_morans_i(
    common_spatial_json_path: str,
    output_json_path: str,
    output_txt_path: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    weights_method: str = "queen",
    k_neighbours: int = 4,
    distance_threshold: Optional[float] = None,
    permutations: int = 999,
    alpha: float = 0.05,
    min_rois: int = 9,
) -> str:
    """Compute Moran's I for any model adapted to common_spatial_features_v1."""
    import numpy as np
    if not os.path.exists(common_spatial_json_path):
        raise FileNotFoundError(f"Common spatial JSON not found: {common_spatial_json_path}")
    with open(common_spatial_json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("format") != "common_spatial_features_v1":
        raise ValueError("Input is not common_spatial_features_v1.")
    regions = list(payload.get("regions") or [])
    if len(regions) < int(min_rois):
        raise ValueError(
            f"Moran's I requires at least {int(min_rois)} ROIs under the configured safeguard; "
            f"only {len(regions)} are available. Reduce the common-grid region_size and rebuild."
        )
    if not metrics:
        raise ValueError("At least one numeric ROI metric must be selected.")
    if not 0 < float(alpha) < 1 or int(permutations) < 0:
        raise ValueError("alpha must be between 0 and 1 and permutations must be non-negative.")

    coordinates = np.asarray([
        [(float(r["x_min"]) + float(r["x_max"])) / 2, (float(r["y_min"]) + float(r["y_max"])) / 2]
        for r in regions
    ], dtype=float)
    n = len(regions); weights = np.zeros((n, n), dtype=float); method = str(weights_method).lower()
    if method in {"queen", "rook"}:
        for i, left in enumerate(regions):
            for j, right in enumerate(regions):
                if i == j: continue
                dx = abs(int(left["grid_x"]) - int(right["grid_x"])); dy = abs(int(left["grid_y"]) - int(right["grid_y"]))
                weights[i, j] = float((dx + dy == 1) if method == "rook" else (max(dx, dy) == 1))
    else:
        distances = np.sqrt(np.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(distances, np.inf)
        if method == "knn":
            k = min(max(1, int(k_neighbours)), n - 1)
            for i in range(n): weights[i, np.argsort(distances[i])[:k]] = 1.0
        elif method == "distance":
            if distance_threshold is None or float(distance_threshold) <= 0:
                raise ValueError("A positive distance_threshold is required for distance weights.")
            weights[distances <= float(distance_threshold)] = 1.0
        else:
            raise ValueError("weights_method must be queen, rook, knn, or distance.")
    row_sums = weights.sum(axis=1)
    if np.any(row_sums == 0):
        raise ValueError("At least one ROI has no neighbours under the selected spatial weights.")
    weights = weights / row_sums[:, None]

    def moran_value(values):
        deviations = values - np.mean(values); denominator = float(np.sum(deviations ** 2)); total_weight = float(np.sum(weights))
        return float((len(values) / total_weight) * (np.sum(weights * deviations[:, None] * deviations[None, :]) / denominator))

    rng = np.random.default_rng(0); results = {}; expected_i = -1.0 / (n - 1)
    for metric in metrics:
        values = [_roi_moran_feature_value(region, metric) for region in regions]
        if any(value is None for value in values):
            results[metric] = {"status": "skipped", "reason": "Feature is missing or non-numeric in at least one ROI."}; continue
        array = np.asarray(values, dtype=float)
        if float(np.var(array)) <= 0:
            results[metric] = {"status": "skipped", "reason": "Feature has zero variance across ROIs."}; continue
        observed = moran_value(array)
        simulated = [moran_value(rng.permutation(array)) for _ in range(int(permutations))]
        extreme = sum(abs(value - expected_i) >= abs(observed - expected_i) for value in simulated)
        p_value = (extreme + 1) / (len(simulated) + 1) if simulated else None
        results[metric] = {
            "status": "computed", "source": "transparent NumPy Moran's I fallback", "roi_count": n,
            "moran_i": observed, "expected_i": expected_i, "permutation_p_value": p_value,
            "significant": bool(p_value is not None and p_value < float(alpha)),
            "interpretation": _moran_interpretation(observed, p_value, float(alpha)),
        }
    output = {
        "analysis": "common_roi_morans_i", "common_spatial_json_path": common_spatial_json_path,
        "model_family": payload.get("model_family"), "model_name": payload.get("model_name"),
        "parameters": {"metrics": metrics, "weights_method": method, "k_neighbours": k_neighbours,
                       "distance_threshold": distance_threshold, "permutations": permutations, "alpha": alpha},
        "results": results,
    }
    _write_json(output_json_path, output)
    if output_txt_path:
        ensure_parent_dir(output_txt_path)
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write("Cross-model ROI Moran's I\n===========================\n")
            for metric, result in results.items():
                file.write(f"\n{metric}: {json.dumps(result, ensure_ascii=False)}\n")
    return f"Cross-model Moran's I completed. JSON: {output_json_path}"


def tool_compute_common_roi_entropy(
    common_spatial_json_path: str,
    output_json_path: str,
    output_txt_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    cell_types: Optional[List[str]] = None,
    normalize: bool = True,
    entropy_base: float = math.e,
    low_threshold: float = 0.40,
    high_threshold: float = 0.70,
) -> str:
    """Compute class-composition entropy for any adapted model family."""
    return tool_compute_kongnet_spatial_entropy(
        regions_json_path=common_spatial_json_path, output_json_path=output_json_path,
        output_txt_path=output_txt_path, output_csv_path=output_csv_path,
        normalize=normalize, entropy_base=entropy_base, low_threshold=low_threshold,
        high_threshold=high_threshold, cell_types=cell_types,
    )


def _load_common_objects(common_spatial_json_path: str, min_probability: float = 0.0):
    import numpy as np
    if not os.path.exists(common_spatial_json_path):
        raise FileNotFoundError(f"Common spatial JSON not found: {common_spatial_json_path}")
    with open(common_spatial_json_path, "r", encoding="utf-8") as file:
        common = json.load(file)
    if common.get("format") != "common_spatial_features_v1":
        raise ValueError("Input is not common_spatial_features_v1.")
    objects_path = (common.get("outputs") or {}).get("objects_csv")
    if not objects_path or not os.path.exists(objects_path):
        raise FileNotFoundError("The common representation does not reference an existing spatial_objects.csv.")
    rows = []
    with open(objects_path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            probability = float(row.get("probability") or 1.0)
            if probability >= float(min_probability):
                rows.append(row)
    if not rows:
        raise ValueError("No common spatial objects remained after probability filtering.")
    coords = np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=float)
    classes = np.asarray([str(row["class_name"]) for row in rows], dtype=object)
    return common, rows, coords, classes


def _validate_common_selected_classes(classes, requested: Optional[List[str]], label: str) -> List[str]:
    available = list(dict.fromkeys(str(value) for value in classes))
    if not requested:
        return available
    lookup = {name.casefold(): name for name in available}
    resolved = []
    for value in requested:
        match = lookup.get(str(value).strip().casefold())
        if not match:
            raise ValueError(f"Unknown {label} class '{value}'. Available classes: {available}")
        if match not in resolved:
            resolved.append(match)
    return resolved


def tool_compute_common_point_pattern_statistics(
    common_spatial_json_path: str,
    output_json_path: str,
    output_txt_path: Optional[str] = None,
    cell_types: Optional[List[str]] = None,
    source_types: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
    radii: Optional[List[float]] = None,
    quadrat_grid_size: int = 4,
    min_points_per_pattern: int = 10,
    min_probability: float = 0.0,
) -> str:
    """Run NNI, quadrat VMR/chi-square, Ripley, and cross-type proximity on common objects."""
    import numpy as np
    common, rows, coords, classes = _load_common_objects(common_spatial_json_path, min_probability)
    selected = _validate_common_selected_classes(classes, cell_types, "point-pattern")
    sources = _validate_common_selected_classes(classes, source_types, "source") if source_types else []
    targets = _validate_common_selected_classes(classes, target_types, "target") if target_types else []
    if bool(sources) != bool(targets):
        raise ValueError("Provide both source_types and target_types for cross-type proximity, or neither.")
    radii = [float(value) for value in (radii or [])]
    if not radii or any(value <= 0 for value in radii):
        raise ValueError("radii must be a non-empty list of positive values.")
    min_xy = coords.min(axis=0); max_xy = coords.max(axis=0)
    area = float(max((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]), 1e-9))
    by_class = {
        name: _summarise_unmarked_point_pattern(coords[classes == name], area, radii, quadrat_grid_size, min_points_per_pattern)
        for name in selected
    }
    cross = None
    if sources and targets:
        cross = {
            "source_types": sources, "target_types": targets,
            "statistics": _summarise_cross_type_proximity(
                coords[np.isin(classes, sources)], coords[np.isin(classes, targets)], area, radii
            ),
        }
    output = {
        "analysis": "common_object_point_pattern_statistics", "common_spatial_json_path": common_spatial_json_path,
        "model_family": common.get("model_family"), "model_name": common.get("model_name"),
        "parameters": {"cell_types": selected, "source_types": sources, "target_types": targets, "radii": radii,
                       "distance_units": common.get("distance_units"), "quadrat_grid_size": quadrat_grid_size,
                       "min_points_per_pattern": min_points_per_pattern, "min_probability": min_probability},
        "object_count": len(rows), "analysed_area": area, "by_class": by_class, "cross_type_proximity": cross,
        "warning": "Ripley-style values are reported without edge correction.",
    }
    _write_json(output_json_path, output)
    if output_txt_path:
        ensure_parent_dir(output_txt_path)
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write("Common-object point-pattern statistics\n======================================\n\n")
            file.write(json.dumps(output, indent=2, ensure_ascii=False))
    return f"Common-object point-pattern statistics completed. JSON: {output_json_path}"


def tool_compute_common_cross_g(
    common_spatial_json_path: str,
    output_json_path: str,
    output_csv_path: Optional[str] = None,
    source_types: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
    radii: Optional[List[float]] = None,
    min_probability: float = 0.0,
) -> str:
    """Compute empirical source-to-target Cross-G from standardized common objects."""
    common, rows, coords, classes = _load_common_objects(common_spatial_json_path, min_probability)
    sources = _validate_common_selected_classes(classes, source_types, "source")
    targets = _validate_common_selected_classes(classes, target_types, "target")
    if not source_types or not target_types:
        raise ValueError("Explicit source_types and target_types are required.")
    radii = [float(value) for value in (radii or [])]
    if not radii or any(value <= 0 for value in radii):
        raise ValueError("radii must contain positive values.")
    min_xy = coords.min(axis=0); max_xy = coords.max(axis=0)
    area = float(max((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]), 1e-9))
    source_indices = [i for i, value in enumerate(classes) if value in sources]
    target_indices = [i for i, value in enumerate(classes) if value in targets]
    stats = _cross_g_curve(
        coords[source_indices], [rows[i]["object_id"] for i in source_indices],
        coords[target_indices], [rows[i]["object_id"] for i in target_indices], area, radii,
    )
    output = {
        "analysis": "common_object_cross_g", "model_family": common.get("model_family"),
        "model_name": common.get("model_name"), "source_types": sources, "target_types": targets,
        "distance_units": common.get("distance_units"), "radii": radii, "statistics": stats,
    }
    _write_json(output_json_path, output)
    if output_csv_path:
        ensure_parent_dir(output_csv_path)
        with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
            fields = ["radius", "empirical_cross_g", "csr_poisson_expected", "difference_from_expected", "interpretation"]
            writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(stats.get("curve", []))
    return f"Common-object Cross-G completed. JSON: {output_json_path}"


def tool_compute_common_cooccurrence(
    common_spatial_json_path: str,
    output_json_path: str,
    radius: float,
    cell_types: Optional[List[str]] = None,
    min_probability: float = 0.0,
) -> str:
    """Count undirected selected-class object pairs within a radius."""
    from collections import Counter
    from scipy.spatial import cKDTree
    common, rows, coords, classes = _load_common_objects(common_spatial_json_path, min_probability)
    selected = _validate_common_selected_classes(classes, cell_types, "co-occurrence")
    mask = [value in selected for value in classes]; filtered_coords = coords[mask]; filtered_classes = classes[mask]
    pairs = cKDTree(filtered_coords).query_pairs(float(radius)); counts = Counter()
    for first, second in pairs:
        counts["--".join(sorted((str(filtered_classes[first]), str(filtered_classes[second]))))] += 1
    output = {"analysis": "common_object_cooccurrence", "model_family": common.get("model_family"),
              "model_name": common.get("model_name"), "radius": float(radius), "distance_units": common.get("distance_units"),
              "cell_types": selected, "pair_count": len(pairs), "pair_counts": dict(counts)}
    _write_json(output_json_path, output)
    return f"Common-object co-occurrence completed. JSON: {output_json_path}"


def tool_compute_common_neighbour_distances(
    common_spatial_json_path: str,
    output_json_path: str,
    source_types: List[str],
    target_types: List[str],
    radius: Optional[float] = None,
    min_probability: float = 0.0,
) -> str:
    """Compute nearest target distance and optional within-radius counts for selected sources."""
    import numpy as np
    from scipy.spatial import cKDTree
    common, rows, coords, classes = _load_common_objects(common_spatial_json_path, min_probability)
    sources = _validate_common_selected_classes(classes, source_types, "source")
    targets = _validate_common_selected_classes(classes, target_types, "target")
    source_coords = coords[np.isin(classes, sources)]; target_coords = coords[np.isin(classes, targets)]
    if len(source_coords) < 1 or len(target_coords) < 1:
        raise ValueError("At least one source and target object are required.")
    distances, _ = cKDTree(target_coords).query(source_coords, k=1)
    output = {
        "analysis": "common_object_neighbour_distances", "model_family": common.get("model_family"),
        "model_name": common.get("model_name"), "source_types": sources, "target_types": targets,
        "distance_units": common.get("distance_units"), "source_count": len(source_coords), "target_count": len(target_coords),
        "mean_nearest_distance": float(np.mean(distances)), "median_nearest_distance": float(np.median(distances)),
        "minimum_nearest_distance": float(np.min(distances)), "maximum_nearest_distance": float(np.max(distances)),
    }
    if radius is not None:
        counts = np.asarray([len(cKDTree(target_coords).query_ball_point(point, float(radius))) for point in source_coords])
        output.update({"radius": float(radius), "sources_with_target_within_radius": int(np.sum(counts > 0)),
                       "mean_targets_within_radius": float(np.mean(counts)), "total_directed_links_within_radius": int(np.sum(counts))})
    _write_json(output_json_path, output)
    return f"Common-object neighbour analysis completed. JSON: {output_json_path}"
