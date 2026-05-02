"""
tia_tools.py
------------
MCP-only pathology agent tool logic.

Main MVP model:
- resnet18-kather100k patch classification via Hugging Face/timm

Core outputs:
- WSI metadata
- Thumbnail
- Tissue mask
- Patch extraction
- Kather patch predictions
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
import re
import csv
import json
import math
import random
import shutil
from collections import Counter, deque
from typing import Optional, Dict, Any, List


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
    stride: int = 448,
    level: int = 0,
    max_patches: int = 100,
    min_tissue_fraction: float = 0.10,
    mpp: float = 8.0,
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

    if model_name == "resnet18-kather100k":
        hf_model_name = "hf-hub:1aurent/resnet18.tiatoolbox-kather100k"

        model = timm.create_model(
            hf_model_name,
            pretrained=True,
        )

        model.eval()
        model.to(device)

        return model, {"source": "huggingface", "hf_model": hf_model_name}

    raise ValueError(f"Unsupported model_name: {model_name}")


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


def tool_predict_kather_resnet18(
    patch_dir: str,
    output_json_path: str,
    output_csv_path: Optional[str] = None,
    model_name: str = "resnet18-kather100k",
    batch_size: int = 16,
    device: str = "auto",
    input_size: int = 224,
) -> str:
    if not isinstance(patch_dir, str) or not os.path.isdir(patch_dir):
        raise ValueError("predict_kather_resnet18 requires a valid patch_dir.")

    if not isinstance(output_json_path, str) or not output_json_path.strip():
        raise ValueError("predict_kather_resnet18 requires output_json_path.")

    ensure_parent_dir(output_json_path)
    if output_csv_path:
        ensure_parent_dir(output_csv_path)

    import cv2
    import torch
    import numpy as np

    patch_files = sorted(
        [
            os.path.join(patch_dir, f)
            for f in os.listdir(patch_dir)
            if f.lower().endswith(".png")
        ],
        key=lambda p: p.lower()
    )

    if not patch_files:
        raise FileNotFoundError(f"No PNG patches found in: {patch_dir}")

    device_resolved = _choose_device(device)
    model, io_config = _load_pretrained_model(model_name, device_resolved)

    predictions: List[Dict[str, Any]] = []

    batch_tensors = []
    batch_paths = []

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_paths, predictions

        if not batch_tensors:
            return

        x = torch.stack(batch_tensors, dim=0).to(device_resolved)

        with torch.no_grad():
            logits = model(x)

        soft = torch.softmax(logits, dim=1)
        probs_np = soft.detach().cpu().numpy()

        for patch_path, probs in zip(batch_paths, probs_np):
            coord = _parse_patch_filename(patch_path)

            class_index = int(np.argmax(probs))
            confidence = float(probs[class_index])

            predicted_class = KATHER_CLASSES[class_index] if class_index < len(KATHER_CLASSES) else f"class_{class_index}"

            class_probs = {}
            for i, prob in enumerate(probs.tolist()):
                name = KATHER_CLASSES[i] if i < len(KATHER_CLASSES) else f"class_{i}"
                class_probs[name] = float(prob)

            tumour_epithelium_probability = float(class_probs.get("TUM", 0.0))
            stroma_probability = float(class_probs.get("STR", 0.0))
            lymphocyte_probability = float(class_probs.get("LYM", 0.0))
            abnormality_score = max(tumour_epithelium_probability, stroma_probability)

            predictions.append({
                "patch_path": patch_path,
                "filename": os.path.basename(patch_path),
                "level": coord["level"],
                "x": coord["x"],
                "y": coord["y"],
                "predicted_class": predicted_class,
                "predicted_class_description": KATHER_CLASS_DESCRIPTIONS.get(predicted_class, predicted_class),
                "confidence": confidence,
                "tumour_epithelium_probability": tumour_epithelium_probability,
                "stroma_probability": stroma_probability,
                "lymphocyte_probability": lymphocyte_probability,
                "abnormality_score": float(abnormality_score),
                "class_probabilities": class_probs,
                "model_name": model_name,
            })

        batch_tensors = []
        batch_paths = []

    for patch_path in patch_files:
        img_bgr = cv2.imread(patch_path)

        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = _preprocess_patch_for_kather(img_rgb, input_size=input_size)

        batch_tensors.append(tensor)
        batch_paths.append(patch_path)

        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()

    if not predictions:
        raise RuntimeError("Model inference completed but no predictions were produced.")

    class_counts = Counter(p["predicted_class"] for p in predictions)
    total = len(predictions)

    class_percentages = {
        cls: float(100.0 * count / total)
        for cls, count in class_counts.items()
    }

    tumour_like_count = sum(1 for p in predictions if p["predicted_class"] == "TUM")
    stroma_count = sum(1 for p in predictions if p["predicted_class"] == "STR")
    lymphocyte_count = sum(1 for p in predictions if p["predicted_class"] == "LYM")

    result = {
        "model_name": model_name,
        "task": "kather100k_patch_tissue_classification",
        "clinical_warning": (
            "This is tissue-type classification and model-confidence analysis, not a clinical diagnosis."
        ),
        "patch_dir": patch_dir,
        "patch_count": total,
        "device": device_resolved,
        "input_size": int(input_size),
        "class_names": KATHER_CLASSES,
        "class_descriptions": KATHER_CLASS_DESCRIPTIONS,
        "class_counts": dict(class_counts),
        "class_percentages": class_percentages,
        "tumour_epithelium_patch_count": tumour_like_count,
        "tumour_epithelium_percentage": float(100.0 * tumour_like_count / total),
        "stroma_patch_count": stroma_count,
        "stroma_percentage": float(100.0 * stroma_count / total),
        "lymphocyte_patch_count": lymphocyte_count,
        "lymphocyte_percentage": float(100.0 * lymphocyte_count / total),
        "mean_tumour_epithelium_probability": float(np.mean([p["tumour_epithelium_probability"] for p in predictions])),
        "mean_abnormality_score": float(np.mean([p["abnormality_score"] for p in predictions])),
        "max_abnormality_score": float(np.max([p["abnormality_score"] for p in predictions])),
        "io_config_present": io_config is not None,
        "predictions": predictions,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if output_csv_path:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "patch_path",
                    "level",
                    "x",
                    "y",
                    "predicted_class",
                    "predicted_class_description",
                    "confidence",
                    "tumour_epithelium_probability",
                    "stroma_probability",
                    "lymphocyte_probability",
                    "abnormality_score",
                    "model_name",
                ],
            )
            writer.writeheader()
            for row in predictions:
                writer.writerow({
                    "filename": row["filename"],
                    "patch_path": row["patch_path"],
                    "level": row["level"],
                    "x": row["x"],
                    "y": row["y"],
                    "predicted_class": row["predicted_class"],
                    "predicted_class_description": row["predicted_class_description"],
                    "confidence": row["confidence"],
                    "tumour_epithelium_probability": row["tumour_epithelium_probability"],
                    "stroma_probability": row["stroma_probability"],
                    "lymphocyte_probability": row["lymphocyte_probability"],
                    "abnormality_score": row["abnormality_score"],
                    "model_name": row["model_name"],
                })

    lines = [
        "ResNet18-Kather100K patch classification completed successfully.",
        f"Model: {model_name}",
        f"Device: {device_resolved}",
        f"Patches predicted: {total}",
        "",
        "Class summary:"
    ]

    for cls, count in class_counts.most_common():
        desc = KATHER_CLASS_DESCRIPTIONS.get(cls, cls)
        pct = class_percentages[cls]
        lines.append(f"  {cls} ({desc}): {count} patches ({pct:.2f}%)")

    lines += [
        "",
        f"TUM patches: {tumour_like_count} ({result['tumour_epithelium_percentage']:.2f}%)",
        f"STR patches: {stroma_count} ({result['stroma_percentage']:.2f}%)",
        f"LYM patches: {lymphocyte_count} ({result['lymphocyte_percentage']:.2f}%)",
        f"Mean TUM probability: {result['mean_tumour_epithelium_probability']:.6f}",
        f"Mean abnormality score: {result['mean_abnormality_score']:.6f}",
        f"Max abnormality score: {result['max_abnormality_score']:.6f}",
        f"JSON saved to: {output_json_path}",
    ]

    if output_csv_path:
        lines.append(f"CSV saved to: {output_csv_path}")

    lines.append("")
    lines.append("Important: this is tissue-type classification/model confidence, not clinical diagnosis.")

    return "\n".join(lines)


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

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])

    if not preds:
        raise RuntimeError("No predictions found in predictions JSON.")

    total = len(preds)

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

    class_distribution = [
        class_counts.get(cls, 0) / total
        for cls in KATHER_CLASSES
    ]

    class_entropy = _prediction_entropy(class_distribution)

    cluster_count = _count_clusters(
        preds=abnormal_preds,
        cluster_distance=cluster_distance,
    )

    metrics = {
        "source_predictions": predictions_json_path,
        "model_name": data.get("model_name", "resnet18-kather100k"),
        "total_predicted_patches": total,
        "class_counts": dict(class_counts),
        "class_percentages": class_percentages,
        "tumour_epithelium_patch_count": len(tumour_preds),
        "tumour_epithelium_percentage": float(100.0 * len(tumour_preds) / total),
        "high_abnormality_patch_count": len(abnormal_preds),
        "high_abnormality_percentage": float(100.0 * len(abnormal_preds) / total),
        "abnormality_threshold": float(abnormality_threshold),
        "mean_tumour_epithelium_probability": float(
            sum(float(p.get("tumour_epithelium_probability", 0.0)) for p in preds) / total
        ),
        "mean_abnormality_score": float(
            sum(float(p.get("abnormality_score", 0.0)) for p in preds) / total
        ),
        "max_abnormality_score": float(
            max(float(p.get("abnormality_score", 0.0)) for p in preds)
        ),
        "class_entropy": class_entropy,
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
        f"TUM patches: {metrics['tumour_epithelium_patch_count']}",
        f"TUM percentage: {metrics['tumour_epithelium_percentage']:.2f}%",
        f"High abnormality patches: {metrics['high_abnormality_patch_count']}",
        f"High abnormality percentage: {metrics['high_abnormality_percentage']:.2f}%",
        f"Mean TUM probability: {metrics['mean_tumour_epithelium_probability']:.6f}",
        f"Mean abnormality score: {metrics['mean_abnormality_score']:.6f}",
        f"Max abnormality score: {metrics['max_abnormality_score']:.6f}",
        f"Class entropy: {class_entropy:.6f}",
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

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
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

    with open(predictions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
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

        if not os.path.exists(src):
            continue

        safe_name = f"top_{idx:03d}_{cls}_score_{score:.3f}_conf_{confidence:.3f}.png"
        dst = os.path.join(output_dir, safe_name)

        shutil.copy2(src, dst)
        copied_paths.append(dst)

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
