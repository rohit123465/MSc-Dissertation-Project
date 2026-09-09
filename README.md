# Whole-Slide Image Analysis with MCP

This project lets Claude Desktop run computational pathology tools through a local Python MCP server. It turns a natural-language request into model predictions, spatial measurements, and saved outputs. This guide covers setup, evaluation, and the results supplied in the project archive.

## 1. How the pipeline works

A **whole-slide image (WSI)** is a large digital tissue image. A **region of interest (ROI)** is a selected area, and a **patch** is a smaller tile processed by a model.

```text
User request → Plan and approval → MCP server → Image preparation
→ Pathology model → AnnotationStore → Spatial analysis → Results
```

1. **Plan:** Claude identifies the image, model, tools, and parameters. The user reviews the plan before execution.
2. **Prepare:** Tools read slide metadata and, where needed, create tissue masks or extract patches. Some model engines perform tiling internally.
3. **Predict:** Models classify patches, detect nuclei, separate individual nuclei, or segment tissue regions.
4. **Store:** An AnnotationStore (`.db`) preserves annotation geometries, classes, and probabilities where available.
5. **Analyse:** Compatible predictions are converted into common spatial features for neighbourhood, distance, diversity, and clustering calculations.
6. **Explain:** Claude receives results and file paths. Users inspect CSV/JSON files, reports, and overlays in a compatible viewer such as TIAViz.

The language model coordinates the workflow; the pathology models process the images. Spatial calculations depend on correct class mappings, coordinates, and **microns per pixel (MPP)** when physical distances are required.

## 2. MCP server and Claude Desktop setup

### How the server communicates

`files_mock_claude_response/tiatoolbox_mcpserver.py` receives JSON-RPC messages over standard input/output (**stdio**). Claude Desktop launches this Python process, discovers its tools, and calls them. No HTTP port is needed. The computational implementations are in the neighbouring `tia_tools.py` file.

The server exposes `propose_pathology_plan` and `approve_pathology_plan`. Approval uses the exact confirmation `I approve this plan` and returns a token for authorized tools. Restarting the server clears its in-memory approvals. Long-running predictions can be monitored with `check_prediction_job`.

### Prepare Python

Extract the project and create an environment. Example in PowerShell, replacing the project path:

```powershell
Set-Location "C:\Projects\MSc-Dissertation-Project"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "tiatoolbox==2.1.2" esda libpysal pointpats
.\.venv\Scripts\python.exe -m pip check
```

This is a starting configuration, not a dependency lock. GPU execution requires a compatible PyTorch/CUDA installation; see the [TIAToolbox installation guide](https://tia-toolbox.readthedocs.io/en/latest/installation.html) and [PyTorch setup](https://pytorch.org/get-started/locally/). Pathology-model weights may download on first use.

### Connect Claude Desktop

Open **Settings → Developer → Edit Config**. On Windows, the file is normally `%APPDATA%\Claude\claude_desktop_config.json`; on macOS, it is `~/Library/Application Support/Claude/claude_desktop_config.json`.

Add this entry using your actual absolute paths. Merge it into existing server settings if needed:

```json
{
  "mcpServers": {
    "tiapathology": {
      "command": "C:\\Projects\\MSc-Dissertation-Project\\.venv\\Scripts\\python.exe",
      "args": [
        "-u",
        "C:\\Projects\\MSc-Dissertation-Project\\files_mock_claude_response\\tiatoolbox_mcpserver.py"
      ]
    }
  }
}
```

On macOS, use the environment's `.venv/bin/python` path. Keep both Python source files together, restart Claude completely, and ask: **“Call the tiapathology health tool.”** See the [official MCP setup guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers).

Then provide an absolute slide path and request a plan. Review the model, outputs, and spatial parameters before approval. The slide must be accessible to the computer running the server.

## 3. Download GPT-OSS 20B with Ollama

The project's `gpt-oss/gpt-oss` folder contains an Ollama runtime and a **workflow evaluator**. GPT-OSS judges recorded workflow evidence; it does not replace the pathology models or automatically connect to Claude's MCP tools. The bundled runtime is for Linux; Windows users should install Windows Ollama.

Install from the [Ollama download page](https://ollama.com/download), then run:

```text
ollama pull gpt-oss:20b
ollama list
ollama run gpt-oss:20b
```

The [20B model](https://ollama.com/library/gpt-oss:20b) is approximately 13 GB to download; runtime memory also depends on context length and GPU/CPU allocation. If Ollama is not running, start `ollama serve` in another terminal. Its default endpoint is `http://127.0.0.1:11434`.

Weights use Ollama's model store, not the current directory. To store them elsewhere, configure `OLLAMA_MODELS` on the Ollama server and restart it; see the [Ollama FAQ](https://docs.ollama.com/faq).

### Run the evaluator

Work in a separate copy of `gpt-oss/gpt-oss/evaluation/alignment_evaluation` to preserve existing results:

```text
python run_evaluator.py --task-id T001 --model gpt-oss:20b --runs 1
```

Inspect `evaluation_results/T001/run_01.json`. For five judgments, preserve the smoke-test result separately, then run:

```text
python run_evaluator.py --task-id T001 --model gpt-oss:20b --runs 5 --temperature 0.2 --seed 42000
```

The evaluator combines `requirements/`, `workflows/`, and the prompt/schema in `evaluator/`. The supplied script uses a 16,384-token context and a 4,096-token generation limit. **Always specify 20B:** the script default and older quick-start refer to 120B.

Check each record's `status`, `error`, and `judgment`; `valid` means basic output checks passed, not that the workflow succeeded. The runner may exit successfully after recording errors and overwrites matching run filenames on repeated invocations.

## 4. Evaluate the individual pathology tools

`evaluation/evaluation/run_tool.py` calls Python functions directly. It does not require Claude or Ollama and tests two tools:

| Tool | Model | Task files |
|---|---|---|
| Semantic segmentation | `fcn_resnet50_unet-bcss` | `semantic_001.json`–`semantic_003.json` |
| Nucleus detection | `KongNet_PanNuke_1` | `kongnet_001.json`–`kongnet_003.json` |

### Prepare and run

1. **Fix the nested path assumption.** In your working copy of `run_tool.py`, change `WORKSPACE_ROOT = Path(__file__).resolve().parents[1]` to `parents[2]` so it resolves to the repository root.
2. **Update task JSON paths.** Replace the original `/dcs/22/...` image paths. Set matching `output_json_path`, `save_dir`, and `expected.required_files` values in a fresh run directory.
3. **Correct the task ID.** `semantic_002.json` mistakenly identifies itself as `semantic_001`.
4. **Check the device.** Supplied tasks request CUDA, batch size 1, and one worker. A CPU run requires an explicit supported device change and is a different configuration.
5. **Execute each adapted task** from the repository root, for example:

```powershell
.\.venv\Scripts\python.exe evaluation/evaluation/run_tool.py --task evaluation/evaluation/tasks/semantic_001.json
```

Repeat with the other task files. Inputs include `nucleii.ome.tif` and `roi.ome.tif` in the nested fixtures folder, plus `CMU-1-Small-Region.svs`, `demo.ome.tif`, and `wsi4_12k_12k.svs` at repository root. The original `Zoomed.ome.tif` for `kongnet_003` must be supplied separately; substituting another image creates a new test condition.

For Slurm, adapt the scripts in `evaluation/evaluation/slurm/`: paths, environment, partition, GPU, logs, and selected task. Both archived templates currently select their `_003` case, not all three tests.

### Interpret the outcome

The runner writes `evaluation_result.json` beside the requested output JSON. It checks required files are nonempty, required JSON parses, and at least one nonempty `.db` exists under `save_dir`. Status is `passed`, `failed_validation`, `execution_error`, or `task_configuration_error`.

Inspect the AnnotationStore and overlay as well: file existence alone does not prove correct predictions, and stale files can satisfy checks. These tests assess **execution and artifact production**, not Dice, IoU, or detection accuracy against ground truth.

All six archived task results are marked passed. Semantic runtimes were **32.91, 29.18, and 47.03 seconds**; KongNet runtimes were **86.79, 240.20, and 106.53 seconds**. They used TIAToolbox 2.1.0, PyTorch 2.12.0+cu130, and an NVIDIA RTX A5000. These are historical measurements, not newly reproduced results.

## 5. Understand the spatial-validation results

`common_validation_complete_with_local_csv_two_folder_view` validates spatial calculations using synthetic inputs with known arrangements.

- **`inputs/`:** synthetic databases/images, common spatial features, object/ROI CSVs, and capability metadata.
- **`outputs/manifest.json`:** environment, parameters, scope, and source hashes.
- **`outputs/summary.json` / `summary.csv`:** overall results and case-level checks.
- **Other outputs:** statistic-specific comparison CSVs, observed JSON, Local Moran CSVs, and overlay databases.

The saved run reports **64/64 cases passed in 13.60 seconds**, with absolute tolerance `1e-9`. The results mean:

| Check | Main result and interpretation |
|---|---|
| Adapter contract | IDs, coordinates, scale, probability filtering, and class mappings were preserved; unsupported numeric mappings were rejected. |
| ROI entropy | Expected and observed values were `[0, 1, 0, 1]`: single-class regions have zero diversity; balanced two-class regions have maximum normalized diversity. |
| Neighbour distances | Distances of 3 and 4 µm gave mean/median 3.5 µm, with two links inside 5 µm. |
| Moran's I | Grouped labels gave **0.8933**, indicating positive association; alternating labels gave **−1**, indicating opposite neighbouring states. Shuffled controls were near zero. |
| Ripley's K | All eight cases matched expected pair-count behaviour across radii. Clustered patterns had more close pairs. The estimator uses **no edge correction**. |
| Cross-G | All 12 cases passed. The hand example gave fractions **1/3, 2/3, and 1** at 1, 2, and 5 µm: progressively more source cells had a nearby target. |
| Nearest-neighbour index | Regular **1.7778**, clustered **0.1111**, random approximately **0.9615–1.0819**. Below 1 means shorter spacing than the specified random reference. |
| Tumour–immune proximity | All 16 cases passed, including radius boundaries and missing classes. Interaction strength can exceed 100 because it is **pairs per 100 ROI cells**, not a percentage. |
| Co-occurrence | All four cases passed: expected pair totals were **3, 1, 0, and 5**. Pairs exactly at the radius were included. |

**Global versus local Moran:** Global Moran summarizes the graph; Local Moran describes each nucleus. The grouped case had global p = 0.001, but all 100 local labels were nonsignificant after adjustment. The alternating case had 50 `low-high` and 50 `high-low` local labels. Global significance does not imply every location is locally significant.

Local CSVs contain nucleus IDs, classes, coordinates, local statistics, raw/adjusted p-values, cluster labels, and neighbour IDs. `High-high` means a high feature surrounded by high values; `high-low` means local contrast. These labels describe the selected feature, not tissue grade.

**Limits:** Passing rejection cases means invalid inputs were handled correctly; blank values are not necessarily zero. This validation supports numerical behaviour, not pathology-model accuracy, clinical validity, or calibrated p-values. Embedded paths still refer to the original machine and may need rebasing.

The saved manifest's source hash differs from the supplied `tia_tools.py`, and the original synthetic-validation runner is absent. Exact regeneration needs that runner and matching source/dependencies; `run_tool.py` does not reproduce these 64 cases.

## Quick troubleshooting

- **Claude tools missing:** check JSON syntax, absolute paths, Python environment, and restart Claude fully.
- **Evaluation import/path errors:** correct the nested root assumption and original image paths.
- **CUDA unavailable:** check the PyTorch build, driver, and GPU allocation.
- **Ollama errors:** check the server is running, explicitly select `gpt-oss:20b`, and inspect saved evaluator errors.
- **Unexpected spatial values:** verify class mapping, pixel/micron units, MPP, and ROI coordinates.

Keep functional tests, synthetic numerical validation, and GPT-OSS workflow judgments separate: each answers a different question. Preserve inputs, parameters, logs, versions, and outputs in distinct directories for new experiments.
