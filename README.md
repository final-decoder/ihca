# Multimodal IHCA discrimination

PyTorch implementation of **Multimodal deep learning integrating electrocardiographic and clinical information for discrimination of subsequent in-hospital cardiac arrest**.

The model combines ten-second, twelve-lead ECG waveforms, automated ECG measurements and interpretation features, and 21 clinical variables. Four multiscale residual convolutional blocks encode each lead, and four-head attention integrates the lead representations. A 256-dimensional ECG representation is fused with a 128-dimensional clinical representation to produce an IHCA discrimination score.

## Installation

Use Python 3.10 or later and install the project from its root directory:

```console
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux and macOS, then install:

```console
python -m pip install ".[experiments,test]"
```

The application uses PyTorch, NumPy and scikit-learn. CPU and CUDA devices are supported through `--device cpu`, `--device cuda` or automatic selection. Configure the batch size in `configs/paper.json` for the available memory. Install the appropriate CUDA build of PyTorch before the project when using a GPU.

## Data preparation

Provide the hospital encounter, department transfer and ECG exports for each site, with the waveforms referenced by the ECG export. Assemble the two cohorts independently. For every case, the matcher applies the case's elapsed hospital time to each candidate's own admission, resolves the department at that instant, and selects the latest eligible ECG strictly before it. The [cohort assembly specification](docs/cohort-assembly.md) defines input columns, timing, matching and exclusions.

```console
ihca-experiments assemble --encounters data/dev/encounters.csv --departments data/dev/departments.csv --ecgs data/dev/ecgs.csv --output data/development.csv
ihca-experiments assemble --encounters data/ext/encounters.csv --departments data/ext/departments.csv --ecgs data/ext/ecgs.csv --output data/external.csv
```

If you already have matched manifests, start with the commands below. Each manifest links participants and matched pairs to an index time, a calibrated ECG file, automated ECG features and clinical variables. Waveforms can be calibrated NumPy arrays or XML. See [lead order, units, required columns and XML adaptation](docs/data.md).

```console
ihca schema --output data/schema.json
ihca prepare --manifest data/development.csv --cohort development --output data/development.npz
ihca prepare --manifest data/external.csv --cohort external --output data/external.npz
```

Provide complete inputs. Preparation rejects missing values, incomplete waveforms, quality scores below 80, incompatible sampling rates and invalid timing. Patient identifiers must be globally consistent across cohorts. The dataset stores calibrated waveforms; within-record normalization is applied during loading.

## Model development

```console
ihca train --data data/development.npz --config configs/paper.json --output outputs/full --device auto
```

The pipeline keeps matched pairs and repeated observations of each patient within the same outer fold. Each outer training partition is split into fitting and inner-validation groups. Scaling is fitted on the fitting partition; inner-validation AUROC selects the checkpoint, and inner-validation Youden's index selects the operating threshold. Each record receives one held-out prediction.

The output directory contains five fold checkpoints, `ensemble.json`, split assignments, optimization histories, pooled out-of-fold predictions and discrimination metrics. Checkpoints retain the model definition, input schema, feature scaling parameters and operating threshold.

The four modality configurations share the same cross-validation pipeline:

```console
ihca train --data data/development.npz --config configs/paper.json --mode clinical --output outputs/clinical
ihca train --data data/development.npz --config configs/paper.json --mode structured --output outputs/structured
ihca train --data data/development.npz --config configs/paper.json --mode waveform --output outputs/waveform
```

`clinical` uses the 21 clinical variables; `structured` adds the automated ECG variables; `waveform` uses the twelve-lead signal; `full` combines all three inputs. The common data interface retains the complete cohort for comparisons across modalities.

## External evaluation

```console
ihca evaluate --data data/external.npz --model-dir outputs/full --output outputs/external --device auto
```

The five frozen models independently transform and score the external records. Their scores are averaged, and the median of the five saved inner-validation thresholds defines the external operating point. Evaluation reports AUROC, trapezoidal AUPRC, average precision, sensitivity, specificity and F1, with confidence intervals from matched-group bootstrap resampling. The score describes discrimination within the cohort's sampling design.

## Inference

Prepare an unlabeled manifest using the same feature definitions and a distinct set of patient identifiers:

```console
ihca prepare --manifest data/inference.csv --cohort prediction --output data/inference.npz
ihca predict --data data/inference.npz --model-dir outputs/full --output outputs/predictions.csv
```

Inference reads trained checkpoints from `--model-dir` and writes record identifiers, scores, thresholds and classifications. The `prediction` data mode requires ECG and clinical acquisition times; labels and matched index times are unnecessary. Commands reject existing result files or populated result directories to preserve previous runs.

## Study experiments

Prepare the development and external data, train all four model configurations above, and evaluate the full model on the external cohort. Supply the scoring fields and candidate ECG records specified in [the experiment protocol](docs/experiments.md). Paths in [the study configuration](configs/experiments.json) resolve relative to that file.

```console
ihca prepare --manifest data/candidates.csv --cohort prediction --output data/candidates.npz
ihca-experiments run --study configs/experiments.json --output outputs/experiments
```

The experiment command writes observed predictions, tables, and figures for model comparisons, paired modality contrasts, structured SHAP attribution, phenotypes, matching ratios, incidence scenarios and cohort characteristics. To run selected analyses, append e.g. `--only ablation subgroups` and choose a new output directory.

```console
ihca-experiments serve --study configs/experiments.json --port 8050
```

Open the local interface at `http://127.0.0.1:8050` to upload an XML recording, enter clinical measurements, inspect the ensemble score and feature contributions, and export a report.

## Project files

```text
configs/                 Model and optimization configuration
docs/                    Data interfaces and method documentation
src/ihca/                Preparation, model, training and inference
tests/                   Numerical, data-boundary and model tests
references.bib           Paper citation
```

## Tests

```console
python -m pytest
```

See [method details](docs/method.md) and [the paper citation](references.bib).
