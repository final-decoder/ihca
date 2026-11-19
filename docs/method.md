# Model and evaluation

## Architecture

For a batch of twelve-lead recordings `X` with shape `[B, 12, 5000]`, the waveform encoder applies shared temporal filters independently to each lead. Four residual blocks have output widths 32, 64, 128 and 256. Each block concatenates parallel convolutions with kernels 7, 15 and 31, adds a projected residual, and uses GELU and max pooling of width/stride two. Every convolution is followed by batch normalization. The total output width is divided across the three parallel branches.

Temporal global average pooling and dropout of 0.3 produce one 256-dimensional representation per lead. A learned projection to 128 dimensions and lead-identity embeddings form twelve tokens. Four-head self-attention acts across those tokens, and averaging its output across leads gives the waveform representation.

The waveform representation and the ordered structured ECG features are concatenated and projected to 256 dimensions with GELU. The clinical branch maps 21 inputs to 64 and then 128 units, with batch normalization and GELU. Concatenating both representations gives 384 units. Fusion layers map 384 to 128 to 64, using GELU and dropout of 0.5 after each hidden layer. A final linear unit supplies a logit; sigmoid supplies the model score.

Training uses binary cross-entropy implemented on logits, AdamW with initial learning rate 0.001 and weight decay 0.01, batch size 64, and at most 100 epochs. Early stopping uses inner-validation AUROC, patience 15 and minimum improvement 0.0001. The default development seed is 42.

## Grouped evaluation

The grouping unit is the connected component of matched-pair and patient identities. Five outer folds partition these components, balancing record counts. Every outer iteration partitions its available groups again into fitting and inner validation. Scaling, optimization and batch-normalization updates use only fitting records. The inner partition selects the epoch and operating threshold. The outer partition contributes only held-out predictions and evaluation labels.

The operating threshold maximizes sensitivity plus specificity minus one on the inner-validation partition. Equal maxima choose the largest observed score threshold. Predictions use `score >= threshold`. Pooled internal threshold metrics use each record's own fold threshold.

External evaluation averages the five sigmoid scores and applies the median of the saved fold thresholds. Models, scaling parameters and thresholds are unchanged. External patient IDs are checked against development patient IDs before inference.

## Metrics

AUROC uses the rank ordering of held-out scores. AUPRC is trapezoidal area under the precision-recall curve; average precision is also returned under its own name. Sensitivity, specificity and F1 use the saved operating point. The case fraction is retained with each report.

Percentile 95% intervals resample whole matched groups with replacement. When a patient appears in multiple pairs, all connected pairs travel together in each bootstrap unit. The default is 2,000 resamples. The `prevalence_predictive_values` function calculates scenario-specific PPV and NPV from sensitivity, specificity and an explicitly supplied prevalence.

## Saved models

`fold_<k>.pt` stores weights, model configuration, input schema, feature scaling, fold ID, selected epoch and threshold. `ensemble.json` lists the fold files with relative paths. A model is reconstructed before state loading, and shape/schema mismatches cause an error. Model loading uses PyTorch's weights-only serialization interface.
