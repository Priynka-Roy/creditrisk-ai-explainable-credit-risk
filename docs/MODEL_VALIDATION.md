# Model validation report

This is an **internal analytical and code validation report**, not validation by a bank, regulator or external independent organization. The untouched customer holdout evaluates the frozen selection/calibration path. Training/validation/calibration/test identifier intersections must be empty and preprocessing fits must remain partition-restricted.

## Metric definitions

- Positive class: source next-month payment default = 1.
- ROC-AUC measures ranking across default/nondefault pairs.
- PR-AUC here is **average precision**, the recall-step-weighted precision measure implemented by scikit-learn, not trapezoidal PR integration. Its prevalence reference depends on the evaluated sample.
- KS is maximum `TPR − FPR` over the ROC curve; no lending cutoff is inferred from the maximizing point.
- Brier score is mean squared probability error; log loss is binary cross entropy. Lower is better for both.
- Threshold flags use `probability >= threshold`; precision is TP/(TP+FP), recall TP/(TP+FN), specificity TN/(TN+FP). FPR and FNR use their own class denominators.
- Calibration intercept/slope are descriptive fitted-logit diagnostics. Observed/expected ratio divides event count by summed probabilities.

One-class groups have no ROC-AUC, average precision or KS in this implementation. Empty class/prediction denominators and empty reliability bins remain unavailable. Constant probabilities can have discrimination metrics where both classes exist, but calibration slope/intercept are unavailable. No undefined metric is fabricated.

## Uncertainty and interpretability

Stratified 300-replicate bootstrap intervals preserve the holdout's event count and condition on this fitted pipeline. Paired model-difference intervals are not included in this release. They are conditional empirical precision, not model-selection, temporal or population uncertainty. Source dependence remains unknown. Threshold comparisons are illustrative controls and may not be described as optimized independently validated credit decisions.

The numerical assessment below is generated from [`reports/results.json`](../reports/results.json). See [methodology](METHODOLOGY.md), [calibration](CALIBRATION.md), [responsible subgroups](FAIRNESS_AND_RESPONSIBLE_AI.md) and [stability](STABILITY.md).

## Final holdout results

6,000 customers include 1,327 next-month defaults (22.1167%). All models below were frozen before this audit; boosting is the deployed uncalibrated path.

| Candidate | ROC-AUC | Average precision | KS | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| baseline | 0.76749 | 0.53143 | 0.40861 | 0.13785 | 0.43909 |
| l2 | 0.76734 | 0.53461 | 0.41568 | 0.13742 | 0.43805 |
| l1 | 0.76794 | 0.53335 | 0.41098 | 0.13747 | 0.43821 |
| boosting | 0.78038 | 0.55488 | 0.42427 | 0.13499 | 0.42919 |

### Selected-path conditional 95% bootstrap intervals

| Metric | Estimate | 2.5th–97.5th percentile |
|---|---:|---:|
| roc_auc | 0.78038 | 0.76586–0.79466 |
| pr_auc | 0.55488 | 0.52923–0.58119 |
| ks | 0.42427 | 0.39756–0.45662 |
| brier | 0.13499 | 0.13105–0.13907 |
| log_loss | 0.42919 | 0.41874–0.43998 |

Calibration intercept -0.01408; slope 0.98449; observed/expected defaults 1.00161. Mean estimate 22.0810%, observed 22.1167%. Near-unity aggregate calibration does not guarantee local, subgroup or future-population calibration.

### Illustrative threshold diagnostics

| Threshold | Precision | Recall | Specificity | TP / FP / TN / FN |
|---|---:|---:|---:|---|
| 0.1 | 0.2878 | 0.9156 | 0.3565 | 1215 / 3007 / 1666 / 112 |
| 0.2 | 0.4267 | 0.6624 | 0.7473 | 879 / 1181 / 3492 / 448 |
| 0.3 | 0.5360 | 0.5335 | 0.8688 | 708 / 613 / 4060 / 619 |
| 0.5 | 0.6604 | 0.3723 | 0.9456 | 494 / 254 / 4419 / 833 |
| 0.7 | 0.7572 | 0.1786 | 0.9837 | 237 / 76 / 4597 / 1090 |
