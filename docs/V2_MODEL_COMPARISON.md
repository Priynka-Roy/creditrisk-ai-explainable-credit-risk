# Retrospective paired model comparisons

Primary pair: frozen histogram boosting versus frozen L2 logistic regression. Secondary pairs: boosting versus L1 and unpenalized baseline logistic. No estimator is newly selected from these comparisons.

**Difference = Model A (boosting) minus Model B (logistic).** Positive ROC-AUC, average precision and KS differences favor A; negative Brier/log-loss differences favor A. Average precision retains the V1 definition rather than trapezoidal PR integration.

The same stratified customer indices select labels and both prediction arrays within every bootstrap replicate. This is a paired 1,000-replicate percentile bootstrap (seed 142), conditional on the frozen estimators, this inspected 6,000-customer test and fixed event/non-event counts. It does not include refitting or hyperparameter-selection uncertainty. It cannot establish practical banking superiority, external transportability or fresh independent validation.

Each saved row contains model names, metric, point difference, lower/upper 95% endpoints, favorable direction, whether the interval includes zero, resample count, sample size and sign convention. Excluding zero is a descriptive conditional finding; these multiple retrospective comparisons are not pre-registered confirmatory claims. Overlap with zero does not prove equivalence.

Computed tables below come from [`reports/v2/paired_comparisons.json`](../reports/v2/paired_comparisons.json) and the corresponding CSV; see [V2 results](V2_RESULTS.md). Original model point estimates and no-calibration decision are preserved in [V1 baseline](V1_BASELINE.md).

## Actual retrospective differences

| A−B pair | Metric | Difference | Percentile95% interval | Favorable sign | Includes zero? |
|---|---|---:|---:|---|---|
| boosting−l2 | roc_auc | 0.013040 | [0.006656, 0.019037] | positive | no |
| boosting−l2 | pr_auc | 0.020268 | [0.007895, 0.032774] | positive | no |
| boosting−l2 | ks | 0.008589 | [-0.002694, 0.019102] | positive | yes |
| boosting−l2 | brier | -0.002436 | [-0.003512, -0.001375] | negative | no |
| boosting−l2 | log_loss | -0.008860 | [-0.012153, -0.005617] | negative | no |
| boosting−l1 | roc_auc | 0.012440 | [0.005883, 0.018457] | positive | no |
| boosting−l1 | pr_auc | 0.021526 | [0.009434, 0.033647] | positive | no |
| boosting−l1 | ks | 0.013290 | [-0.001588, 0.020838] | positive | yes |
| boosting−l1 | brier | -0.002483 | [-0.003634, -0.001360] | negative | no |
| boosting−l1 | log_loss | -0.009014 | [-0.012498, -0.005737] | negative | no |
| boosting−baseline | roc_auc | 0.012890 | [0.006332, 0.019161] | positive | no |
| boosting−baseline | pr_auc | 0.023452 | [0.011247, 0.036083] | positive | no |
| boosting−baseline | ks | 0.015664 | [0.000765, 0.025355] | positive | no |
| boosting−baseline | brier | -0.002867 | [-0.004078, -0.001651] | negative | no |
| boosting−baseline | log_loss | -0.009901 | [-0.013645, -0.006365] | negative | no |

Against L2, ROC-AUC/AP and probability-score intervals favor boosting conditionally; KS includes zero. Report this negative/uncertain result rather than implying every discrimination statistic is distinguishable. No comparison changes the V1 selection.

These are pointwise 95% intervals, without multiplicity adjustment across metrics or secondary comparisons. They do not provide joint 95% family-wise coverage; secondary comparisons are descriptive and no result is used for model reselection.
