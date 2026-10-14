# Probability calibration

The probability target is estimated next-month payment default. Independent calibration observations were withheld from base-model training and model selection. The frozen calibration partition is split equally into stratified fit and selection halves. No-calibration, sigmoid of base logits and isotonic candidates are compared on the selection half by log loss; simpler methods within the 0.002 tolerance take precedence. The chosen mapping is then refitted on the complete calibration partition. Final-test labels fit neither the base estimator nor its calibrator.

Sigmoid learns an intercept and slope on stabilized base logits. Isotonic learns a nondecreasing mapping with boundary clipping. The latter can introduce flat regions and alter ties; ROC-AUC may change even while probability error improves. Neither method is assumed better in advance. The no-calibration candidate is retained when independently measured benefit does not satisfy the rule.

Brier score averages squared probability error. Log loss is the binary proper scoring rule with numerical probability clipping. Reliability groups display mean prediction, observed event rate, count and event uncertainty. Empty bins remain absent/unavailable; tiny bins are not precise calibration evidence.

Diagnostic calibration intercept and slope fit a descriptive logistic relationship between observed test labels and predicted logits. Ideal values are approximately 0 and 1. These are diagnostics **after** decisions, not another fitted deployable calibration step. They are unavailable for one-class outcomes or constant probabilities. Observed/expected counts are descriptive, not a guarantee that an individual outcome matches a probability.

See saved calibration-selection results for the selected mapping, and [model validation](MODEL_VALIDATION.md) for final numerical diagnostics. See [methodology](METHODOLOGY.md) for split isolation and [limitations](LIMITATIONS.md) for transportability.

## Realized calibration decision

The calibration-selection half records log loss 0.43354 without calibration, 0.43391 with sigmoid and 0.45175 with isotonic; corresponding Brier scores are 0.13593, 0.13600 and 0.13759. **No calibration is selected.** The candidate mappings were implemented and independently assessed; claiming that recalibration improved the deployed path would be incorrect. Final-test comparisons of the full-calibration refits are audit outputs, not a reason to reverse that decision.

## Final-test descriptive audit

| Mapping | ROC-AUC | Brier | Log loss | Slope |
|---|---:|---:|---:|---:|
| none | 0.78038 | 0.134986 | 0.429192 | 0.98449 |
| sigmoid | 0.78038 | 0.134985 | 0.429191 | 0.98487 |
| isotonic | 0.77924 | 0.134807 | 0.455723 | 0.89563 |

Sigmoid is effectively unchanged on test. Isotonic slightly lowers Brier but worsens log loss materially, illustrating why one favorable metric is insufficient. These outcomes do not change the pre-test no-calibration selection. Reliability-bin counts and Wilson intervals are saved in the same result report.
