# Responsible subgroup diagnostics

The dashboard examines historical age bands and original sex, education and marital-status codes. These fields are dataset categories, not assumptions about a current person's identity. The deployed application accepts no real customer details or uploads; any profile examples are explicitly synthetic model-behavior illustrations.

For each group, report customer count, default/nondefault denominators, observed rate with Wilson interval, mean prediction, calibration gap and threshold-specific false-positive/false-negative rates. FPR divides by nondefaults; FNR divides by defaults. Undefined rates remain unavailable rather than being set to zero. Groups below 100 observations or 20 events require suppression/caution, and larger groups with few events can still be imprecise. Threshold sensitivity is descriptive, not an approval-rate recommendation.

Random-holdout subgroup results do not establish causal discrimination or deployment fairness. Differences can reflect case mix, measurement choices and selection; apparent parity is not compliance. Do not combine undocumented source codes under invented labels. Correlated demographic proxies can remain even if a sensitive field is excluded. Intersectional, external-population and prospective governance analyses would require further evidence.

The saved subgroup report is the numerical source of truth. Interpretation and group-level findings are summarized in [model validation](MODEL_VALIDATION.md). See [limitations](LIMITATIONS.md) for unsupported uses and regulatory boundaries.

## Saved sex-category diagnostics (test; threshold 0.5)

| Historical category | n | Events | Observed | Mean prediction | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| 1 · male | 2,402 | 561 | 23.36% | 22.72% | 5.98% | 64.35% |
| 2 · female | 3,598 | 766 | 21.29% | 21.66% | 5.08% | 61.62% |

Male-coded and female-coded groups have different descriptive outcome and error rates. This is a diagnostic finding, not a discrimination judgment or fairness certification. Undocumented education code 0 has only two test observations and no events; its FNR is undefined. Small groups are flagged when n < 100 **or** events < 20; count/event cautions remain visible even where a metric can be computed. Full education, age-band and marriage diagnostics and threshold sensitivity are saved in the report.
