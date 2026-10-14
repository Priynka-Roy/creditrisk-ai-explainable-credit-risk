# Version 2 computed results

All additional original-test analyses are **retrospective**. Frozen V1 choices/metrics remain unchanged. Sources: [`artifacts/v2/dashboard.json`](../artifacts/v2/dashboard.json), [`reports/v2/experiment_registry.json`](../reports/v2/experiment_registry.json) and versioned component reports. Figures are generated from these actual values; simulation outputs are hypothetical.

## Paired comparison and calibration

Primary boosting−L2 difference in log loss is **−0.008860**, percentile 95% interval **[−0.012153,−0.005617]**. AUC difference is **+0.013040 [0.006656,0.019037]**; Brier difference **−0.002436 [−0.003512,−0.001375]**. KS **+0.008589 [−0.002694,0.019102]** includes zero. Sign conventions and all secondary comparisons: [paired report](V2_MODEL_COMPARISON.md).

Overall ordinary-bootstrap calibration intervals: intercept **[−0.108520,0.081006]**, slope **[0.924002,1.045924]**, O:E **[0.959926,1.042264]**. The previously chosen no-additional-calibration path remains fixed. [Calibration report](V2_CALIBRATION_UNCERTAINTY.md).

## Development-only stability

100 training-resampled L2 fits preserve original category references and are evaluated on the existing validation partition. These show training sensitivity, not new validation or selected-boosting fit uncertainty.

| Diagnostic | Mean across refits | 2.5th–97.5th resample range |
|---|---:|---:|
| roc_auc | 0.771016 | [0.767995,0.774168] |
| pr_auc | 0.534474 | [0.527736,0.542420] |
| brier | 0.137419 | [0.136878,0.138149] |
| log_loss | 0.437352 | [0.435863,0.439274] |
| intercept | -0.047996 | [-0.112720,0.024951] |
| slope | 0.942338 | [0.906901,0.980020] |
| observed_expected_ratio | 1.010751 | [0.986703,1.040159] |

LIMIT_BAL coefficient is negative in 100/100 fits; AGE is positive in 66/100, with a coefficient range crossing zero. Directional instability is retained rather than marketed as a stable causal effect. Original-V1 train/test PSI remains unchanged; V2 training/validation distributions are separately recorded.

## Hypothetical Monte Carlo scenarios

1,000 fixed synthetic accounts from the rounded anonymous histogram; 5,000 portfolios each; baseline EAD 100,000 NTD/account, LGD 45%, seed 442. All monetary outputs are NTD. VaR/ES use 95% loss-tail conventions.

| Assumption | rho | Analytical EL | Simulated mean | SD | VaR95 | ES95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline · independent | 0.00 | 9,933,705 | 9,939,546 | 517,492 | 10,800,000 | 11,016,000 |
| Baseline · common factor | 0.15 | 9,933,705 | 9,927,396 | 4,192,568 | 17,685,000 | 20,115,180 |
| Mild adverse | 0.15 | 12,576,443 | 12,568,572 | 4,750,584 | 21,240,000 | 23,674,140 |
| Severe adverse | 0.15 | 19,588,962 | 19,577,136 | 6,821,699 | 31,920,000 | 35,049,840 |
| LGD increase only | 0.15 | 13,244,940 | 13,236,528 | 5,590,091 | 23,580,000 | 26,820,240 |
| EAD increase only | 0.15 | 12,417,131 | 12,409,245 | 5,240,710 | 22,106,250 | 25,143,975 |

Independent and rho 0.15 baseline scenarios share analytical EL 9,933,705 NTD. Their simulated means 9,939,546 and 9,927,396 differ by Monte Carlo sampling. Their means are 0.80 and −0.11 simulation standard errors from theory; these are numerical checks, not bank validation. Dependence raises ES95 from 11,016,000 to 20,115,180 NTD under these assumptions. The rounded synthetic portfolio differs slightly from V1 exact-customer scenario arithmetic, whose original baseline is 9,936,454 NTD.

[Simulation methodology](V2_RISK_SIMULATION.md) documents marginals, finite-run uncertainty, units, VaR/ES/ties, stress and workload bounds. The saved marginal-check summary reports individual probability errors; the hypothetical latent rho is not empirically identified.

## Educational governance

All 10 default point diagnostics fall within the explicitly assumed criteria. This is **not model approval**, fairness certification or evidence of real monitoring. Tightening the ROC-AUC floor from 0.75 to 0.80 changes its status to Review suggested for the same 0.780379 observed value; no model or predictions change. The maximum supported-group absolute calibration gap is 0.016845 under the source group-size filter. Follow-up and executive risk summary: [governance](V2_MODEL_GOVERNANCE.md).

## Provenance and implementation evidence

The registry preserves source/split hashes, original-file hashes, seeds, replicate counts, model reproduction differences, fitted-library versions, assumptions and measured runtime. No customer-aligned prediction array is published. Final software, app and CI checks are recorded in [internal final review](FINAL_REVIEW.md). No fresh external data or genuine temporal cohort exists.

## Exported interactive figures

- [Paired comparison intervals](../reports/v2/figures/paired-comparison.html)
- [Hypothetical independent/common-factor loss histograms](../reports/v2/figures/dependence-loss.html)

Download/open the HTML files locally to interact with the plots; GitHub's source-file viewer does not execute them. The dashboard presents the same computed evidence directly.
