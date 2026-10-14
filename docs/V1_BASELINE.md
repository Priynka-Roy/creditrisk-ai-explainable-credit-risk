# Version 1 baseline — frozen historical release

Preserved commit: `790ddbe1db74c4f9258ada80070140e563beec42`, tagged **v1.0.0**. The annotated tag resolves to this commit; no tag overwrite or history rewrite is permitted. Version 2 develops on `feature/creditrisk-v2` in the same repository.

The source cohort has 30,000 observations and 6,636 next-month payment defaults. Original split: training 15,000; validation 4,500; calibration 4,500; test 6,000. The chosen estimator is histogram gradient boosting (150 iterations, learning rate 0.05, seven leaves, L2 regularization 10), with the validation-selected engineered group. **No additional calibration is selected.** This decision remains fixed.

## Exact saved original numerical results

Values below are copied from frozen `reports/results.json` without recomputation; table entries retain the saved floating-point values.

| Quantity | Original saved value |
|---|---:|
| roc_auc | 0.78037890551487 |
| pr_auc | 0.5548774627538484 |
| ks | 0.4242720330084916 |
| brier | 0.13498569410997835 |
| log_loss | 0.42919206489363365 |
| mean_probability | 0.22081009428972326 |
| observed_rate | 0.22116666666666668 |
| observed_expected_ratio | 1.0016148373021188 |
| intercept | -0.014077052610673556 |
| slope | 0.9844851618728686 |

| Original model | ROC-AUC | Average precision | KS | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| baseline | 0.7674890353617948 | 0.5314252353500772 | 0.40860844844382527 | 0.13785249968073449 | 0.4390934986616844 |
| l2 | 0.7673387387436784 | 0.5346090930066174 | 0.4156833553429722 | 0.1374218684161699 | 0.43805203447722313 |
| l1 | 0.7679391189038152 | 0.5333516589626803 | 0.41098207067779097 | 0.13746901831607533 | 0.43820568588424375 |
| boosting | 0.78037890551487 | 0.5548774627538484 | 0.4242720330084916 | 0.13498569410997835 | 0.42919206489363365 |

## Preservation checks

Original configuration, decisions, reports and deployment artifacts remain at their V1 paths, unchanged. Version 2 outputs belong under `reports/v2/` and `artifacts/v2/`. Original file hashes:

| Path | SHA-256 |
|---|---|
| `artifacts/dashboard.json` | `f18d9403113eaa671a14e9e851d03023d49c13a96606560496d058ae86f7e971` |
| `reports/results.json` | `687b70868880629ee27e0078c510e4a321dec1721a605bbb4f2d44bad484a81e` |
| `reports/frozen_decision.json` | `111c42c9d6a9bc2cea14310b926cebd724f36bae2cf1b8383fdbe56d84f28fd3` |
| `reports/data_audit.json` | `2a441d58ccb59d9187d6e1957d3739521d508dbef2dfa4d72ffc4d62d56c5db6` |
| `reports/source_manifest.json` | `28bab9c962e7871802d6da85ad20107bee693276275e3458b84e6196e0392d03` |
| `configs/protocol.json` | `6c211be16b022066edfa713ae94a701538500f1906e5fe263e8df10767638f63` |

The original manifest itself is also preserved; the V2 manifest records preservation separately. Raw ZIP/XLS and customer-ID split arrays remain local/ignored. No individual model predictions are inferred from rounded aggregate metrics. When V2 needs paired customer-level predictions it reproduces the frozen pipeline against the same source and split, checks agreement, and exports aggregates only.

## Evaluation status

The V1 test outcomes have already been inspected. Every V2 analysis of these observations is **retrospective Version 2 analysis**, never a fresh independent evaluation. No new model, feature group, hyperparameter or calibration method is selected using this holdout. There is no new external dataset or genuine temporal cohort. V1 cases A–D and their original results are retained alongside the V2 additions.
