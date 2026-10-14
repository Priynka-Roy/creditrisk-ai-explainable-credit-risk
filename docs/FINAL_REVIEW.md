# Internal five-perspective technical review

This review was conducted within the implementation team against the frozen protocol, source files, code, actual saved metrics and deterministic tests. It is not independent regulatory validation, an external bank review, certification or evidence of employer endorsement.

## A — Applied statistician

Checked split isolation, training-fold preprocessing, proper scoring rules, calibration separation, event denominators and uncertainty interpretation. The final metric report uses a stratified customer holdout and 300 conditional bootstrap replicates. Coefficient resampling is exploratory with only 20 replicates, not a well-resolved inferential interval. The source does not establish independence. No paired bootstrap difference interval or genuine out-of-time validation is claimed.

**Defect fixed:** rare categories in coefficient resamples initially changed vector dimensions/reference levels. Reuse the original training-only category dictionary/reference levels and align coefficient outputs; no final-test-driven feature/model/calibration change. The reporting rerun is disclosed in methodology.

## B — Banking credit-risk specialist

Checked target horizon, monetary units, credit-limit interpretation and loss assumptions. All probabilities refer to next-month payment default, not regulatory one-year PD. Scenario losses use hypothetical EAD/LGD/count and odds shifts; no observed recoveries or loss outcomes are invented. EBA references distinguish real IRB scope from this project. Thresholds are analytical controls, not recommended lending decisions.

**Negative finding retained:** independent calibration selection favors no additional mapping. Isotonic's slightly better test Brier accompanies worse log loss and does not authorize reversing the frozen selection.

## C — Machine-learning engineer

Checked train-only categorical/numeric preprocessing, limited training CV, pre-test decision persistence, source hashes and aggregate artifact architecture. Unseen categories do not refit encoders; identifiers/outcomes do not enter features. The original ZIP/XLS remains unchanged/ignored; the app uses supplied artifacts and performs no model fitting or source download at startup. Lightweight integrity checks bind published numerical reports to artifact hashes.

**Defect fixed:** a duplicate recent-payment summary was replaced before experiments with a distinct September-minus-August payment-change summary. The final feature definition and ablation evidence are documented. The public app needs no estimator pickle, customer upload, account database or API secret.

## D — Responsible-AI reviewer

Checked preservation of undocumented codes, subgroup counts/events, unavailable rates for empty denominators, small-group cautions, public-output minimization and unsupported uses. Demographic metrics are historical descriptive associations; no causality, unlawful-discrimination judgment or fairness-compliance claim follows. Raw customer IDs and linked individual records are excluded from public outputs. No real-person approval/rejection interface is provided.

**Residual limits:** extremely small groups and low-event groups cannot support strong comparisons; historical demographic coding is limited and transportability is unproven.

## E — Technical hiring manager

Checked whether evidence supports the career-facing descriptions. The project demonstrates interpretable baselines, regularization, boosting, probability calibration assessment, uncertainty, validation diagnostics and reproducible application engineering. The four guided cases report actual outcomes, including lack of recalibration benefit. CV/LinkedIn/interview text describes a portfolio project rather than employment or approved bank risk management.

**Residual limits:** banking relevance is methodological. Deployment to Streamlit is an owner action after review; no public URL should be claimed until verified.

## Verification record

Local verification completed on 14 October 2026: 57 pytest tests pass, including 21 Streamlit application tests covering all eight sections and edge-case controls. Ruff passes and six artifact hashes plus aggregate contracts verify. The app starts on local port 8503 without downloading raw data or training. Browser checks confirm actual overview/calibration rendering and navigation; desktop 1440-pixel and mobile 390-pixel layouts were inspected, with no mobile horizontal overflow or captured console errors. Real screenshots are included in docs/assets. Top spacing, long metric text and sidebar link contrast were corrected during visual review. Publication and CI status will be recorded after the new repository is pushed. Public Streamlit deployment remains an owner action after review.
