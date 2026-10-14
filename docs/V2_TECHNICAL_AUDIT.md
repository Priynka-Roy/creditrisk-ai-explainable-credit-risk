# Version 2 initial technical audit

Audit baseline: commit `790ddbe1db74c4f9258ada80070140e563beec42` / tag `v1.0.0`. Reviewed source ingestion/schema checks, feature transformations, customer splitting, train-only preprocessing/CV, validation model choice, independent calibration, test metrics, bootstrap, model explanations, subgroup/stability reports, scenario math, eight-section Streamlit interface, tests, CI and artifact integrity. This is an implementation-team audit, not external bank validation.

## Findings before the V2 additions

| Priority | Finding | Required action |
|---|---|---|
| P0 — correctness constraint | V1 holdout outcomes are already known; relabeling V2 diagnostics as fresh independent validation would be false | Freeze the V1 choices and label all additional original-holdout inference retrospective; prevent model/calibration selection on it |
| P0 — reporting constraint | Original point estimates/configuration/artifacts must not be overwritten by V2 | Preserve byte hashes and version outputs separately |
| P0 — simulation prerequisite | V1 supplies expected-loss arithmetic, not a loss distribution; inventing tail risk from aggregate metrics would be invalid | Implement and numerically test a defined stochastic model before reporting VaR/ES |
| P1 | Model point comparisons have no paired uncertainty around their differences | Reproduce actual paired predictions from unchanged fitted pipeline; same-customer bootstrap indices for both candidates |
| P1 | Calibration intercept/slope and O/E are point diagnostics with limited local uncertainty | Add ordinary customer bootstrap plus band/group counts, uncertainty and explicit undefined cases |
| P1 | Twenty coefficient resamples are an exploratory stability check | Increase development-only replicates while fixing original training category dictionary/reference levels |
| P1 | Expected-loss scenarios omit systematic dependence and tail variability | Add bounded independent/common-factor simulation with hypothetical correlation and check marginal/default-loss consistency |
| P1 | Subgroup error rates omit interval uncertainty | Preserve denominators and source coding; add uncertainty and sparse-group warnings without fairness certification |
| P1 | Governance is informational rather than a structured review table | Add educational tolerances, observed/reference values, diagnostics and follow-up; no production approval status |
| P2 | Provenance categories can be clearer in the interface | Distinguish frozen V1, V2 development resampling, retrospective test diagnostics and hypothetical simulated scenarios |
| P2 | Career-facing cases cover only the first-release methods | Retain cases A–D exactly; add actual-evidence cases E–H and measured career descriptions |

## Existing strengths retained

Original source hashes/schema and raw preservation; identifiers excluded; undocumented categories retained; fixed stratified split and train-fold preprocessing; probability proper scores and a simplicity-aware validation rule; separate calibration-fit/selection observations; honest no-calibration result; zero/negative ratio handling; model-specific predictive explanations; denominator-aware subgroup flags; random-split PSI limitations; odds rather than naive probability stress; lightweight aggregate app artifacts; deterministic tests and Linux CI.

## Implementation acceptance criteria

No V1 prediction/model-selection changes; numerical agreement after frozen-pipeline reproduction; paired-resampling sign/unit correctness; ordinary calibration bootstrap permits varying event totals; finite/undefined sparse diagnostics handled transparently; valid probability/LGD/exposure inputs; endpoint defaults and zero-loss cases exact; simulated mean consistent with theoretical loss to Monte Carlo error; factor dependence preserves intended marginals; discrete VaR/ES conventions documented and tested; workload bounds/caching; all eight app sections tested; final Ruff/pytest/integrity and GitHub Actions verified. Results and fixes are recorded in V2 documentation and final internal review after evidence exists.
