# Illustrative model governance and executive risk summary

The module demonstrates how analysts can record evidence, assumed tolerances, diagnostic status and follow-up. It is an educational retrospective review table, not live monitoring, official bank policy, EBA/IFRS9 compliance or production model approval.

## Assumed criteria

`configs/v2_governance.json` supplies inclusive limits. Missing/nonfinite evidence is **Unavailable**, outside an assumed bound is **Review suggested**, and inside is **Within illustrative tolerance**. None is a banking approval label.

| Diagnostic | Educational bound | Reference |
|---|---|---|
| ROC-AUC | ≥0.75 | 1 |
| Brier | ≤0.15 | 0 |
| Log loss | ≤0.50 | 0 |
| Calibration slope | 0.8–1.2 | 1 |
| Joint calibration intercept | −0.1–0.1 | 0 |
| Observed/expected defaults | 0.8–1.2 | 1 |
| Maximum feature PSI | ≤0.1 | 0 |
| Prediction PSI | ≤0.1 | 0 |
| Source missing-cell rate | 0 | 0 |
| Maximum supported-group absolute calibration gap | ≤0.05 | 0 |

References express idealized diagnostics, not minimum achievable business performance. Supported groups require at least 100 observations and20 events. The source missing rate divides missing cells by rows×25 source columns, including ID/target; it is a frozen audit, not a live feed. PSI uses within-cohort random-split distributions, not temporal drift. Point-status calculations ignore interval width; users must inspect uncertainty alongside them.

A tightened hypothetical ROC-AUC floor 0.80 applied to the existing 0.78038 estimate suggests review. This illustrates sensitivity to chosen criteria; it changes no model, probability, decision or bank policy. Follow-up calls for new governed validation data and analysis rather than tuning the inspected holdout.

## Executive model-risk summary

**What works:** the preserved historical pipeline supports interpretable baselines, controlled nonlinear comparison, independent calibration assessment and reproducible numerical outputs. V2 adds paired uncertainty, local calibration context, hypothetical loss distributions/dependence and explicit review criteria. Model choices and provenance remain frozen.

**What remains uncertain:** unknown customer dependence, sampling/selection effects, local sparse calibration, model-development uncertainty and transportability beyond Taiwan 2005. No genuinely new test cohort is available. Conditional bootstrap precision cannot resolve those gaps.

**Simulation risk:** marginal model probabilities and assumed EAD/LGD/latent correlation determine the simulated portfolio. Dependence can widen the tail while preserving analytic expectation, but rho is not estimated from this dataset. The application cannot measure bank capital adequacy or realized provisioning.

**Evidence needed for real-world validation:** documented lender-specific population/default definition; properly dated development and independent temporal/external cohorts; governed exposures and recoveries; empirically supported dependence/stress assumptions; legal/responsible-use review and monitoring ownership.

**Future monitoring:** validate data contracts and missingness on a governed feed; track distributions and outcomes by actual observation date; review probability calibration and uncertainty using sufficient events; investigate subgroup errors without making automatic eligibility decisions; record approved version changes and genuinely new evaluation evidence. The current dashboard provides none of that institutional authorization.

Actual retrospective statuses and case H are reported in [V2 results](V2_RESULTS.md) and [portfolio cases](PORTFOLIO_CASE_STUDY.md).
