# Analytical boundaries

CreditRisk AI estimates **next-month payment default probabilities** in a historical Taiwanese credit-card dataset. The observations do not establish a regulatory default definition, an observed twelve-month outcome, or applicability to present-day Finnish or European bank customers.

- A random customer holdout assesses performance within this historical sample. It does not establish out-of-time generalization, macroeconomic robustness, or transportability to another lender.
- One row per identifier does not prove customers are independent. Customer-level bootstrap intervals omit uncertainty from shared economic conditions, selection bias, missing cohorts, and model development.
- Source repayment codes −2 and 0, education codes 0, 5 and 6, and marriage code 0 have no meaning established by the current official dictionary. Preserve them as labeled source codes; do not invent business interpretations.
- Associations, coefficient odds ratios, permutation importance and synthetic perturbations do not identify causal effects. Correlated history variables can share or substitute predictive information.
- Subgroup diagnostics use historical sex, education, marital-status and age fields. Uncertainty, event denominators and small-group suppression matter. Metric parity does not certify fairness; a disparity does not by itself establish unlawful discrimination.
- Hypothetical LGD, hypothetical EAD and assumed probability stress generate illustrative monetary scenarios. Credit limit is not measured EAD. No recovery observations, realized loss validation or macroeconomic stress estimates are available.
- Repeated inspection of the holdout after this release would compromise its independence if used to revise models. A future substantive model revision requires a documented new evaluation design.

This is not a production scoring system, underwriting service, IRB model, IFRS 9 impairment engine or regulatory certification. It accepts no real borrower profiles and provides no lending eligibility advice. No compliance with Basel, EBA, IFRS 9, the EU AI Act, GDPR or Finnish banking regulations is claimed. No bank endorses this work.

## Regulatory context

The [EBA's PD/LGD modeling guidance](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd) concerns the IRB framework. Its [Q&A on counting obligors](https://eba.europa.eu/single-rule-book-qa/qna/view/publicId/2019_4599) discusses PD estimation from long-run one-year default rates. These references explain why this dataset's next-month label is a different target; citing them confers no approval or compliance. Default definitions also require evidence absent here; see the [EBA default-definition guidance](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-application-definition-default?version=2016).

## Future improvements

Obtain properly governed, time-stamped lender cohorts and a documented default definition; assess transportability and temporal calibration; validate exposure and recovery data separately; extend repeated development resampling and external validation; analyze dependencies and intersectional subgroup uncertainty before drawing stronger conclusions. Each requires new evidence, not reinterpretation of the existing holdout.

## Version 2 additions and their limits

V1 test outcomes are already known. V2 customer-bootstrap inference is retrospective and conditional on frozen fitted probabilities; it does not make those observations fresh or estimate all training/selection uncertainty. Stratified paired inference fixes event counts, whereas ordinary calibration resampling allows prevalence variation. Interval exclusion of zero is not practical bank superiority. Group intervals are exploratory without multiplicity correction and tiny groups remain weak evidence.

The 100-refit stability extension is development-only logistic training sensitivity evaluated on the original validation observations, not new validation or a selected-boosting refit experiment. Standardization is refitted during resampling, so transformed standardized coefficient magnitudes need that scale interpretation even with fixed categorical references.

Monte Carlo loss variability is conditional on assumed portfolio probabilities, LGD/EAD and latent correlation. It is distinct from fitted-model uncertainty and economic-scenario probability. Assumed common-factor dependence preserves marginal expected loss but can materially alter tails. Finite simulations leave numerical uncertainty; rounded anonymous probability histograms introduce approximation. No correlation is identified by these historical records.

Governance bounds are educational assumptions applied to retrospective point estimates. They are not official policies, externally validated tolerances or production approval criteria. Real monitoring requires governed future outcomes, evidence ownership and institutional review.
