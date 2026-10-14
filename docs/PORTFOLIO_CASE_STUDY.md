# Banking analytics case study and presentation

**CreditRisk AI — Explainable Credit Risk Modeling & Model Validation** is an educational historical project by Priynka Roy. It demonstrates statistical risk analysis and analytical engineering without claiming banking employment or production approval.

## A · Historical repayment patterns

**Question:** How do repayment-status categories relate to next-month default?

**Data:** UCI 30,000-customer historical Taiwan 2005 dataset; target next-month payment default.

**Method:** Full-data descriptive rates with Wilson intervals, no causal inference.

**Computed result:** Observed prevalence 22.12%; 6,636 events among 30,000 released records.

**Banking interpretation:** Repayment status is a predictive association to investigate, not a causal lever.

**Limitation:** Historical Taiwan 2005 sample; undocumented codes retained.

## B · Logistic versus boosting

**Question:** Does nonlinear modeling improve probability quality?

**Data:** UCI 30,000-customer historical Taiwan 2005 dataset; target next-month payment default.

**Method:** Three-fold training CV; separate validation; untouched final-test descriptive audit.

**Computed result:** Selected boosting by validation log loss. Final raw L2 log loss 0.4381; boosting 0.4292.

**Banking interpretation:** Discrimination and probability quality both matter; final comparisons do not select the model.

**Limitation:** One frozen random split; no out-of-time evidence.

## C · Discrimination and calibration

**Question:** Does independent calibration improve estimated probabilities?

**Data:** UCI 30,000-customer historical Taiwan 2005 dataset; target next-month payment default.

**Method:** Calibration half-fit/half-selection; chosen method refit full calibration only.

**Computed result:** Chosen none. Final log loss 0.4292, Brier 0.1350, calibration slope 0.984.

**Banking interpretation:** Calibration changes probability scale; improved AUC alone does not imply accurate probabilities.

**Limitation:** Selection partition uncertainty and calibration estimates do not certify transportability.

## D · Illustrative loss sensitivity

**Question:** How do assumed shocks affect modeled loss?

**Data:** UCI 30,000-customer historical Taiwan 2005 dataset; target next-month payment default.

**Method:** Odds stress times hypothetical LGD times hypothetical EAD times 1,000 accounts.

**Computed result:** Baseline modeled loss 9,936,454 NTD; severe assumed scenario 19,592,343 NTD.

**Banking interpretation:** Loss assumptions materially affect a scenario arithmetic result.

**Limitation:** No observed LGD/EAD, no IFRS9 or regulatory loss calculation.

## Original Version 1 career presentation (preserved)

### Three evidence-based CV bullets

- Built a reproducible next-month credit-card default analytics pipeline for 30,000 UCI records, comparing unpenalized/L1/L2 logistic regression and histogram gradient boosting with training-only preprocessing and an isolated 6,000-customer holdout.
- Implemented independent sigmoid/isotonic calibration assessment and conditional bootstrap validation; the validation-selected boosting path achieved test ROC-AUC 0.7804 (95% interval 0.7659–0.7947), Brier 0.13499 and log loss 0.42919, with no additional calibration selected.
- Developed an eight-section Streamlit/Plotly model-validation dashboard with model-specific explanations, denominator-aware subgroup diagnostics, random-holdout stability checks and explicitly hypothetical PD×LGD×EAD stress scenarios.

## LinkedIn project description

CreditRisk AI is my historical credit-risk analytics portfolio project using the UCI Default of Credit Card Clients dataset. I compared logistic, regularized logistic and histogram gradient boosting models with separate training, validation, calibration and test partitions. The project focuses on probability quality, independent calibration assessment, uncertainty, explainability and responsible subgroup diagnostics. An eight-section Streamlit app connects model evidence to illustrative credit-loss sensitivity. The target is next-month payment default, not regulatory one-year default; this is an educational demonstration, not a real borrower decision service.

## Short portfolio description

A reproducible historical credit-card default modeling and validation platform combining interpretable statistical baselines, gradient boosting, independent calibration assessment, bootstrap uncertainty and responsible diagnostics in Streamlit. Illustrative loss scenarios retain clear hypothetical exposure and recovery assumptions.

## Technical interview explanation

“I used the source-defined next-month target and froze a stratified customer split before experiments because reliable customer timestamps were unavailable. I tuned only on training CV, chose features and model using validation probability scores with a simplicity preference, and assessed recalibration on separate observations. Boosting was selected before test access. Additional calibration did not win independently, so I retained the raw probabilities rather than claiming an improvement. The holdout's AUC 0.7804 is only part of the story: Brier/log loss, reliability counts, bootstrap uncertainty and subgroup denominators determine what the evidence supports. My loss module combines fitted next-month probabilities with explicitly assumed LGD/EAD and odds stress. It demonstrates modeling reasoning and reproducibility, not bank-approved underwriting or regulatory provisioning.”

The numerical source is [`reports/results.json`](../reports/results.json); selection evidence is [`reports/frozen_decision.json`](../reports/frozen_decision.json). More credible improvement would require new time-stamped cohorts and measured exposure/recovery information, rather than further tuning on this test set.

## Version 2 cases E–H

Cases A–D above retain their original V1 text/results. The following additions distinguish retrospective inference from hypothetical simulation.

### E · Paired model comparison

**Business question:** Are fitted-model differences distinguishable on the inspected V1 test?

**Statistical method:** 1,000 paired stratified customer bootstrap replicates; Model A minus Model B.

**Actual computed result:** Boosting−L2 log loss difference -0.00886; percentile 95% interval [-0.01215, -0.00562].

**Interpretation:** Negative log-loss differences favor boosting on this historical sample; intervals do not establish practical banking superiority.

**Limitations:** Retrospective, conditional on fitted models and observed class counts; no selection uncertainty.

### F · Calibration uncertainty

**Business question:** How uncertain is calibration around the V1 point estimate?

**Statistical method:** Ordinary customer bootstrap retaining event-count variability.

**Actual computed result:** Calibration slope 0.9845; approximate 95% interval [0.9240, 1.0459].

**Interpretation:** An overall estimate near one does not guarantee local or subgroup calibration.

**Limitations:** Fixed fitted predictions, unknown customer dependence, historical population only.

### G · Dependence and portfolio tails

**Business question:** How does assumed dependence alter a hypothetical loss tail?

**Statistical method:** 5,000 Gaussian latent-factor simulations; identical PD/LGD/EAD, rho 0 versus assumed rho 0.15.

**Actual computed result:** Independent ES95 11,016,000 NTD; common-factor ES95 20,115,180 NTD; common theoretical expected loss 9,933,705 NTD.

**Interpretation:** Common shocks change loss variability and tail risk while preserving marginal expected losses.

**Limitations:** Dependence, LGD and EAD are assumed; tail risk is neither observed bank loss nor regulatory capital.

### H · Illustrative monitoring

**Business question:** How can assumed monitoring tolerances direct analytical follow-up?

**Statistical method:** Compare frozen diagnostic values with labeled educational criteria.

**Actual computed result:** Default criteria suggest review for 0 diagnostics; this is a demonstration rather than model approval.

**Interpretation:** A diagnostic status supports investigation; changing assumed tolerances changes the status.

**Limitations:** No external bank validation, regulatory certification or model reselection.

Case H additional control: increasing the assumed AUC floor to 0.80 flags the actual 0.78038 value as Review suggested; unchanged predictions and model choices demonstrate that governance criteria are assumptions, not performance improvements.

## Version 2 career presentation

### Three evidence-based CV bullets

- Extended a reproducible 30,000-record historical credit-risk platform with 1,000-replicate paired bootstrap comparisons; boosting−L2 log-loss difference −0.00886 (conditional 95% interval −0.01215 to −0.00562) was analyzed retrospectively without selecting new models on the inspected holdout.
- Added ordinary-bootstrap calibration uncertainty, denominator-aware subgroup/error diagnostics and 100 training-only logistic resamples, preserving category references and the original no-additional-calibration decision.
- Implemented bounded 5,000-portfolio Monte Carlo credit-loss scenarios with assumed Gaussian default dependence, tested marginal/expected-loss consistency and discrete VaR/ES definitions, and integrated educational governance criteria into an eight-section Streamlit dashboard.

### LinkedIn description

CreditRisk AI V2 extends my historical credit-card risk analytics portfolio project with paired fitted-model inference, calibration/subgroup uncertainty, development-only resampling and hypothetical portfolio-loss Monte Carlo simulation. Independent and assumed common-factor scenarios show how tail risk can change while analytical expected loss remains unchanged. A structured governance dashboard records educational tolerances and analytical follow-up. The original models, results and no-calibration decision remain preserved, and all additional original-test diagnostics are retrospective. This demonstrates applied Statistics and quantitative risk reasoning, not bank employment, real borrower scoring or regulatory approval.

### Short website description

An explainable historical credit-risk validation platform combining logistic/boosting benchmarks, paired bootstrap inference, calibration uncertainty, hypothetical dependence-aware loss simulation and educational model-governance diagnostics in Streamlit. Original-test reuse and financial assumptions are explicitly labeled.

### Banking-oriented interview explanation

“I preserved V1 before extending the analysis. Because its test results were already seen, I treated all new test diagnostics as retrospective and did not choose new estimators or calibration mappings. Paired bootstrap differences use the same stratified customers for both models; ordinary calibration resampling instead lets event totals vary. I separated those fitted-prediction intervals from 100 development-only logistic refits. My hypothetical loss module uses marginal next-month probabilities with assumed EAD/LGD and a Gaussian common factor. The dependence scenario preserves theoretical mean loss while widening the simulated tail; I tested endpoint defaults, mean consistency and fractional-tail Expected Shortfall rather than relying on plausible charts. Governance limits are educational assumptions with review prompts, not model approval. Stronger bank claims would require new governed temporal/external cohorts, verified exposures/recoveries and institutional validation.”
