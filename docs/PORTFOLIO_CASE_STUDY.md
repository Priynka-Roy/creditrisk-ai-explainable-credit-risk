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

## Three evidence-based CV bullets

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
