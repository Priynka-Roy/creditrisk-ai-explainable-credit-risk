# Data card

**Source:** [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). Attribution: Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H). UCI identifies I-Cheng Yeh as creator and links the 2009 Yeh/Lien research paper. Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the repository's MIT license applies to code, not the source dataset.

The official workbook has 30,000 customer rows, an identifier, 23 explanatory variables and one binary outcome. Repayment, bill and payment histories cover April–September 2005 in Taiwan. Monetary values are New Taiwan dollars. The outcome is source-defined **next-month payment default**, encoded 1 for default and 0 for nondefault. The source does not establish twelve-month defaults, measured EAD, recoveries, realized losses or a regulatory default classification.

## Dictionary

| Workbook field | Meaning / treatment |
|---|---|
| ID | Audit identifier; never a predictor or displayed customer record |
| LIMIT_BAL | Granted credit amount, including supplementary/family credit; not measured EAD |
| SEX | Historical source categories 1 male / 2 female; no modern identity inference |
| EDUCATION | 1 graduate school / 2 university / 3 high school / 4 other; preserve additional observed codes separately |
| MARRIAGE | 1 married / 2 single / 3 other; preserve additional observed codes separately |
| AGE | Age in years |
| PAY_0, PAY_2 … PAY_6 | September backwards to April repayment statuses; −1 duly paid, positive codes indicate months of delay, 9 means nine or more; source does not define −2 or 0 |
| BILL_AMT1 … BILL_AMT6 | September backwards to April statement amounts; retain legitimate zero/negative observations |
| PAY_AMT1 … PAY_AMT6 | September backwards to April prior payment amounts |
| default payment next month | Workbook target; loader canonicalizes its name explicitly |

The UCI metadata also exposes X1–X23/Y names; the pipeline reads the workbook's descriptive headers rather than assuming the metadata labels are the workbook schema. Missing values, duplicate IDs, target counts and unusual-code frequencies are checked in the saved data audit. No undocumented category is silently imputed or merged.

## Provenance and public outputs

The original ZIP and XLS remain unchanged under ignored `data/raw/`. [`reports/source_manifest.json`](../reports/source_manifest.json) records download URL, byte sizes and SHA-256 hashes. Expected hashes:

- ZIP: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`
- XLS: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`

Public deployment outputs are aggregate diagnostics and reduced analytical artifacts, with identifiers and unnecessary individual fields removed. Synthetic examples are labeled and are not customer records. The app does not accept uploads of customer information. Any feature transformations and aggregated outputs are adaptations of the attributed source, not a corrected original workbook.

## Scope

This historical cohort is not representative evidence for a contemporary European lender. No additional consent history or reidentification guarantee is inferred from public availability. See [responsible use](FAIRNESS_AND_RESPONSIBLE_AI.md) and [limitations](LIMITATIONS.md).

## Validated source audit

Actual audit: 6,636 defaults / 30,000 observations (22.12%); zero missing cells and zero duplicate IDs. There are 56 repeated predictor patterns, which remain distinct customer identifiers rather than being silently deduplicated. There are 3,932 negative bill cells across the six bill fields. These may influence dependence/generalization; unique IDs do not prove independence.

| Partition | Customers | Events |
|---|---:|---:|
| training | 15,000 | 3,318 |
| validation | 4,500 | 995 |
| calibration | 4,500 | 996 |
| test | 6,000 | 1,327 |

Exact identifier-set SHA-256 hashes are in the saved audit; local identifier arrays are ignored. Undocumented education codes 0/5/6 occur 14/280/51 times, marriage code 0 occurs 54 times. The category-frequency audit preserves every repayment code without inventing meanings.
