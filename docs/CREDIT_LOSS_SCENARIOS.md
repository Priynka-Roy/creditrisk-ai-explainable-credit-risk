# Illustrative next-month loss scenarios

For a hypothetical portfolio, modeled expected loss is

`mean(estimated next-month payment default probability) × hypothetical LGD × hypothetical EAD per account × account count`.

LGD is a fraction between 0 and 1, EAD is an assumed currency amount per account, and the portfolio size is a count. The result has the same monetary units as EAD. Credit limit is not substituted for measured EAD. Neither actual LGD nor actual EAD is available from the dataset.

## Stress transformation

Stress shifts odds rather than multiplying and clipping probabilities:

`p_stressed = logistic(logit(p) + log(odds multiplier))`.

A multiplier of 1 preserves baseline probabilities, greater than 1 increases odds, and the transformation preserves ordering for finite interior probabilities. Exact 0/1 probabilities remain 0/1; computational clipping only stabilizes the logit calculation. Invalid probabilities, negative exposure/count, LGD outside [0,1], or a nonpositive/nonfinite multiplier raise an error. Zero LGD or EAD gives zero modeled loss.

Baseline, mild and severe scenarios are **assumptions**, not estimated recession shocks. The displayed probabilities come from a fitted historical model, while LGD/EAD/count and odds shifts are chosen illustrative inputs. Averaging segment probability distributions changes hypothetical mix; it does not estimate a current lender's borrower mix. No independence assumption is needed for the arithmetic expectation; no loss distribution or tail capital calculation is implied.

Saved scenario results and the guided case are documented in [portfolio case study](PORTFOLIO_CASE_STUDY.md). The dashboard permits sensitivity analysis. It does not produce IFRS 9 expected credit loss, multi-period survival probabilities, capital requirements, realized bank losses or lending advice.

## Saved scenario results

Each scenario assumes 1,000 accounts and EAD of 100,000 NTD per account. Results use exact holdout probabilities.

| Assumption | Odds multiplier | LGD | Mean stressed probability | Modeled loss (NTD) |
|---|---:|---:|---:|---:|
| Baseline | 1.0 | 45% | 22.0810% | 9,936,454 |
| Mild adverse | 1.5 | 45% | 27.9536% | 12,579,103 |
| Severe adverse | 2.0 | 60% | 32.6539% | 19,592,343 |

The interactive dashboard uses an anonymous probability histogram rounded to 0.001, so interactive sensitivity arithmetic can differ slightly from the exact saved table. These are modeled assumed losses, never observed losses.
