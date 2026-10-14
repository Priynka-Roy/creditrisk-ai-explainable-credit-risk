# Hypothetical Monte Carlo credit-loss simulation

The target probabilities remain **next-month payment default**. EAD, LGD, dependence and stress are hypothetical. Currency is NTD; there is no silent EUR conversion and no realized banking loss claim.

## Default construction

For fixed synthetic account probabilities pᵢ and assumed exposure/loss fractions, portfolio loss is `L = Σ Dᵢ × EADᵢ × LGDᵢ`. Its theoretical expectation is `Σ pᵢ × EADᵢ × LGDᵢ`.

Independent defaults use rho=0. The dependent extension uses independent standard-normal Z and εᵢ:

`Dᵢ = 1[√rho × Z + √(1−rho) × εᵢ < Φ⁻¹(pᵢ)]`.

The latent variable remains marginally standard normal, so each unconditional default probability remains pᵢ. Rho is an assumed **latent asset correlation**, not an estimated Bernoulli default correlation. At rho=1 all uncertainty is systematic; exact p=0 and p=1 endpoints remain deterministic. Dependence changes portfolio variation/tails without changing the analytic expected loss when p/EAD/LGD are unchanged.

The one-factor idea is described in the [BIS explanatory note on IRB risk-weight functions](https://www.bis.org/bcbs/irbriskweight.pdf). This project uses an educational finite-portfolio simulation with assumed rho; it does not implement or reproduce regulatory capital requirements.

## Portfolio and simulation assumptions

The anonymous V1 histogram rounds probabilities to 0.001. A deterministic midpoint-quantile selection creates the chosen number of synthetic accounts. This is an illustrative mix, not a sample of actual new borrowers. Default simulation holds those account probabilities fixed; different size/mix settings can slightly alter mean probability through quantile discretization.

Default release runs use 1,000 accounts, 5,000 portfolios, seed 442 and batches of 250. EAD 100,000 NTD/account and baseline LGD 45% are assumptions. Odds stress preserves ordering and changes probabilities by `logistic(logit(p)+log(multiplier))`; it is not multiplication of probabilities. Higher LGD/exposure and alternative assumed factor dependence are separate scenario controls.

## Tail conventions

For m simulated nonnegative losses sorted ascending, empirical **VaR at alpha** is the inverse empirical CDF: element `ceil(alpha × m)`, with one-based indexing. It is a loss quantile, not capital net of expected loss.

**Expected Shortfall at alpha** is the mean of the upper `(1−alpha)` empirical probability mass. Take the largest `floor((1−alpha)m)` observations and a fractional weight of the next observation when required; divide by `(1−alpha)m`. This handles discreteness and ties correctly. It is not the mean of every observation satisfying `loss >= VaR`, which can unintentionally include excessive probability mass at ties. Alpha must lie strictly between 0 and 1. For fewer than two simulations, reported SD/mean-SE is0 under the explicit implementation convention.

Monte Carlo mean standard error is SD/√m. It describes numerical sampling uncertainty conditional on assumptions, not fitted-model uncertainty or economic-scenario probability. Tail quantiles and ES also have finite-run uncertainty; a displayed point estimate is not an exact population risk measure. No VaR/ES confidence interval is claimed where not computed.

## Numerical controls

Finite bounded PD/LGD, nonnegative exposure, exact endpoints/zero loss, reproducible RNG and batching are tested. The app bounds workload; there is no training or raw-source download during navigation. Checks compare simulated marginal defaults and mean loss to intended theoretical values using Monte Carlo tolerances. Different realized simulated means across rho scenarios are sampling variation, not an analytical expected-loss change.

Actual scenario tables appear in [V2 results](V2_RESULTS.md). These assumptions omit real contagion, multiple economic factors, LGD/EAD randomness, empirical default dependence and measured macroeconomic stress. They are not IFRS9 loss estimates, Basel capital calculations or eligibility recommendations.
