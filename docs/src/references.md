# References

## Fourier pricing

- P. Carr and D. B. Madan, *Option Valuation Using the Fast Fourier Transform*, Journal of Computational Finance 2(4),
  61–73, 1999. The damped call transform and its FFT ([`CarrMadanMethod`](@ref)).
- A. L. Lewis, *A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes*, 2001,
  [SSRN 282110](https://ssrn.com/abstract=282110). The integral along the line ``\operatorname{Im} u = 1/2``
  ([`LewisMethod`](@ref), [`CarrMadanLewisMethod`](@ref)).
- F. Fang and C. W. Oosterlee, *A Novel Pricing Method for European Options Based on Fourier-Cosine Series
  Expansions*, SIAM Journal on Scientific Computing 31(2), 826–848, 2008. The COS method ([`CosMethod`](@ref),
  [`CosDensityMethod`](@ref)), and the rule of the cumulants for its range that the package replaces with a Chernoff
  bound ([Truncation range](@ref)).
- C. W. Clenshaw, *A Note on the Summation of Chebyshev Series*, Mathematical Tables and Other Aids to Computation
  9(51), 118–120, 1955. The recurrence that sums the cosine series of a smile, and of a density, at one `sincos` per
  point.

## Models

- R. C. Merton, *Option Pricing when Underlying Stock Returns Are Discontinuous*, Journal of Financial Economics
  3(1–2), 125–144, 1976.
- S. G. Kou, *A Jump-Diffusion Model for Option Pricing*, Management Science 48(8), 1086–1101, 2002.
- D. B. Madan, P. Carr and E. C. Chang, *The Variance Gamma Process and Option Pricing*, European Finance Review 2(1),
  79–105, 1998.
- O. E. Barndorff-Nielsen, *Processes of Normal Inverse Gaussian Type*, Finance and Stochastics 2(1), 41–68, 1998.
- S. L. Heston, *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency
  Options*, Review of Financial Studies 6(2), 327–343, 1993.
- H. Albrecher, P. Mayer, W. Schoutens and J. Tistaert, *The Little Heston Trap*, Wilmott Magazine, January 2007,
  83–92. The form of the Heston characteristic function that stays continuous in the maturity, used by the package.
- L. B. G. Andersen and V. V. Piterbarg, *Moment Explosions in Stochastic Volatility Models*, Finance and Stochastics
  11(1), 29–50, 2007. The time at which the moments of Heston explode, which bounds its truncation range.

## Tail bounds

- H. Chernoff, *A Measure of Asymptotic Efficiency for Tests of a Hypothesis Based on the Sum of Observations*, Annals
  of Mathematical Statistics 23(4), 493–507, 1952. The exponential bound on the tails ([Chernoff bounds](@ref)).
- H. Cramér, *Sur un nouveau théorème-limite de la théorie des probabilités*, Actualités Scientifiques et
  Industrielles 736, 5–23, 1938. The rate of large deviations, the Legendre transform of the cumulant generating
  function.
- W. Squire and G. Trapp, *Using Complex Variables to Estimate Derivatives of Real Functions*, SIAM Review 40(1),
  110–112, 1998. The complex step that gives the cumulant generating function and its derivative in one evaluation
  ([Finding the minimum](@ref)).
- R. P. Brent, *Algorithms for Minimization without Derivatives*, Prentice-Hall, 1973, Chapter 4. The root finder of
  the truncation range.

## Calibration

- K. Levenberg, *A Method for the Solution of Certain Non-Linear Problems in Least Squares*, Quarterly of Applied
  Mathematics 2(2), 164–168, 1944.
- D. W. Marquardt, *An Algorithm for Least-Squares Estimation of Nonlinear Parameters*, Journal of the Society for
  Industrial and Applied Mathematics 11(2), 431–441, 1963. The damping scaled by the diagonal of ``J^\top J``
  ([Levenberg–Marquardt](@ref)).
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd edition, Springer, 2006, Chapter 10: nonlinear least
  squares, Gauss–Newton and Levenberg–Marquardt.
- P. Jäckel, *Let's Be Rational*, Wilmott 2015(75), 40–53, 2015. The implied volatility of `blsimpv` in
  FinancialToolbox.jl, used for the model and quoted volatilities.

## Implicit function theorem

- W. Rudin, *Principles of Mathematical Analysis*, 3rd edition, McGraw-Hill, 1976, Theorem 9.28.
- S. G. Krantz and H. R. Parks, *The Implicit Function Theorem: History, Theory, and Applications*, Birkhäuser, 2002.

It gives the derivatives of the model volatilities ([Derivatives of the model volatilities](@ref)) and the sensitivities
of the calibrated parameters to the quotes ([Sensitivity of the calibrated parameters to the quotes](@ref)) without
differentiating a solver.

## Sensitivities from densities

- M. Broadie and P. Glasserman, *Estimating Security Price Derivatives Using Simulation*, Management Science 42(2),
  269–285, 1996. The likelihood ratio method, whose weights are the scores of a [`density`](@ref).
- M. B. Giles, *Vibrato Monte Carlo Sensitivities*, in Monte Carlo and Quasi-Monte Carlo Methods 2008, Springer, 2009,
  369–382.
