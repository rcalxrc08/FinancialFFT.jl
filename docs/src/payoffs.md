# Payoffs

Payoffs are defined in FinancialMonteCarlo.jl. Every method prices the following ones; any other payoff, such as a
barrier option, raises a `MethodError`.

| Payoff | Constructor | Result |
|:--|:--|:--|
| European option | `EuropeanOption(T, K, iscall = true)` | number |
| Binary option | `BinaryEuropeanOption(T, K, iscall = true)` | number |
| Smile of European options | `FinancialMonteCarlo.EuropeanOptionSmile(T, Ks, iscalls)` | vector, one price per strike |
| Smile of binary options | `FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, Ks, iscalls)` | vector, one price per strike |

```@setup payoffs
using FinancialFFT, FinancialMonteCarlo
S0, r, d, σ = 100.0, 0.02, 0.01, 0.2
model = BlackScholesProcess(σ, Underlying(S0, d))
zero_rate = ZeroRate(r)
method = CosMethod(2^12)
```

## Single options

```@example payoffs
pricer(model, zero_rate, method, EuropeanOption(1.0, 105.0)), pricer(model, zero_rate, method, BinaryEuropeanOption(1.0, 105.0, false))
```

## Smiles

A smile has one maturity and several strikes, each with its own call or put flag. This is how to price several strikes:
`pricer` does not take a vector of strikes.

```@example payoffs
smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, [90.0, 100.0, 110.0], [false, true, true])
pricer(model, zero_rate, method, smile)
```

```@example payoffs
smile = FinancialMonteCarlo.BinaryEuropeanOptionSmile(1.0, [90.0, 100.0, 110.0], [false, true, true])
pricer(model, zero_rate, method, smile)
```

## Different maturities

Options with different maturities are priced by broadcasting, and `pricer` does not take an array of options.

```@example payoffs
options = [EuropeanOption(0.5, 100.0), BinaryEuropeanOption(1.0, 105.0), EuropeanOption(2.0, 110.0, false)]
pricer.(model, zero_rate, method, options)
```

## Puts

Each method prices one of the two and gets the other from the put–call parity, so calls and puts share their accuracy,
and their cost when they are in the same smile. `LewisMethod` and the Carr-Madan methods compute calls. `CosMethod`
and the two density methods compute puts: a put is bounded by its strike, so the probability left outside their
[truncation range](@ref "Truncation range") costs at most that much, while the coefficients of a call grow like the
exponential of the upper end of the range, so a call would need the right tail under the share measure.
