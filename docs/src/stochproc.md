# [Stochastic Processes](@id stochproc)

The package support the following processes:

* `Black Scholes Process`
* `Heston Process`
* `Kou Process`
* `Merton Process`
* `Variance Gamma Process`
* `Normal Inverse Gaussian Process`
* `Lognormal Mixture Process`
* `Shifted Lognormal Mixture Process`

## Common Interface

A single method is implemented for each process, which provide the simulation output.

### Simulation

Each process behave in its own different way but returning the same kind of object after simulation,
the generic interfaces for simulating are the following:
DELETE HERE.
The following process are already implemented:
```@docs
CarrMadanLewisMethod
LewisMethod
CarrMadanMethod
```