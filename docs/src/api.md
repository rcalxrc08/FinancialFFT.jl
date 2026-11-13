```@raw html
<div class="fc-page-head">
<p class="fc-kicker"><span>Reference</span> Every exported name</p>
```

# API reference


```@raw html
<nav class="fc-toc">
<a href="#Methods"><b>01</b>Methods</a>
<a href="#Densities"><b>02</b>Densities</a>
<a href="#Pricing"><b>03</b>Pricing</a>
<a href="#Index"><b>04</b>Index</a>
</nav>
</div>
```
## Methods

```@docs
LewisMethod
CosMethod
CarrMadanLewisMethod
CarrMadanMethod
CosDensityMethod
FFTDensityInverter
```

## Densities

```@docs
density
FinancialFFT.density_values
FinancialFFT.AbstractDensity
FinancialFFT.FFTDensity
FinancialFFT.CosDensity
```

## Pricing

```@docs
pricer(::FinancialMonteCarlo.BaseProcess, ::FinancialMonteCarlo.AbstractZeroRateCurve, ::LewisMethod, ::FinancialFFT.FourierPayoff)
```

## Index

```@index
```
