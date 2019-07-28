
function CharactheristicExponent(mcProcess::FinancialMonteCarlo.BlackScholesProcess)

	σ=mcProcess.σ;

	CharExp(v::Number)::Number=0.5*v*σ.*σ*(1im-v);

	return CharExp;

end

function CharactheristicExponent(mcProcess::FinancialMonteCarlo.MertonProcess)

	σ=mcProcess.σ;
	sigma1=mcProcess.σJ
	mu1=mcProcess.μJ
	lam=mcProcess.λ

	CharExp(u::Number)::Number=-σ*σ*u*u*0.5+lam*(exp(-sigma1*sigma1*u*u*0.5+1im*mu1*u)-1);

	return CharExp;

end

function CharactheristicExponent(mcProcess::FinancialMonteCarlo.KouProcess)

	σ=mcProcess.σ;
	lamp=mcProcess.λp
	lamm=mcProcess.λm
	p=mcProcess.p
	lam=mcProcess.λ

	CharExp(u::Number)::Number=-σ*σ*u*u/2.0+1im*u*p*(lam/(lamp-1im*u)-(1-lam)/(lamm+1im*u));

	return CharExp;

end

function CharactheristicExponent(mcProcess::FinancialMonteCarlo.VarianceGammaProcess)

	σ=mcProcess.σ;
	theta1=mcProcess.θ
	k1=mcProcess.κ

	CharExp(u::Number)::Number=-1/k1*log(1+u*u*σ*σ*k1/2.0-1im*theta1*k1*u);

	return CharExp;

end

function CharactheristicExponent(mcProcess::FinancialMonteCarlo.NormalInverseGaussianProcess)

	σ=mcProcess.σ;
	theta1=mcProcess.θ
	k1=mcProcess.κ

	CharExp(v::Number)=(1-1*sqrt(1.0 + ((v^2)*(σ*σ)-2*1im*theta1*v)*k1))/k1;

	return CharExp;

end