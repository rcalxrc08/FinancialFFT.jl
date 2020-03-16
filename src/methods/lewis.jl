"""
Struct for Lewis Integration Method

		bsProcess=LewisMethod(σ::num1) where {num1 <: Number}

Where:\n
		σ	=	volatility of the process.
"""
mutable struct LewisMethod{ num <: Number , num_1 <: Integer}<:FinancialMonteCarlo.AbstractMethod
	A::num
	N::num_1
	function LewisMethod(A::num,N::num_1) where {num <: Number, num_1 <: Integer}
        if A <= 0.0
            error("A must be positive")
        elseif N <= 2
            error("N must be greater than 2")
        else
            return new{num,num_1}(A,N)
        end
    end
end

function LewisPricer(mcProcess::FinancialMonteCarlo.BaseProcess,K::Number,r::Number,T::Number,N::Integer,A::Real,d::Number=0.0)
	S0=mcProcess.underlying.S0;
	CharExp(v)=FinancialFFT.CharactheristicExponent(v,mcProcess);
    EspChar(v)= CharExp(v)-v*1im*CharExp(-1im);
    CharFunc(v)= exp(T*EspChar(v));
	x__=log(S0/K)+(r-d)*T
	func_(z)=exp(-z*1im*x__)*CharFunc(-z-1im*0.5)/(z^2+0.25);
	int_1=real(integral_1(func_,-A,A,N));
	price=S0*(1-exp(-x__/2)*int_1/(2*pi))*exp(-d*T)
	return price;
end

function integral_1(f,xmin,xmax,N)
	x=range(xmin,length=N,stop=xmax);
	dx=x[2]-x[1];
	sum_=f(x[1])*dx*0.0;
	for x_ in x
		sum_+=f(x_)*dx;
	end
	return sum_;
end



function pricer(mcProcess::FinancialMonteCarlo.BaseProcess,spotData::FinancialMonteCarlo.AbstractZeroRateCurve,method::LewisMethod,abstractPayoffs_::Array{U})where {U <: FinancialMonteCarlo.AbstractPayoff}

	S0=mcProcess.underlying.S0;
	r=spotData.r;
	d=mcProcess.underlying.d;
	A=method.A
	N=method.N

	f1(x::T1) where T1 =(T1<:EuropeanOption);
	abstractPayoffs=filter(f1,abstractPayoffs_);
	
	TT=unique([opt.T for opt in abstractPayoffs]);
	prices=Array{Number}(undef,length(abstractPayoffs_));

	for T in TT
		index_same_t=findall(op->(op.T==T && f1(op)),abstractPayoffs_);
		payoffs=abstractPayoffs_[index_same_t];
		strikes=[opt.K for opt in payoffs];
		r_tmp=FinancialMonteCarlo.integral(r,T)/T;
		d_tmp=FinancialMonteCarlo.integral(d,T)/T;
		prices[index_same_t].=LewisPricer.(mcProcess,strikes,r_tmp,T,N,A,d_tmp);
	end

	length(abstractPayoffs) < length(abstractPayoffs_) ? (return prices) : (return prices*1.0);
end



function pricer(mcProcess::FinancialMonteCarlo.BaseProcess,spotData::FinancialMonteCarlo.AbstractZeroRateCurve,method::LewisMethod,abstractPayoff::FinancialMonteCarlo.EuropeanOption)

	S0=mcProcess.underlying.S0;
	r=spotData.r;
	r_tmp=FinancialMonteCarlo.integral(r,abstractPayoff.T)/abstractPayoff.T;
	d=mcProcess.underlying.d;
	A=method.A
	N=method.N

	return LewisPricer(mcProcess,abstractPayoff.K,r_tmp,abstractPayoff.T,N,A,d);
end

	
	
