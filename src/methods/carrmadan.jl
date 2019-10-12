import FinancialMonteCarlo.pricer, FinancialMonteCarlo.AbstractMethod;

"""
Struct for Carr Madan Method

		bsProcess=CarrMadanMethod(σ::num1) where {num1 <: Number}

Where:\n
		σ	=	volatility of the process.
"""
mutable struct CarrMadanMethod{ num <: Number , num_1 <: Integer}<:FinancialMonteCarlo.AbstractMethod
	A::num
	Npow::num_1
	function CarrMadanMethod(A::num,Npow::num_1) where {num <: Number, num_1 <: Integer}
        if A <= 0.0
            error("A must be positive")
        elseif Npow <= 2
            error("Npow must be greater than 2")
        else
            return new{num,num_1}(A,Npow)
        end
    end
end

using Interpolations,FFTW;
"""
Pricing European Options through Fast Fourier Transform Method (Carr Madan)

		VectorOfPrice=CarrMadanPricer(mcProcess::FinancialMonteCarlo.BaseProcess,S0::Number,StrikeVec::Array{U},r::Number,T::Number,Npow::Integer,A::Real,d::Number=0.0) where {U <: Number}

Where:\n
		S0 = Spot price.
		StrikeVec = Vector of Strike of the Option to price.
		r= zero rate with tenor T.
		T= tenor of the options.
		CharExp= characteristic function of the model that you want to use.
		Npow= Integer Parameter for the FFT. Represent the log2 of the number of nodes.
		A= Real Parameter of the FFT. Represent a maximum for the axis.

		VectorOfPrice= Price of the European Options with Strike equals to StrikeVec, tenor T and the market prices a risk free rate of r.
"""
function CarrMadanPricer(mcProcess::FinancialMonteCarlo.BaseProcess,S0::Number,StrikeVec::Array{U},r::Number,T::Number,Npow::Integer,A::Real,d::Number=0.0) where {U <: Number}
    N=2^Npow;
	CharExp=CharactheristicExponent(mcProcess);
    EspChar(v::Number)::Number= CharExp(v)-v.*1im*CharExp(-1im);
    #v-> compute integral as a summation
    eta1=A/N;
    v=collect(0:eta1:A*(N-1)/N);
    v[1]=1e-22;
    # lambda-> compute summation via FFT
    lambda=2*pi/(N*eta1);
    k=-lambda*N/2.0.+lambda.*(0:N-1);
    CharFunc(v::Number)::Number= exp(T*EspChar(v));
    Z_k=exp.(1im*(r-d)*v*T).*(CharFunc.(v.-1im).-1.0)./(1im*v.-v.^2)
    # Option Price
    w=ones(N); w[1]=0.5; w[end]=0.5;
	#Z_k=w.*eta1.*Z_k.*exp.(1im*pi*(0:N-1));
    Z_k=w.*eta1.*Z_k.*[i%2==1 ? 1 : -1 for i in 1:N];
    w=real(fft(Z_k))/pi;
    C=S0.*(w+max.(1.0.-exp.(k.-(r-d)*T),0));
    K=S0.*exp.(k);
    idx1=findfirst(x-> x>0.4*S0,K);
    idx2=findlast(x-> x<3.0*S0,K);
    index=idx1:idx2;
	priceInterpolator = interpolate((K[index],), C[index], Gridded(Linear()))
    VectorOfPrice=exp(-d*T).*priceInterpolator.(StrikeVec);

	return VectorOfPrice;
end


function pricer(mcProcess::FinancialMonteCarlo.BaseProcess,spotData::FinancialMonteCarlo.equitySpotData,method::CarrMadanMethod,abstractPayoffs_::Array{U})where {U <: FinancialMonteCarlo.AbstractPayoff}

	S0=spotData.S0;
	r=spotData.r;
	d=spotData.d;
	A=method.A
	Npow=method.Npow

	f1(x::T) where T =(T<:EuropeanOption);
	abstractPayoffs=filter(f1,abstractPayoffs_);
	
	TT=unique([opt.T for opt in abstractPayoffs]);
	prices=Array{Number}(undef,length(abstractPayoffs_));

	for T in TT
		index_same_t=findall(op->(op.T==T && f1(op)),abstractPayoffs_);
		payoffs=abstractPayoffs_[index_same_t];
		strikes=[opt.K for opt in payoffs];
		prices[index_same_t]=CarrMadanPricer(mcProcess,S0,strikes,r,T,Npow,A,d);
	end

	length(abstractPayoffs) < length(abstractPayoffs_) ? (return prices) : (return prices*1.0);
end



function pricer(mcProcess::FinancialMonteCarlo.BaseProcess,spotData::FinancialMonteCarlo.equitySpotData,method::CarrMadanMethod,abstractPayoff::FinancialMonteCarlo.EuropeanOption)

	S0=spotData.S0;
	r=spotData.r;
	d=spotData.d;
	A=method.A
	Npow=method.Npow

	return CarrMadanPricer(mcProcess,S0,[abstractPayoff.K],r,abstractPayoff.T,Npow,A,d)[1];
end
