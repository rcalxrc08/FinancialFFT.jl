using DualNumbers,FFTW
import FFTW.fft;
function fft(x::Array{DualComplex256})
	N1=length(x);
	Xcomplex=DualNumbers.value.(x);
	Xder=epsilon.(x);
	fft!(Xcomplex);
	derY=zeros(N1);
	for i in 1:N1
		if !isapprox(real(Xder[i]),0.0)&&!isapprox(imag(Xder[i]),0.0)
			Xup=zeros(ComplexF64,N1);
			Xup[i]=1.0;
			fft!(Xup);
			derY+=Xup.*Xder[i];
		end
	end
	#Yout=[DualComplex256(Xcomplex[i],derY[i]) for i in 1:N1];
	Yout=[DualComplex256(real1,epsilon1) for (real1,epsilon1) in zip(Xcomplex,derY)];

	return Yout;
end
