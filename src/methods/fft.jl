using DualNumbers,FFTW
import FFTW.fft;
function fft(x::Array{DualComplex256})
	Xcomplex=DualNumbers.value.(x);
	Xder=epsilon.(x);
	fft!(Xcomplex);
	fft!(Xder)
	Yout=[DualComplex256(real1,epsilon1) for (real1,epsilon1) in zip(Xcomplex,Xder)];

	return Yout;
end
