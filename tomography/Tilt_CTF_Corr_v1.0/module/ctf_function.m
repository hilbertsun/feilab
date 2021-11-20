function CTF = ctf_function( ac_volt,defocus,apix,image_size,CS, Bfactor,amp)

%ac_volt(kV)
%defocus(um), defocus is positive 
%apix(angstrom)
%CS(mm)
%Bfactor(ang-2)
%amp
ac_volt=ac_volt*1000;
defocus=defocus*1e-6;
apix=apix*1e-10;
N=image_size;
CS=CS*1e-3;
Bfactor=Bfactor*(1e-20);

%Calculate wavelength of electron, with respect to Special Theory of
%Relativity
m0=9.10938291*(10^-31);%static matter of electron, (kg)
e=1.602176565*(1e-19);
h=6.62606957*(1e-34);%Plank Constant, J.s
c=299792458;%speed of light, m/s
wavelength=h/sqrt(2*e*ac_volt*m0+(e*ac_volt)^2/(c^2));

%generate spatial frequency
freq=-1/(2*apix):1/(N*apix):1/(2*apix)-1/(N*apix);

%generate envelope function
env=exp(-Bfactor*(freq.^2));

%CTF
gamma_func=3.1415926*wavelength*(freq.^2).*(defocus-0.5*(wavelength^2)*(freq.^2)*CS);
CTF=sqrt(1-amp^2)*env.*sin(gamma_func)+env.*amp.*cos(gamma_func);

end

