function lena_ctf = ctf_transfer( lena,ac_volt,defocus,apix,CS, Bfactor,amp )
[image_size,tmp]=size(lena);
CTF2d=ctf_image(ac_volt,defocus,apix,image_size,CS, Bfactor,amp );
lena_ctf=ifft2c(fft2c(lena).*CTF2d);

end

