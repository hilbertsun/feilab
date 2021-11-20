function lena_fliped = phase_flip( lena,ac_volt,defocus,apix,CS, Bfactor,amp )
[image_size,tmp]=size(lena);
CTF2d=ctf_image(ac_volt,defocus,apix,image_size,CS, Bfactor,amp );

mask_neg=logical(CTF2d<0);
mask=~mask_neg;
mask_neg=mask_neg*-1;
mask=mask+mask_neg;
lena_fliped=ifft2c(fft2c(lena).*(mask));

end

