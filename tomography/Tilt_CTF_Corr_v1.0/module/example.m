close all;
clear all;
% Sample test of 512_lena
load lena.mat;
CTF2d=ctf_image(300,6,1.41,512,2.7,150,0.07);
lena_ctf=ctf_transfer(lena,300,6,1.41,2.7,150,0.07);
lena_flip=phase_flip(lena_ctf,300,6,1.41,2.7,150,0.07);

figure
subplot(2,2,1)
imshow(lena,[])
title('Original Lena')


subplot(2,2,2)
imshow(CTF2d,[])
title('CTF curve')

subplot(2,2,3)
imshow(lena_ctf,[])
title('CTF modulated')

subplot(2,2,4)
imshow(lena_flip,[])
title('Phase flip corrected')

