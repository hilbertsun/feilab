function out = fft2c( w )
%usage: generate correct fft2 result condiserting fftshift
%Date: Fri, Nov 28th ,2014
%Author: Yuchen Deng.

out=fftshift(fft2(fftshift(w)));

end

