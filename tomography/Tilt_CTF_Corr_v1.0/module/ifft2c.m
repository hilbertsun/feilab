function out = ifft2c( w )
%usage: generate correct ifft2 result condiserting fftshift
%Date: Fri, Nov 28th ,2014
%Author: Yuchen Deng.

out=ifftshift(ifft2(ifftshift(w)));

end

