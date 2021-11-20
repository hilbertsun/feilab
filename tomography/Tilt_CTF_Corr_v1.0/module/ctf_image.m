function CTF2d= ctf_image(ac_volt,defocus,apix,image_size,CS, Bfactor,amp )
CTF = ctf_function( ac_volt,defocus,apix,image_size,CS, Bfactor,amp);
N=image_size;
bigSize=N;
w=CTF(N/2+1:N);
x_index=ones(bigSize,1)*(1:1:bigSize)-bigSize/2-1;
y_index=x_index';
radius=round(sqrt(x_index.^2+y_index.^2))+1;
CTF2d=zeros(bigSize,bigSize);

for i = 1: bigSize
    for j=1: bigSize
        if radius(i,j)<=N/2
            CTF2d(i,j)=w(1,radius(i,j));
        end
    end
end
end
