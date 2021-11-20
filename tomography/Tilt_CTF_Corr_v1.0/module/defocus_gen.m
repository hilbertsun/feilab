function defocus_series  = defocus_gen( points_trans,bigSize,apix,transParam,meanDefocus )
[~,num]=size(points_trans);
defocus_series=zeros(1,num);
theta=transParam(1,7);
theta=deg2rad(theta);
apix=apix*1e-4;%A->um
for i =1:num
    defocus_series(1,i)=meanDefocus-(points_trans(1,i)-bigSize/2+1)*apix*sin(theta);
end
end

