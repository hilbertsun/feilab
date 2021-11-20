function trans= trans_get(name)
%1 2 3 4 -- rotation matrix a11 a12 a21 a22(anticlockwise), 5 6 -- shift x
%y(x right, y down(up in imod)), 7 -- fixed tilt angle
xf=load(sprintf('%s.xf',name));
rot=load(sprintf('%s.tlt',name));

[num,~]=size(xf);

trans=zeros(num,7);
trans(:,1:6)=xf;
trans(:,7)=rot;


end

