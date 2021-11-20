function tiltCTFcorrect(tilt_series_FileName, tilt_series_corr_FileName, ...
    tilt_xf_FileName, tilt_angle_FileName, tilt_defocus_FileName, inv_angle, ac_volt, ...
    apix, Cs, Bfactor, amp)

display(tilt_series_FileName);
display(tilt_series_corr_FileName);
display(tilt_xf_FileName);
display(tilt_angle_FileName);
display(tilt_defocus_FileName);

if ischar(inv_angle)
    inv_angle = str2num(inv_angle);
end

if ischar(ac_volt)
    ac_volt = str2double(ac_volt);
end

if ischar(apix)
    apix = str2double(apix);
end

if ischar(Cs)
    Cs = str2double(Cs);
end

if ischar(Bfactor)
    Bfactor = str2double(Bfactor);
end

if ischar(amp)
    amp = str2double(amp);
end

tilt_series = tom_mrcread(tilt_series_FileName);
tilt_series = double(tilt_series.Value);
series_size = size(tilt_series);
series_num = series_size(3);

tilt_corr_series=zeros(series_size(1),series_size(2),series_size(3));

transParam = trans_get( tilt_xf_FileName, tilt_angle_FileName);

defocus = load(tilt_defocus_FileName);

for i=1:series_num
    
    message = sprintf('phase flipping the %d th tilt micrograph.',i);
    display(message);
    tic
    tmp = singleTilt(tilt_series(:,:,i),256,2,defocus(i),transParam(i,:),...
        inv_angle,ac_volt,apix,Cs,Bfactor,amp);
    mode = getmode(tmp);
    tmp = tmp - mode;
    tilt_corr_series(:,:,i) = tmp;
    toc;
end

tom_mrcwrite(tilt_corr_series,'name', tilt_series_corr_FileName);

end

function out = fft2c( w )
%usage: generate correct fft2 result condiserting fftshift
%Date: Fri, Nov 28th ,2014
%Author: Yuchen Deng.

out=fftshift(fft2(fftshift(w)));

end

function out = ifft2c( w )
%usage: generate correct ifft2 result condiserting fftshift
%Date: Fri, Nov 28th ,2014
%Author: Yuchen Deng.

out=ifftshift(ifft2(ifftshift(w)));

end

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

function lena_fliped = phase_flip( lena,ac_volt,defocus,apix,CS, Bfactor,amp )
[image_size,tmp]=size(lena);
CTF2d=ctf_image(ac_volt,defocus,apix,image_size,CS, Bfactor,amp );

mask_neg=logical(CTF2d<0);
mask=~mask_neg;
mask_neg=mask_neg*-1;
mask=mask+mask_neg;
lena_fliped=ifft2c(fft2c(lena).*(mask));

end

function image_correted = wiener_filter(image,ac_volt,defocus,apix,CS, Bfactor,amp )
[image_size,tmp]=size(image);
CTF2d=ctf_image(ac_volt,defocus,apix,image_size,CS, Bfactor,amp );

image_fft = fft2c(image);
image_fft_flip = image_fft.*CTF2d;
CTF_weight = CTF2d.*CTF2d + 10.0;
image_fft_wiener = image_fft_flip./CTF_weight;

image_correted = ifft2c(image_fft_wiener);

end

function points = point_gens( smallSize,bigSize,overlapFactor )
startx=smallSize/2+1;
starty=smallSize/2+1;
endx=bigSize-smallSize/2+1;
endy=bigSize-smallSize/2+1;

[~,xsize] = size(startx:smallSize/overlapFactor:endx);

points=zeros(2,xsize^2);
count=0;
for i=startx:smallSize/overlapFactor:endx
    for j = starty:smallSize/overlapFactor:endy
        count=count+1;
        points(:,count)=[i;j];
    end
end
end

function point_trans = point_trans(transParam,points,bigSize )
rotMatrix=[transParam(1,1),transParam(1,2); ...
    transParam(1,3),transParam(1,4)];
%rotMatrix=[1,0;0,1];
shift=transParam(1,5:6)';
[~,num]=size(points);
points_centre=points-[(bigSize/2+1);(bigSize/2+1)]*ones(1,num);
point_trans=rotMatrix*points_centre ...
    +(shift+[(bigSize/2+1);(bigSize/2+1)])*ones(1,num);


end

function defocus_series  = defocus_gen(points_trans,bigSize,apix,transParam,...
meanDefocus, inv_angle )
[~,num]=size(points_trans);
defocus_series=zeros(1,num);
theta=transParam(1,7);
theta=deg2rad(theta);
apix=apix*1e-4;%A->um
for i =1:num
    if (inv_angle > 0 )    
        
        %inv_angle > 0 means that at the positive angle the absolute value
        %of defocus is larger at the right side than that at the left side.
        
        defocus_series(1,i)=meanDefocus + (points_trans(1,i)-bigSize/2+1)*apix*sin(theta);
    else
        
        %inv_angle < 0 means that at the positive angle the absolute value
        %of defocus is larger at the right side than that at the left side.
        
        defocus_series(1,i)=meanDefocus - (points_trans(1,i)-bigSize/2+1)*apix*sin(theta);
    end
    
end
end

function coor = cutCoor(smallSize,point )
coor=[point(1,1)-smallSize/2,point(1,1)+smallSize/2-1; ...
    point(2,1)-smallSize/2,point(2,1)+smallSize/2-1,];

end

function trans= trans_get(tilt_xf_FileName, tilt_angle_FileName)
%1 2 3 4 -- rotation matrix a11 a12 a21 a22(anticlockwise), 5 6 -- shift x
%y(x right, y down(up in imod)), 7 -- fixed tilt angle
xf=load(tilt_xf_FileName);
rot=load(tilt_angle_FileName);

[num,~]=size(xf);

trans=zeros(num,7);
trans(:,1:6)=xf;
trans(:,7)=rot;

end

function CTFcorr = singleTilt( orig,smallSize,overlapFactor,meanDefocus, ...
    transParam,inv_angle, ac_volt,apix,CS, Bfactor,amp )

%generate defocus set and points set.
[~,bigSize]=size(orig);
points_unrot=point_gens(smallSize,bigSize,overlapFactor);
point_rot=point_trans(transParam,points_unrot,bigSize);
defocus=defocus_gen(point_rot,bigSize,apix,transParam,meanDefocus,inv_angle);

[~,pointNum]=size(defocus);


%edge Check
maxX=max(points_unrot(1,:));
minX=min(points_unrot(1,:));
maxY=max(points_unrot(2,:));
minY=min(points_unrot(2,:));

CTFcorr=zeros(bigSize,bigSize);
for i = 1:pointNum
    coor=cutCoor(smallSize,points_unrot(:,i));
    image=orig(coor(1,1):coor(1,2), ...
        coor(2,1):coor(2,2));
    tmp=phase_flip(image,ac_volt,defocus(1,i),apix,CS, Bfactor,amp);
    %tmp=wiener_filter(image,ac_volt,defocus(1,i),apix,CS, Bfactor,amp);
    %display(defocus(1,i));
    %pasteback
    if points_unrot(1,i)==maxX
        CTFcorr(points_unrot(1,i)+smallSize/overlapFactor/2:coor(1,2), ...
        coor(2,1):coor(2,2)) ...
        =tmp(smallSize/2+smallSize/overlapFactor/2+1:smallSize,:);
    end
    
    if points_unrot(1,i)==minX
        CTFcorr(coor(1,1):points_unrot(1,i)-smallSize/overlapFactor/2-1, ...
        coor(2,1):coor(2,2)) ...
        =tmp(1:smallSize/2-smallSize/overlapFactor/2,:);
    end
    
    if points_unrot(2,i)==maxY
        CTFcorr(coor(1,1):coor(1,2), ...
        points_unrot(2,i)+smallSize/overlapFactor/2:coor(2,2)) ...
        =tmp(:,smallSize/2+smallSize/overlapFactor/2+1:smallSize);
    end
    
    if points_unrot(2,i)==minY
        CTFcorr(coor(1,1):coor(1,2), ...
        coor(2,1):points_unrot(2,i)-smallSize/overlapFactor/2-1) ...
        =tmp(:,1:smallSize/2-smallSize/overlapFactor/2);
    end
    
    CTFcorr(points_unrot(1,i)-smallSize/overlapFactor/2:points_unrot(1,i)+smallSize/overlapFactor/2-1, ...
        points_unrot(2,i)-smallSize/overlapFactor/2:points_unrot(2,i)+smallSize/overlapFactor/2-1) ...
        =tmp(smallSize/2+1-smallSize/overlapFactor/2:smallSize/2+1+smallSize/overlapFactor/2-1, ...
        smallSize/2+1-smallSize/overlapFactor/2:smallSize/2+1+smallSize/overlapFactor/2-1);
    
end


end

function mode = getmode(image)
    [w,b]=hist(image(:),50);
    [~,c]=max(w(:));
    mode=b(1,c);
end

