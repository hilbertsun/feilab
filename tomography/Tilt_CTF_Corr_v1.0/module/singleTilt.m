function CTFcorr = singleTilt( orig,smallSize,overlapFactor,meanDefocus, ...
    transParam,ac_volt,apix,CS, Bfactor,amp )

%generate defocus set and points set.
[~,bigSize]=size(orig);
points_unrot=point_gens(smallSize,bigSize,overlapFactor);
point_rot=point_trans(transParam,points_unrot,bigSize);
defocus=defocus_gen(point_rot,bigSize,apix,transParam,meanDefocus);

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

