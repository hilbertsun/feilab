function point_trans = point_trans( transParam,points,bigSize )
rotMatrix=[transParam(1,1),transParam(1,2); ...
    transParam(1,3),transParam(1,4)];
%rotMatrix=[1,0;0,1];
shift=transParam(1,5:6)';
[~,num]=size(points);
points_centre=points-[(bigSize/2+1);(bigSize/2+1)]*ones(1,num);
point_trans=rotMatrix*points_centre ...
    +(shift+[(bigSize/2+1);(bigSize/2+1)])*ones(1,num);


end

