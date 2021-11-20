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

