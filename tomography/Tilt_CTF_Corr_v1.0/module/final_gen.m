final=zeros(4096,4096,21);

for i=1:21
    
    display(i);
    tic
    final(:,:,i)=singleTilt(test(:,:,i),256,2,defocus(i),transParamfull(i,:),300,2.27,2.7,30,0.1);
    toc;
end


