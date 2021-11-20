#include "readSlicePart.cuh"

void readSlicePart(cunfftplan1d plan,CUFFTDATATYPE *fhat,CUFFTDATATYPE *bfCU,FILE *infile,MrcHeader *inhead,int yNum)
{
    int i,k;
    float *slice;
    slice = (float*)malloc(inhead->nx*inhead->nz*sizeof(float));
    CUFFTDATATYPE* sliceC = (CUFFTDATATYPE*)malloc(inhead->nx*inhead->nz*sizeof(CUFFTDATATYPE));
    // read slice part
    mrc_read_slice(infile, inhead, yNum, 'y', slice);
    //transform

    for (i = 0 ; i < inhead->nz ; i++)
    {
        for (k = 0 ; k < inhead->nx ; k++)
        {
            sliceC[k].x = (DATATYPE)slice[i*inhead->nx+k];
            sliceC[k].y = 0;
        }
        cudaMemcpy(fhat,sliceC,inhead->nx*sizeof(CUFFTDATATYPE),cudaMemcpyHostToDevice);
	cunfft_trafo_1d(plan,bfCU+i*inhead->nx,fhat);
    }
    /*CUFFTDATATYPE * f = (CUFFTDATATYPE*)malloc(inhead->nx*inhead->nx*sizeof(CUFFTDATATYPE));
    cudaMemcpy(f,bfCU,inhead->nx*inhead->nz*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
    for (i = 0 ; i < 20 ; i++)
	printf("%.20lf %.20lf\n",f[i].x,f[i].y);
    free(f);
    exit(0);*/
    free(slice);
    free(sliceC);
}





