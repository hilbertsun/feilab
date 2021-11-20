#include <stdio.h>
#include <math.h>
#include "reProject_NFFT.cuh"


void reProject_NFFT(cunfftplan1d plan1d,cunfftplan2d plan2d,GICONPara GP,DATATYPE *reprojection)
{

    //2D NFFT transform
    cunfft_trafo_2d_R2C(plan2d,GP.fCU,GP.mCU);
    /*cudaMemcpy(reprojection,GP.fCU,plan2d.M*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
    int i;
    for (i = 0 ; i < 20 ; i++)
		printf("********%.20lf %.20lf\n",reprojection[i].x,reprojection[i].y);*/
    //1D NFFT transform
    cunfft_adjoint_1d(plan1d,GP.fCU,GP.fhat1dCU);
    cudaMemcpy(GP.reprojection,GP.fhat1dCU,plan1d.N*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
    int i;
    for (i = 0 ; i < plan1d.N ; i++)
	reprojection[i] = GP.reprojection[i].x;
    /*for (i = 0 ; i < 20 ; i++)
		printf("oooooooo%.20lf %.20lf\n",reprojection[i].x,reprojection[i].y);*/
}

void reProject_NFFT2(cunfftplan1d plan1d,cunfftplan2d2 plan2d,GICONPara GP,DATATYPE *reprojection)
{

    //2D NFFT transform
    cunfft_trafo_2d_R2C2(plan2d,GP.fCU,GP.mCU);
    /*cudaMemcpy(reprojection,GP.fCU,plan2d.M*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
    int i;
    for (i = 0 ; i < 20 ; i++)
		printf("********%.20lf %.20lf\n",reprojection[i].x,reprojection[i].y);*/
    //1D NFFT transform
    cunfft_adjoint_1d(plan1d,GP.fCU,GP.fhat1dCU);
    cudaMemcpy(GP.reprojection,GP.fhat1dCU,plan1d.N*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
    int i;
    for (i = 0 ; i < plan1d.N ; i++)
	reprojection[i] = GP.reprojection[i].x;
    /*for (i = 0 ; i < 20 ; i++)
		printf("oooooooo%.20lf %.20lf\n",reprojection[i].x,reprojection[i].y);*/
}
