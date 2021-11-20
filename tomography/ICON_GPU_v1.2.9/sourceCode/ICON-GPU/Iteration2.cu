#include "Iteration2.cuh"

void Iteration3(cunfftplan2d2 plan, GICONPara GP,MrcHeader *inhead,int iter_num,int Yvalue,char *resultpath,int *before,float *betain)
{
    clock_t btime, etime;
    int i,N[2],/*n[2],*/beforeupdate,/*threadnum,*/ beforeend,dataType,thickness;
    float threshold;


    beforeupdate = before[0];
    beforeend = before[1];
    //threadnum = before[2];
    dataType = before[3];
    threshold = betain[0];
    thickness = (int)betain[1];


    //float uptop;
    //DATATYPE thita[2];
    int M = inhead->nx*inhead->nz;
    N[0] = plan.N[0];
    N[1] = plan.N[1];
    //uptop = pow(2,ceil(log10(N[0])/log10(2.0)));
    //thita[0] = 2.0*uptop/(double)N[0];
    //uptop = pow(2,ceil(log10(N[1])/log10(2.0)));
    //thita[1] = 2.0*uptop/(double)N[1];

    //n[0] = ceil(thita[0]*N[0]);
    //n[1] = ceil(thita[1]*N[1]);


    btime = clock();
    cudaMemset(GP.mCU,0,N[0]*N[1]*sizeof(DATATYPE));
    //DATATYPE * f2 = (DATATYPE*)malloc(inhead->nx*inhead->nx*sizeof(DATATYPE));

    /*cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    float msecTotal1,msecTotal2;*/

    for (i = 0 ; i < iter_num ; i++)
    {
        //Am
	/*{
	DATATYPE * f2 = (DATATYPE*)malloc(N[0]*N[1]*sizeof(DATATYPE));
	cudaMemcpy(f2,GP.mCU,N[0]*N[1]*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("f_hat %.20lf\n",f2[ii]);
	free(f2);
	}*/
	//cudaEventRecord(start1, NULL);
	cunfft_trafo_2d_R2C2(plan,GP.fCU,GP.mCU);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("cunfft_trafo_2d_R2C2 : %f\n ",msecTotal1);*/
	/*{
	CUFFTDATATYPE * f = (CUFFTDATATYPE*)malloc(inhead->nx*inhead->nz*sizeof(CUFFTDATATYPE));
	cudaMemcpy(f,GP.fCU,inhead->nx*inhead->nz*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("f-- %.20lf %.20lf\n",f[ii].x,f[ii].y);
	free(f);
	}*/
        //wAm
	//cudaEventRecord(start1, NULL);
	complexMulDoubleCU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(GP.fCU,GP.weightCU,M);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("complexMulDoubleCU : %f\n ",msecTotal1);*/


        //AhwAm
	//cudaEventRecord(start1, NULL);
	cunfft_adjoint_2d2(plan,GP.fCU,GP.fhatCU);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("cunfft_adjoint_2d2 : %f\n ",msecTotal1);*/
	/*{
	CUFFTDATATYPE * f = (CUFFTDATATYPE*)malloc(N[0]*N[1]*sizeof(CUFFTDATATYPE));
	cudaMemcpy(f,GP.fhatCU,N[0]*N[1]*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("ooooo%.20lf %.20lf\n",f[ii].x,f[ii].y);
	free(f);
	}*/

	//cudaEventRecord(start1, NULL);
	complexMinusCU<<<dim3(GRIDDIMX,((N[0]*N[1] - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(GP.rhatCU,GP.fhatCU,GP.AhwbCU,N[0]*N[1]);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("complexMinusCU : %f\n ",msecTotal1);*/
	/*{	
	CUFFTDATATYPE * f = (CUFFTDATATYPE*)malloc(inhead->nx*inhead->nx*sizeof(CUFFTDATATYPE));
	cudaMemcpy(f,GP.rhatCU,inhead->nx*inhead->nx*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("rrrrrr%.20lf %.20lf\n",f[ii].x,f[ii].y);
	free(f);
	}*/
	/*cudaMemcpy(f,GP.AhwbCU,inhead->nx*inhead->nx*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("AAAAA%.20lf %.20lf\n",f[ii].x,f[ii].y);*/

	//cudaEventRecord(start1, NULL);
 	Alpha2(inhead,GP,plan,thickness);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("Alpha2 : %f\n ",msecTotal1);*/

	/*printf("al %.20lf\n",al);
	cudaMemcpy(f,GP.fhatCU,inhead->nx*inhead->nx*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("xxx%.20lf %.20lf\n",f[ii].x,f[ii].y);*/
	//printf("threshold %f\n",threshold);
	//cudaEventRecord(start1, NULL);
	if (beforeupdate <= i && i < iter_num - beforeend)
		ICONUpdateCU_R<<<dim3(GRIDDIMX,((N[0]*N[1] - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(GP.mCU,GP.rhatCU,GP.alCU,dataType,threshold,N[0]*N[1]);
	else
		INFRUpdateCU_R<<<dim3(GRIDDIMX,((N[0]*N[1] - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(GP.mCU,GP.rhatCU,GP.alCU,N[0]*N[1]);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("ICONUpdateCU_R : %f\n ",msecTotal1);*/
	/*cudaMemcpy(GP.m,GP.mCU,inhead->nx*inhead->nx*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("%.20lf\n",GP.m[ii]);*/
	/*cudaMemcpy(f,GP.mCU,inhead->nx*inhead->nx*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("%.20lf %.20lf\n",f[ii].x,f[ii].y);*/

    }
    //cudaThreadSynchronize();
    etime = clock();
    printf("slice %d finished, time %lf\n",Yvalue,(DATATYPE)(etime - btime) / CLOCKS_PER_SEC);
}



