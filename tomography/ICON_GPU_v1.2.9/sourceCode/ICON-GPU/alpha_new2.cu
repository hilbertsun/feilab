#include "alpha_new2.cuh"

void Alpha2(MrcHeader *inhead,GICONPara GP,cunfftplan2d2 plan,int thickness)
{
    int  N[2],/*n[2],*/size;


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

     /*cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    float msecTotal1,msecTotal2;*/

    //r^T*r
	//cudaEventRecord(start1, NULL);
    alpha0<<<dim3(GRIDDIMX,((N[0]*N[1] - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(GP.alphaTmpCU,GP.rhatCU,N[0]*N[1]);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("alpha0 : %f\n ",msecTotal1);*/
    /*ALPHADATATYPE *alpha2 = (ALPHADATATYPE *)malloc(sizeof(ALPHADATATYPE)*N[0]*N[1]);
    cudaMemcpy(alpha2,GP.alphaTmpCU[0],N[0]*N[1]*sizeof(ALPHADATATYPE),cudaMemcpyDeviceToHost);
    FILE *fff = fopen("alpha","w+");
    for (i = 0 ; i < N[0]*N[1] ; i++)
	fprintf(fff,"%lf\n",alpha2[i]);
    fclose(fff);
    free(alpha2);*/
	//cudaEventRecord(start1, NULL);
    size = N[0]*N[1];
    while (size > 1)
    {
	matrixAdd<<<dim3(ALPHAGRIDDIMX,((size - 1)/ALPHABLOCKDIM)/ALPHAGRIDDIMX+1),dim3(ALPHABLOCKDIMX,ALPHABLOCKDIMY)>>>(GP.alphaTmpCU,size);
	if (size%ALPHABLOCKDIM != 0)
		size = size/ALPHABLOCKDIM+1;
	else
		size = size/ALPHABLOCKDIM;
	/*ALPHADATATYPE *alpha3 = (ALPHADATATYPE *)malloc(sizeof(DATATYPE)*size);
    	cudaMemcpy(alpha3,GP.alphaTmpCU[tmpmark1],size*sizeof(ALPHADATATYPE),cudaMemcpyDeviceToHost);
    	FILE *fff2 = fopen("alpha2","w+");
    	for (i = 0 ; i < size ; i++)
		fprintf(fff2,"%lf\n",alpha2[i]);
    	fclose(fff2);
    	free(alpha3);*/
    }
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("matrixAdd : %f\n ",msecTotal1);*/
    //cudaMemcpy(&rtr,GP.alphaTmpCU[tmpmark1],sizeof(ALPHADATATYPE),cudaMemcpyDeviceToHost);
    //printf("rtr %lf\n",rtr);
    //Ar
  /*{
  CUFFTDATATYPE * f = (CUFFTDATATYPE*)malloc(N[0]*N[1]*sizeof(CUFFTDATATYPE));
    int ii;
    cudaMemcpy(f,GP.rhatCU,N[0]*N[1]*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("rhat %.20lf %.20lf\n",f[ii].x,f[ii].y);
	free(f);
	}*/
	//cudaEventRecord(start1, NULL);
    cunfft_trafo_2d2(plan,GP.rCU,GP.rhatCU);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("cunfft_trafo_2d2 : %f\n ",msecTotal1);*/
    /*{
  CUFFTDATATYPE * f = (CUFFTDATATYPE*)malloc(M*sizeof(CUFFTDATATYPE));
    int ii;
    cudaMemcpy(f,GP.rCU,M*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("r %.20lf %.20lf\n",f[ii].x,f[ii].y);
	free(f);
	}*/
    //(Ar)^Hw(Ar)
	//cudaEventRecord(start1, NULL);
    ALPHADATATYPE * alphaTmpCU;
    alphaTmpCU = GP.alphaTmpCU + 1;
    alpha1<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(alphaTmpCU,GP.rCU,GP.weightCU,M);
    size = M;
    while (size > 1)
    {
	matrixAdd<<<dim3(ALPHAGRIDDIMX,((size - 1)/ALPHABLOCKDIM)/ALPHAGRIDDIMX+1),dim3(ALPHABLOCKDIMX,ALPHABLOCKDIMY)>>>(alphaTmpCU,size);
	if (size%ALPHABLOCKDIM != 0)
		size = size/ALPHABLOCKDIM+1;
	else
		size = size/ALPHABLOCKDIM;
    }
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("matrixAdd : %f\n ",msecTotal1);*/
    //cudaMemcpy(&rtar,GP.alphaTmpCU[tmpmark1],sizeof(ALPHADATATYPE),cudaMemcpyDeviceToHost);
    //printf("rtar %lf\n",rtar);
    alpha2<<<1,1>>>(GP.alphaTmpCU,alphaTmpCU,GP.alCU);
    /*cudaMemcpy(&rtr,GP.alphaTmpCU[tmpmark1],sizeof(ALPHADATATYPE),cudaMemcpyDeviceToHost);
    cudaMemcpy(&rtar,alphaTmpCU[tmpmark1],sizeof(ALPHADATATYPE),cudaMemcpyDeviceToHost);
    cudaMemcpy(&al,GP.alCU,sizeof(ALPHADATATYPE),cudaMemcpyDeviceToHost);
    printf("%lf\n%lf\n%.20lf\n",rtr,rtar,al);*/
}


