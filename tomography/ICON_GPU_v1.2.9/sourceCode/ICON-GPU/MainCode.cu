#include "MainCode.cuh"

int Lib_2D2(int sliceBegin,int sliceEnd,int iternum,MrcHeader* inhead,DATATYPE *reprojection,FILE *infile,char *anglefilename,char* resultpath,int *before,float *betain)
{
    char loginfo[1000];
    int Ang_Num,i,N[2],n[2];
    FILE * anglefile;
    float *thital,omitang;
    DATATYPE * weight;
    DATATYPE *bpx2D;
    int Yvalue;
    thital = (float*)malloc(180*sizeof(float));
    GICONPara GP;
    int thickness = (int)betain[1];
    if (thickness == 0)
	thickness = inhead->nx;

    //read mrc head
    MrcHeader *inhead_reProj=(MrcHeader *)malloc(sizeof(MrcHeader));
    memcpy(inhead_reProj,inhead,sizeof(MrcHeader));
    inhead_reProj->nz = 1;
    if ((anglefile = fopen(anglefilename,"r")) == NULL)
    {
        printf("\nCan not open anglefile!\n");
	//log Write
    	sprintf(loginfo,"running state:\n   ");
    	logwrite(loginfo);
    	sprintf(loginfo,"fail!\n");
    	logwrite(loginfo);
	sprintf(loginfo,"Error message:\n   ");
	logwrite(loginfo);
	sprintf(loginfo,"Can not open anglefile '%s'!\n",anglefilename);
	logwrite(loginfo);
    	//end of log Write	
        exit(1);
    }

    Ang_Num = 0;
    while (fscanf(anglefile,"%f",&(thital[Ang_Num]))!=EOF)
    {
	// simplest way to generate reconstruction as IMOD(xyz)
        thital[Ang_Num] *=-1;
        Ang_Num++;
    }
    fclose(anglefile);
    omitang = thital[0];
    for (i = 0 ; i < Ang_Num ; i++)
	if (fabs(omitang) > fabs(thital[i]))
		omitang = thital[i];
    //end of read projection
  
    weight = (DATATYPE *)malloc(inhead->nx*inhead->nz*sizeof(DATATYPE));
    Weight(weight,inhead,thital);
    //initGICONPara(&GP,inhead->nx,inhead->nz,weight);
    initGICONPara2(&GP,inhead->nx,thickness,inhead->nz,weight);
   
    bpx2D = (DATATYPE*)malloc(2*inhead->nx*inhead->nz*sizeof(DATATYPE));
    px2D(bpx2D,inhead,thital);
    
    float uptop;
    DATATYPE thita[2];
    int nfft_m = NFFTM;
    int M = inhead->nx*inhead->nz;
    N[0] = inhead->nx;
    N[1] = thickness;
    uptop = pow(2,ceil(log10(N[0])/log10(2.0)));
    thita[0] = 2.0*uptop/(double)N[0];
    uptop = pow(2,ceil(log10(N[1])/log10(2.0)));
    thita[1] = 2.0*uptop/(double)N[1];
    n[0] = ceil(thita[0]*N[0]);
    n[1] = ceil(thita[1]*N[1]);
    DATATYPE *px1d;
    px1d = (DATATYPE*)malloc(inhead->nx*sizeof(DATATYPE));
    int k;
    for (k = 0 ; k < inhead->nx ; k++)
    {
        px1d[k] = ((DATATYPE)k-(DATATYPE)inhead->nx/2.0)/(DATATYPE)inhead->nx;
    }
    cunfftplan1d plan1d;
    cunfft_initPlan_1d(&plan1d,inhead->nx,thita[0],inhead->nx,nfft_m,px1d);
    /*DATATYPE *ck1d = (DATATYPE *)malloc(N[0]*sizeof(DATATYPE));
    cudaMemcpy(ck1d,plan1d.ckCU,N[0]*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
    for (i = 0 ; i < 20 ; i++)
	printf("%.20lf\n",ck1d[i]);
    free(ck1d);*/
    free(px1d);

    cunfftplan2d2 plan2d;
    DATATYPE *ck2dCU; 
    CUFFTDATATYPE *g2dCU;
    cufftHandle planTmp2d;
    cudaMalloc((void**)&(ck2dCU), sizeof(DATATYPE)*N[0]*N[1]);
    cudaMalloc((void**)&(g2dCU), sizeof(CUFFTDATATYPE)*n[0]*n[1]);  
    cufftPlan2d(&(planTmp2d),n[0],n[1],CUFFTTYPE);  
    cunfft_mallocPlan_2d2(&plan2d,N[0],N[1],thita,M,nfft_m,CUNFFTPLAN_NO_MALLOC_CK | CUNFFTPLAN_NO_MALLOC_G | CUNFFTPLAN_NO_MALLOC_PLAN | CUNFFTPLAN_NO_PRECOMPUTE);
    plan2d.ckCU = ck2dCU;
    plan2d.gCU = g2dCU;
    plan2d.planTmp = planTmp2d;
    cunfft_initPlan_2d2(&plan2d,N[0],N[1],thita,M,nfft_m,bpx2D);

    float thital2[1];
    thital2[0] = omitang;
    px2D(bpx2D,inhead_reProj,thital2);
    cunfftplan2d2 plan2d_reProj;
    cunfft_mallocPlan_2d2(&plan2d_reProj,N[0],N[1],thita,N[0],nfft_m,CUNFFTPLAN_NO_MALLOC_CK | CUNFFTPLAN_NO_MALLOC_G | CUNFFTPLAN_NO_MALLOC_PLAN | CUNFFTPLAN_NO_PRECOMPUTE);
    plan2d_reProj.ckCU = ck2dCU;
    plan2d_reProj.gCU = g2dCU;
    plan2d_reProj.planTmp = planTmp2d;
    cunfft_initPlan_2d2(&plan2d_reProj,N[0],N[1],thita,N[0],nfft_m,bpx2D);
    
    //CUFFTDATATYPE * f = (CUFFTDATATYPE*)malloc(inhead->nx*inhead->nx*sizeof(CUFFTDATATYPE));
    //int ii;
    for (Yvalue = sliceBegin ; Yvalue < sliceEnd && Yvalue < inhead->ny; Yvalue++)
    {
        readSlicePart(plan1d,GP.fhat1dCU,GP.fCU,infile,inhead,Yvalue);
	/*cudaMemcpy(f,GP.fCU,inhead->nx*inhead->nz*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("%f %f\n",f[ii].x,f[ii].y);
	free(f);
	return 0;*/	

    //wb
	complexMulDoubleCU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(GP.fCU,GP.weightCU,M);
	/*cudaMemcpy(f,GP.fCU,inhead->nx*inhead->nz*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("yyyyyyy%.20lf %.20lf\n",f[ii].x,f[ii].y);*/
    //Ahwb
	cunfft_adjoint_2d2(plan2d,GP.fCU,GP.fhatCU);
	//cudaMemcpy(f,GP.fhatCU,N[0]*N[1]*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	//for (ii = 0 ; ii < 20 ; ii++)
		//printf("kkkk %.20lf %.20lf\n",f[ii].x,f[ii].y);
	//free(f);
	//return 0;
	complexCopyCU<<<dim3(GRIDDIMX,((N[0]*N[1] - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(GP.AhwbCU,GP.fhatCU,N[0]*N[1]);
        Iteration3(plan2d,GP,inhead,iternum,Yvalue,resultpath,before,betain);
	//if (Yvalue < 255)
	{
	cudaMemcpy(GP.m,GP.mCU,N[0]*N[1]*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	/*for (i = 0 ; i < 20 ; i++)
		printf("########%.20lf\n",GP.m[i]);*/
	printf("full reconstruction\n");
	reProject_NFFT2(plan1d,plan2d_reProj,GP,reprojection+(Yvalue-sliceBegin)*inhead->nx);
	/*for (i = 0 ; i < 20 ; i++)
		printf("----------%.20lf %.20lf\n",reprojection[i].x,reprojection[i].y);*/
	{
		//FILE * fff = fopen("txt","w+");
    		char outfilename[1000];

    		sprintf(outfilename,"%s/mid%05d.mrc",resultpath,Yvalue);
	
    		FILE *outfile = fopen(outfilename,"w+");
   		MrcHeader *outhead = (MrcHeader *)malloc(sizeof(MrcHeader));
		saveMRC_real_inv(inhead,outfile,outhead,GP.m,inhead->nx,thickness);
    		free(outhead);
    		fclose(outfile);
    		mrc_update_head(outfilename);
		/*for (i = 0 ; i < inhead->nx*inhead->nx ; i++)
			fprintf(fff,"%f\n",GP.m[i].x);
		fclose(fff);*/
	}
	}
    }
    cunfft_destroyPlan_1d(&plan1d);
    cunfft_destroyPlan_2d2(&plan2d);
    cunfft_destroyPlan_2d2(&plan2d_reProj);

    cudaFree(ck2dCU);
    cudaFree(g2dCU);
    cufftDestroy(planTmp2d);
    destroyGICONPara(&GP);

    free(bpx2D);
    free(weight);
    free(inhead_reProj);
    free(thital);
    return 0;
}
