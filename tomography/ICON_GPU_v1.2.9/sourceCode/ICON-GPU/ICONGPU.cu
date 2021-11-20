#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include <cuda_runtime_api.h>
#include "MainCode.cuh"
#include "MainCode_crossV.cuh"
#include "calFRC.cuh"
#include <pthread.h>

typedef struct{
	int deviceIndex;
	int Ystart;
	int Yend;
	int iternum;
	MrcHeader * projectionHead;
	FILE * projectionfileH;
	char *anglefile;
	char *resultpath;
	float *beta;
        int *before;
	DATATYPE *reprojection_full;
	DATATYPE *reprojection_crossV;
} threadFuncP;

void help()
{
	printf("ICON parameter\n######\n");
	printf("-input (-i) : the aligned tilt series.\n######\n");
	printf("-tiltfile (-t) : the aligned tilt file.\n######\n");
	printf("-outputPath (-o) : the path of a folder saving the result, two folder named “crossValidation” and “reconstruction” will be created inside.\n######\n");
	printf("-slice (-s) : the slices of reconstruction that include 2 parts split by ',' . For example, 0,511 means that reconstruct 512 slices ranging from slice 0 to slice 511.\n######\n");
	printf("-ICONIteration (-iter) : the iteration number including 3 parts split by ',' . For example, 5,50,10 means that, firstly, reconstruct using INFR for 5 iterations to generate a stable initial value, and then reconstruct using ICON for 50 iterations, and finally reconstruct using INFR for 10 iterations for fidelity.\n######\n");
	printf("-dataType (-d) :the type of dataset. There are two options: 1 for cryoET or plastic embedded ET (signal in black and background in white); 2 for negatively stained ET (signal in white and background in black); default as 1.\n######\n");
	printf("-threshold (-thr) : the threshold used in ICON, default as 0.03\n######\n");
	//printf("-thickness (-thi) : the thickness of reconstruction, in pixel.\n######\n");
	printf("-skipCrossValidation (-skipCV) : if this parameter is 1, the cross validation will not be performed and the frc files will not be generated, default as 0.\n######\n");
	printf("-gpu (-g) : the gpu list used for calculation. For example, 0,2,4,6 means using four gpus: 0, 2, 4 and 6 for calcualtion. Default as -1, meaning using all gpus in the system for calculation.\n######\n");
	printf("-help (-h) : for help.\n");
}

void *threadFunc(void *arg){

threadFuncP *tfp = (threadFuncP *)arg;


cudaSetDevice(tfp->deviceIndex);
cudaSetDeviceFlags(cudaDeviceMapHost);

Lib_2D2(tfp->Ystart,tfp->Yend+1,tfp->iternum,tfp->projectionHead,tfp->reprojection_full,tfp->projectionfileH,tfp->anglefile,tfp->resultpath,tfp->before,tfp->beta);

if (tfp->before[4] == 0){
	Lib_2D_crossV2(tfp->Ystart,tfp->Yend+1,tfp->iternum,tfp->projectionHead,tfp->reprojection_crossV,tfp->projectionfileH,tfp->anglefile,tfp->resultpath,tfp->before,tfp->beta);
}

return ((void *)0);
}

void frcSmooth(float *frc,int size,int overlapfactor)
{
	int i,j;
	int jbe,jen;
	float sum;
	int num;
	float *frctmp = (float *)malloc(size*sizeof(float));
	for (i = 0 ; i < size ; i++)
	{
		sum = 0;
		num = 0;
		jbe = i-overlapfactor/2;
		if (jbe < 0)
			jbe = 0;
		jen = jbe + overlapfactor;
		if (jen > size - 1)
		{
			jbe -= jen - size + 1;
			jen = size - 1;
		}
		if (jbe < 0)
			jbe = 0;
		for (j = jbe ; j < jen ; j++)
		{
			sum += frc[j];
			num++;
		}
		frctmp[i] = sum/num;
	}
	for (i = 0 ; i < size ; i++)
		frc[i] = frctmp[i];
	free(frctmp);
}

int main(int argc,char *argv[])
{

	//time_t now;
	//struct tm *timenow;

	//time(&now); 
	//timenow = localtime(&now); 
	//printf("ICONGPU begins at %s",asctime(timenow)); 
    	char cmd[1000];

	int Ystart,Yend,threadnum,iternum;
	char projectionfile[1000],anglefile[1000],resultpath[1000];
	int beforeupdate,beforemid,beforeend,dataType;
	int skipCrossValidation;
	float threshold;
	
	int i,j,k,thickness;
	int paraNum = 10;
	int *paraMark = (int *)malloc(paraNum*sizeof(int));

	int deviceList[MAXGPUNUM],deviceCount,deviceI;
	char deviceListC[1000];
	
	char loginfo[1000];
    	time_t rawtime;
    	struct tm * timeinfo;
    	//log Write
    	time(&rawtime);
    	timeinfo = localtime(&rawtime);
    	sprintf(loginfo,"##############################\n");
	logwrite(loginfo);
    	sprintf(loginfo,"time:\n   ");
    	logwrite(loginfo);
    	sprintf(loginfo,"%s",asctime(timeinfo));
    	logwrite(loginfo);
    	sprintf(loginfo,"cmd:\n   ");
    	logwrite(loginfo);
    	for (i = 0 ; i < argc ; i++)
    	{
		sprintf(loginfo,"%s ",argv[i]);
		logwrite(loginfo);
    	}
    	sprintf(loginfo,"\n");
    	logwrite(loginfo);
    	//end of log Write
        
	// read Parameter
	i = 1;
	memset(paraMark,0,paraNum*sizeof(int));
	paraMark[5] = 1;
	dataType = 1;
	paraMark[6] = 1;
	threshold = 520;
	paraMark[7] = 1;
	paraMark[8] = 1;
	paraMark[9] = 1;
	thickness = 0;
	skipCrossValidation = 0;
	memset(deviceList,-1,MAXGPUNUM*sizeof(int));
	cudaGetDeviceCount(&deviceCount);
	printf("##################\n%d GPU(s) are detected in the system!\n",deviceCount);
	cudaDeviceProp prop[deviceCount];
	for (deviceI = 0 ; deviceI < deviceCount ; deviceI++)
	{
		cudaSetDevice(deviceI);
		cudaGetDeviceProperties(&prop[deviceI],deviceI);
	}
	for (deviceI = 0 ; deviceI < deviceCount ; deviceI++)
		deviceList[deviceI] = deviceI;
	while (i < argc)	
	{
		if (argv[i][0] == '-')
		{
			if (strcmp(argv[i]+1,"input") == 0 || strcmp(argv[i]+1,"i") == 0)
			{	
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%s",projectionfile);
				i++;
				paraMark[0] = 1;
			}
			else
			if (strcmp(argv[i]+1,"tiltfile") == 0 || strcmp(argv[i]+1,"t") == 0)
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%s",anglefile);
				i++;
				paraMark[1] = 1;
			}
			else
			if (strcmp(argv[i]+1,"outputPath") == 0 || strcmp(argv[i]+1,"o") == 0)
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%s",resultpath);
				i++;
				paraMark[2] = 1;
			}
			else
			if (strcmp(argv[i]+1,"slice") == 0 || strcmp(argv[i]+1,"s") == 0)
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%d,%d",&Ystart,&Yend);
				i++;
				paraMark[3] = 1;
			}
			else
			if (strcmp(argv[i]+1,"ICONIteration") == 0 || strcmp(argv[i]+1,"iter") == 0)
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%d,%d,%d",&beforeupdate,&beforemid,&beforeend);
				iternum = beforeupdate+beforemid+beforeend;
				i++;
				paraMark[4] = 1;
			}
			else
			if (strcmp(argv[i]+1,"dataType") == 0 || strcmp(argv[i]+1,"d") == 0)
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%d",&dataType);
				i++;
				paraMark[5] = 1;
			}
			else
			if (strcmp(argv[i]+1,"threshold") == 0 || strcmp(argv[i]+1,"thr") == 0)
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%f",&threshold);
				i++;
				paraMark[6] = 1;
			}
			else
			if (strcmp(argv[i]+1,"gpu") == 0 || strcmp(argv[i]+1,"g") == 0)
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%s",deviceListC);
				int deviceListCLen = strlen(deviceListC);
				int ii,deviceListTmp;
				deviceI = 0;
				char *tmp = deviceListC;
				for (ii = 0 ; ii < deviceListCLen ; ii++)
					if (deviceListC[ii] == ',')
					{
						sscanf(tmp,"%d",&(deviceListTmp));
						if (deviceListTmp == -1)
							break;
						else 
							deviceList[deviceI] = deviceListTmp;
						deviceI++;
						tmp = deviceListC+ii+1;
					}
				sscanf(tmp,"%d",&(deviceListTmp));
				if (deviceListTmp != -1)
				{
					deviceList[deviceI] = deviceListTmp;
					deviceI++;
					deviceCount = deviceI;
					/*for (ii = 0 ; ii < deviceCount ; ii++)
						printf("%d\n",deviceList[ii]);
					exit(0);*/
				}
				i++;
				paraMark[7] = 1;
			}
			else
			if (strcmp(argv[i]+1,"thickness") == 0 || strcmp(argv[i]+1,"thi") == 0 )
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%d",&thickness);
				i++;
				paraMark[8] = 1;
			}
			else
			if (strcmp(argv[i]+1,"skipCrossValidation") == 0 || strcmp(argv[i]+1,"skipCV") == 0 )
			{
				i++;
				//printf("%s\n",argv[i]);
				sscanf(argv[i],"%d",&skipCrossValidation);
				i++;
				paraMark[9] = 1;
			}
			else
			if (strcmp(argv[i]+1,"help") == 0 || strcmp(argv[i]+1,"h") == 0)
			{
				i++;
				help();
				//log Write
    	    			sprintf(loginfo,"running state:\n   ");
    	    			logwrite(loginfo);
    	    			sprintf(loginfo,"ICON-GPU help finish!\n");
    	    			logwrite(loginfo);
    	    			//end of log Write	
				return -1;
			}
			else
				i++;
		}
		else
			i++;
	}
	for (i = 0 ; i < paraNum ; i++)
		if (paraMark[i] == 0)
		{
			printf("parameter error!\n Please use -help to see the manual\n");
			//log Write
    	    		sprintf(loginfo,"running state:\n   ");
    	    		logwrite(loginfo);
    	    		sprintf(loginfo,"fail!\n");
    	    		logwrite(loginfo);
	    		sprintf(loginfo,"Error message:\n   ");
	    		logwrite(loginfo);
	    		sprintf(loginfo,"parameter error!\n");
	    		logwrite(loginfo);
    	    		//end of log Write	
			return -1;
		}
	if (fabs(threshold - 520) < 0.0001)
	{
		if (dataType == 1)
			threshold = 0.03;
		if (dataType == 2)
			threshold = -0.03;
	}
	int slicetotal = Yend - Ystart + 1;
	int gpuUsedNum = 0;
	printf("parameter:\n");
	printf("input : %s\n",projectionfile);
	printf("tiltfile : %s\n",anglefile);
	printf("outputPath : %s\n",resultpath);
	printf("slice : %d,%d\n",Ystart,Yend);
	printf("ICONIteration : %d,%d,%d\n",beforeupdate,iternum-beforeupdate-beforeend,beforeend);
	printf("dataType : %d\n",dataType);
	printf("threshold : %f\n",threshold);
	if (thickness != 0)
		printf("thickness : %d\n",thickness);
	printf("use GPU : ");
	for (i = 0 ; i < (deviceCount < slicetotal ? deviceCount : slicetotal) ; i++)
	{
		printf("%d ",deviceList[i]);
		gpuUsedNum++;
	}
	printf("\n");
	if (gpuUsedNum == 0)
	{
		printf("Notice : No GPU is specified to use!\n");
		//log Write
    	    	sprintf(loginfo,"running state:\n   ");
    	    	logwrite(loginfo);
    	    	sprintf(loginfo,"fail!\n");
    	    	logwrite(loginfo);
	    	sprintf(loginfo,"Error message:\n   ");
	    	logwrite(loginfo);
	    	sprintf(loginfo,"No GPU is specified to use!\n");
	    	logwrite(loginfo);
    	    	//end of log Write	
		exit(0);
	}
	if (skipCrossValidation == 1)
	{
		printf("skip cross validation!\n");
	}
	// end of read Parameter
        float beta[10];
        int before[10];
        char crossVresPath[1000],reconstructionPath[1000];
	{
		sprintf(crossVresPath,"%s/crossValidation",resultpath);
		//if (access(crossVresPath,0) == -1)
		{
			sprintf(cmd,"mkdir %s",crossVresPath);
			system(cmd);
		}
		sprintf(reconstructionPath,"%s/reconstruction",resultpath);
		//if (access(reconstructionPath,0) == -1)
		{
			sprintf(cmd,"mkdir %s",reconstructionPath);
			system(cmd);
		}
	}
	threadnum = 1;
        before[0] = beforeupdate;
        before[1] = beforeend;
        before[2] = threadnum;
	before[3] = dataType;
	before[4] = skipCrossValidation;
	beta[0] = threshold;
	beta[1] = thickness; 
	MrcHeader * projectionHead = (MrcHeader *)malloc(sizeof(MrcHeader));
	FILE * projectionfileH = fopen(projectionfile,"r");
	mrc_read_head(projectionfileH,projectionHead);

	//-----------------------------------------------------------------
	pthread_t tid[deviceCount];
	threadFuncP tfp[deviceCount];
	int start,end;
	int rate[deviceCount];
	int orderP[deviceCount];
	double multiProcSum = 0;
	for (i = 0 ; i < deviceCount ; i++)
		rate[i] = 0;
	for (deviceI = 0 ; deviceI < deviceCount ; deviceI++)
	{
		for (i = 0 ; i < deviceCount ; i++)
			if (rate[i] == 0)
			{
				orderP[deviceI] = i;
				for (j = 0 ; j < deviceCount ; j++)
				{
					if (rate[j] == 0 && prop[deviceList[orderP[deviceI]]].multiProcessorCount < prop[deviceList[j]].multiProcessorCount)
						orderP[deviceI] = j;
				}
			}
		rate[orderP[deviceI]] = 1;
	}
	for (deviceI = 0 ; deviceI < deviceCount ; deviceI++)
		multiProcSum += prop[deviceList[deviceI]].multiProcessorCount;
	for (deviceI = 0 ; deviceI < deviceCount ; deviceI++)
	{
		rate[deviceI] = floor((double)prop[deviceList[deviceI]].multiProcessorCount/multiProcSum*slicetotal);
	}	
	multiProcSum = 0;
	for (deviceI = 0 ; deviceI < deviceCount ; deviceI++)
		multiProcSum += rate[deviceI];
	int sliceleft = slicetotal - multiProcSum;
	j = 0;
	for (i = 0 ; i < sliceleft ; i++)
	{
		rate[orderP[j]]++;
		j = (j+1)%deviceCount;
	}	
	multiProcSum = 0;
	DATATYPE *reprojection_full = (DATATYPE *)malloc(slicetotal*projectionHead->nx*sizeof(DATATYPE));
	DATATYPE *reprojection_crossV = (DATATYPE *)malloc(slicetotal*projectionHead->nx*sizeof(DATATYPE));
	for (deviceI = 0 ; deviceI < (deviceCount < slicetotal ? deviceCount : slicetotal) ; deviceI++)
	{
		if (deviceI != deviceCount - 1)
		{	
			start = multiProcSum;
			if (rate[deviceI] != 0)
				end = start + rate[deviceI] - 1;
			else
				end = start; 
			multiProcSum = end+1;
		}
		else
		{
			start = multiProcSum;
			end = Yend - Ystart;
		}
		printf("thread %d using GPU %d , deals with slices : start %d end %d\n",deviceI,deviceList[deviceI],start + Ystart,end + Ystart);
		tfp[deviceI].deviceIndex = deviceList[deviceI];
		tfp[deviceI].Ystart = start+Ystart;
		tfp[deviceI].Yend = end+Ystart;
		tfp[deviceI].iternum = iternum;
		tfp[deviceI].projectionHead = projectionHead;
		tfp[deviceI].projectionfileH = fopen(projectionfile,"r");
		tfp[deviceI].anglefile = anglefile;
		tfp[deviceI].resultpath = reconstructionPath;
		tfp[deviceI].beta = beta;
        	tfp[deviceI].before = before;
		tfp[deviceI].reprojection_full = reprojection_full+start*projectionHead->nx;
		tfp[deviceI].reprojection_crossV = reprojection_crossV+start*projectionHead->nx;
		pthread_create(&tid[deviceI],NULL,threadFunc,(void *)(&tfp[deviceI]));
	}
	for (deviceI = 0 ; deviceI < (deviceCount < slicetotal ? deviceCount : slicetotal) ; deviceI++)
		pthread_join(tid[deviceI],NULL);
	
	if (skipCrossValidation == 0){
		DATATYPE * frcfull = (DATATYPE *)malloc(projectionHead->nx*sizeof(DATATYPE));
		DATATYPE * frccrossV = (DATATYPE *)malloc(projectionHead->nx*sizeof(DATATYPE));
		//printf("go1\n");
		//get GroundTruth
		float *GroundTruth1 = (float *)malloc(projectionHead->ny*projectionHead->nx*sizeof(float));
		DATATYPE *GroundTruth = (DATATYPE *)malloc(slicetotal*projectionHead->nx*sizeof(DATATYPE));
		//printf("*******%d %d\n",slicetotal*projectionHead->nx*sizeof(float),slicetotal*projectionHead->nx*sizeof(DATATYPE));
			//get omitindex
		int Ang_Num = 0;
		float *thital = (float*)malloc(180*sizeof(float));
		FILE *anglefileF = fopen(anglefile,"r");
	   	while (fscanf(anglefileF,"%f",&(thital[Ang_Num]))!=EOF)
	    	{
			thital[Ang_Num] *=-1;
			Ang_Num++;
	    	}
	    	fclose(anglefileF);
	    	float omitang = thital[0];
		int omitindex = 0;
	   	 for (i = 0 ; i < Ang_Num ; i++)
			if (fabs(omitang) > fabs(thital[i]))
			{
				omitang = thital[i];
				omitindex = i;
			}
		free(thital);
			//end of get omitindex
		mrc_read_slice(projectionfileH, projectionHead, omitindex, 'z', GroundTruth1);
		k = 0;
		for (i = Ystart; i <= Yend; i++)
			for (j = 0 ; j < projectionHead->nx ; j++,k++)
			{
				GroundTruth[k] = GroundTruth1[i*projectionHead->nx+j];
			}
		//end of get GroundTruth
		//save GroundTruth
		{
			//FILE * fff = fopen("txt","w+");
	    		char outfilename[1000];

	    		sprintf(outfilename,"%s/GroundTruth.mrc",crossVresPath);

	    		FILE *outfile = fopen(outfilename,"w+");
	   		MrcHeader *outhead = (MrcHeader *)malloc(sizeof(MrcHeader));
			saveMRC_real(projectionHead,outfile,outhead,GroundTruth,projectionHead->nx,slicetotal);
				free(outhead);
	    		fclose(outfile);
	    		mrc_update_head(outfilename);
		}
		//end of save GroundTruth
		//save reProjection_full
		{
			//FILE * fff = fopen("txt","w+");
	    		char outfilename[1000];

	    		sprintf(outfilename,"%s/fullRec_reProjection.mrc",crossVresPath);

	    		FILE *outfile = fopen(outfilename,"w+");
	   		MrcHeader *outhead = (MrcHeader *)malloc(sizeof(MrcHeader));
			saveMRC_real(projectionHead,outfile,outhead,reprojection_full,projectionHead->nx,slicetotal);
	    		free(outhead);
	    		fclose(outfile);
	    		mrc_update_head(outfilename);
		}
		//end of save reProjection_full
		//save reProjection_crossV
		{
			//FILE * fff = fopen("txt","w+");
	    		char outfilename[1000];

	    		
	    		sprintf(outfilename,"%s/crossV_reprojection.mrc",crossVresPath);

	    		FILE *outfile = fopen(outfilename,"w+");
	   		MrcHeader *outhead = (MrcHeader *)malloc(sizeof(MrcHeader));
			saveMRC_real(projectionHead,outfile,outhead,reprojection_crossV,projectionHead->nx,slicetotal);
	    		free(outhead);
	    		fclose(outfile);
	    		mrc_update_head(outfilename);
		}
		//end of save reProjection_crossV
		/*if (slicetotal != projectionHead->nx)
		{
			printf("\nNOTICE : Re-projection should be square to calculate frc!\n\n");
			//log Write
	    		sprintf(loginfo,"Warning Message:\n   ");
	    		logwrite(loginfo);
	    		sprintf(loginfo,"Re-projection should be square to calculate frc!\n");
	    		logwrite(loginfo);
	    		//end of log Write
		}
		else*/
		{
		        calFRC_nonsqure(GroundTruth,reprojection_full,frcfull,projectionHead->nx,slicetotal);
			calFRC_nonsqure(GroundTruth,reprojection_crossV,frccrossV,projectionHead->nx,slicetotal);
			//calFRC_nonsqure_padOrClip(GroundTruth,reprojection_full,frcfull,projectionHead->nx,slicetotal);
			//calFRC_nonsqure_padOrClip(GroundTruth,reprojection_crossV,frccrossV,projectionHead->nx,slicetotal);
			//calFRC(GroundTruth,reprojection_full,frcfull,projectionHead->nx);
			//calFRC(GroundTruth,reprojection_crossV,frccrossV,projectionHead->nx);
	

			//save frcfull & frccrossV
			double resstep = 1.0/projectionHead->nx;
			char frcPath[1000];
			sprintf(frcPath,"%s/fullRec.frc",crossVresPath);	
			FILE *frcfullfile = fopen(frcPath,"w+");
			frcSmooth(frcfull,projectionHead->nx/2,10);
			for (i = 0 ; i < projectionHead->nx/2 ; i++)
				fprintf(frcfullfile,"%f %f\n",resstep*i,frcfull[i]);
			fclose(frcfullfile);
			sprintf(frcPath,"%s/crossV.frc",crossVresPath);
			FILE *frccrossVfile = fopen(frcPath,"w+");
			frcSmooth(frccrossV,projectionHead->nx/2,10);
			for (i = 0 ; i < projectionHead->nx/2 ; i++)
			{
		        	fprintf(frccrossVfile,"%f %f\n",resstep*i,frccrossV[i]);
			}
			fclose(frccrossVfile);	
		}
		free(GroundTruth1);
		free(GroundTruth);
		free(frcfull);
		free(frccrossV);
	}
	free(reprojection_full);
	free(reprojection_crossV);
	free(projectionHead);
	fclose(projectionfileH);
	for (deviceI = 0 ; deviceI < (deviceCount < slicetotal ? deviceCount : slicetotal) ; deviceI++)
		fclose(tfp[deviceI].projectionfileH );
	//time(&now); 
	//timenow = localtime(&now); 
	//printf("ICONGPU ends at %s",asctime(timenow));
	//log Write
    	sprintf(loginfo,"running state:\n   ");
    	logwrite(loginfo);
    	sprintf(loginfo,"ICON-GPU finish!\n");
    	logwrite(loginfo);
    	//end of log Write
	return 0;
}



