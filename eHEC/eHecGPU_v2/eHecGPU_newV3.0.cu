#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "mrcfile2.h"
#include <cuda_runtime_api.h>

//only for matrixAdd 
#define ALPHABLOCKDIM 256
#define ALPHABLOCKDIMX  16
#define ALPHABLOCKDIMY  16
#define ALPHAGRIDDIMX  4

#define BLOCKDIMX  16
#define BLOCKDIMY  16
#define BLOCKDIM BLOCKDIMX*BLOCKDIMY
#define GRIDDIMX  16

#define PARAM8 1;
#define PARAM7 0.0;
#define PARAM6 0.0;
#define PARAM5 0.0;
//#define PARAM7 0.5;
//#define PARAM6 0.2;
//#define PARAM5 0.1;

//control the format of input
#define DATATYPE float

texture<DATATYPE, 1, cudaReadModeElementType> texData;

__device__ void addShared256(DATATYPE* data, int tid,int lowBound) {
    int i;
    for (i = 128 ; i >= 1 ; i/=2)
    {
	if (tid < i && tid + i < lowBound){
		data[tid] += data[tid + i];
	}
	__syncthreads();
    }
}

__global__ void matrixAdd(DATATYPE *matrix,int size)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    __shared__ DATATYPE data[ALPHABLOCKDIM];
    if (pn < size)
    {
	int lowBound = ALPHABLOCKDIM;
	if (blockid == size/ALPHABLOCKDIM)
		lowBound = size%ALPHABLOCKDIM;
	data[threadid] = matrix[pn];
	__syncthreads();
	addShared256(data,threadid,lowBound);
	if (threadid == 0)
		matrix[blockid] = data[0];
    }
}

__global__ void minusCU(DATATYPE* data,DATATYPE* res,DATATYPE b,int size){
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    	int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    	int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		res[pn] = (data[pn]-b)*(data[pn]-b)/size;
	}
}

__global__ void eHecCU(DATATYPE* eHeced_data,int nx,int ny,DATATYPE *param,DATATYPE std,float bg_coef){

	__shared__ DATATYPE datatmp[(BLOCKDIMY+2)*(BLOCKDIMX+2)];
	int apnx,apny,x,y;
	DATATYPE s_ave;

	x = threadIdx.x;
	y = threadIdx.y;
	apnx = blockIdx.x*blockDim.x+x;
	apny = blockIdx.y*blockDim.y+y;
	//step 1 
	datatmp[(y+1)*(BLOCKDIMX+2)+x+1] = tex1Dfetch(texData,apny*nx+apnx);
	//step 2
	if (x == 0){
		if (apnx > 0)
			datatmp[(y+1)*(BLOCKDIMX+2)+0] = tex1Dfetch(texData,apny*nx+apnx-1);
		else
			datatmp[(y+1)*(BLOCKDIMX+2)+0] = 0;
	}
	//step 3
	if (x == BLOCKDIMX-1){
		if (apnx < nx-1)
			datatmp[(y+1)*(BLOCKDIMX+2)+BLOCKDIMX+1] = tex1Dfetch(texData,apny*nx+apnx+1);
		else
			datatmp[(y+1)*(BLOCKDIMX+2)+BLOCKDIMX+1] = 0;
	}
	//step 4
	if (y == 0){
		if (apny > 0)
			datatmp[(0)*(BLOCKDIMX+2)+x+1] = tex1Dfetch(texData,(apny-1)*nx+apnx);
		else
			datatmp[(0)*(BLOCKDIMX+2)+x+1] = 0;
	}
	//step 5
	if (y == BLOCKDIMY-1){
		if (apny < ny-1)
			datatmp[(BLOCKDIMY+1)*(BLOCKDIMX+2)+x+1] = tex1Dfetch(texData,(apny+1)*nx+apnx);
		else
			datatmp[(BLOCKDIMY+1)*(BLOCKDIMX+2)+BLOCKDIMX+1] = 0;
	}
	//step 6
	if (x == 0 && y == 0){
		if (apnx > 0 && apny > 0)
			datatmp[0] = tex1Dfetch(texData,(apny-1)*nx+apnx-1);
		else
			datatmp[0] = 0;
	}
	//step 7
	if (x== 0 && y == BLOCKDIMY-1){
		if (apnx > 0 && apny < ny-1)
			datatmp[(BLOCKDIMY+1)*(BLOCKDIMX+2)+0] = tex1Dfetch(texData,(apny+1)*nx+apnx-1);
		else
			datatmp[(BLOCKDIMY+1)*(BLOCKDIMX+2)+0] = 0;
	}
	//step 8
	if (x== BLOCKDIMX-1 && y == 0){
		if (apnx < nx-1 && apny > 0)
			datatmp[0*(BLOCKDIMX+2)+BLOCKDIMX+1] = tex1Dfetch(texData,(apny-1)*nx+apnx+1);
		else
			datatmp[0*(BLOCKDIMX+2)+BLOCKDIMX+1] = 0;
	}
	//step 9
	if (x== BLOCKDIMX-1 && y == BLOCKDIMY-1){
		if (apnx < nx-1 && apny < ny-1)
			datatmp[(BLOCKDIMY+1)*(BLOCKDIMX+2)+BLOCKDIMX+1] = tex1Dfetch(texData,(apny+1)*nx+apnx+1);
		else
			datatmp[(BLOCKDIMY+1)*(BLOCKDIMX+2)+BLOCKDIMX+1] = 0;
	}
	__syncthreads();

	x = x + 1;
	y = y + 1;

//	if ((datatmp[y*(BLOCKDIMX+2)+x] < 0.5*std) || (apny == 0) || (apny == ny-1) || (apnx == 0) || (apnx == nx-1))
//	if ((datatmp[y*(BLOCKDIMX+2)+x] < 8.0) || (apny == 0) || (apny == ny-1) || (apnx == 0) || (apnx == nx-1))
//	if ((datatmp[y*(BLOCKDIMX+2)+x] < 5.0*std) || (apny == 0) || (apny == ny-1) || (apnx == 0) || (apnx == nx-1))
	if ((apny == 0) || (apny == ny-1) || (apnx == 0) || (apnx == nx-1))
		eHeced_data[apny*nx+apnx] = 0;
	else
	{
		int sum = 0;
		//l
		if (datatmp[y*(BLOCKDIMX+2)+x] > datatmp[(y-1)*(BLOCKDIMX+2)+x])
			sum++;
		//r
		if (datatmp[y*(BLOCKDIMX+2)+x] > datatmp[(y+1)*(BLOCKDIMX+2)+x])
			sum++;
		//u
		if (datatmp[y*(BLOCKDIMX+2)+x] > datatmp[y*(BLOCKDIMX+2)+x-1])
			sum++;
		//d
		if (datatmp[y*(BLOCKDIMX+2)+x] > datatmp[y*(BLOCKDIMX+2)+x+1])
			sum++;
		//ul
		if (datatmp[y*(BLOCKDIMX+2)+x] > datatmp[(y-1)*(BLOCKDIMX+2)+x-1])
			sum++;
		//ur
		if (datatmp[y*(BLOCKDIMX+2)+x] > datatmp[(y+1)*(BLOCKDIMX+2)+x-1])
			sum++;
		//dl
		if (datatmp[y*(BLOCKDIMX+2)+x] > datatmp[(y-1)*(BLOCKDIMX+2)+x+1])
			sum++;
		//dr
		if (datatmp[y*(BLOCKDIMX+2)+x] > datatmp[(y+1)*(BLOCKDIMX+2)+x+1])
			sum++;
		
		s_ave = (datatmp[(y-1)*(BLOCKDIMX+2)+x]+datatmp[(y+1)*(BLOCKDIMX+2)+x]+ datatmp[y*(BLOCKDIMX+2)+x-1]+datatmp[y*(BLOCKDIMX+2)+x+1]+datatmp[(y-1)*(BLOCKDIMX+2)+x-1]+datatmp[(y+1)*(BLOCKDIMX+2)+x-1]+datatmp[(y-1)*(BLOCKDIMX+2)+x+1]+datatmp[(y+1)*(BLOCKDIMX+2)+x+1])/8.0;

		if (datatmp[y*(BLOCKDIMX+2)+x] < (s_ave + bg_coef*std))
			eHeced_data[apny*nx+apnx] = 0;
		else
			eHeced_data[apny*nx+apnx] = param[sum];
	}
		
}

__global__ void AddCU(DATATYPE* a,DATATYPE* b,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    	int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    	int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		a[pn] += b[pn];
	}
}

void help()
{
	printf("eHec parameter\n######\n");
	printf("-input (-i) : the input file.\n######\n");
	printf("-output (-o) : the counting result.\n######\n");
	printf("-frameNum (-f) : the number of frames to sum up.\n######\n");
	printf("-help (-h) : for help.\n");
}


int main(int argc,char *argv[])
{
	int i = 0;
	char input[10000],output[10000];
	int frameNum;
	float bg_coef = 5.0;

    	int mark = 0;
	while (i < argc)	
	{
		if (argv[i][0] == '-')
		{
			if (strcmp(argv[i]+1,"input") == 0 || strcmp(argv[i]+1,"i") == 0)
			{	
				i++;
				sscanf(argv[i],"%s",input);
				mark++;
				i++;
			}
			else
			if (strcmp(argv[i]+1,"output") == 0 || strcmp(argv[i]+1,"o") == 0)
			{
				i++;
				sscanf(argv[i],"%s",output);
				mark++;
				i++;
			}
			else
			if (strcmp(argv[i]+1,"frameNum") == 0 || strcmp(argv[i]+1,"f") == 0)
			{
				i++;
				sscanf(argv[i], "%d", &frameNum);
				mark++;
				i++;
			}
			else
			if (strcmp(argv[i]+1,"help") == 0 || strcmp(argv[i]+1,"h") == 0)
			{
				i++;
				help();
				return -1;
			}
			else
			if (strcmp(argv[i]+1,"bg_coef") == 0 || strcmp(argv[i]+1,"bg") == 0) 
			{
				i++;
				sscanf(argv[i], "%f", &bg_coef);
				i++;
			}
			else
				i++;
		}
		else
			i++;
	}
    
	if (mark != 3){
		printf("Parameter errors! The input frames or output file or framenumber needs provided. \n");
		return 0;
	}
	FILE * fin = fopen(input,"r");
	if (fin == NULL){
		printf("Can not open input %s.\n",input);
		exit(-1);
	}
	MrcHeader *inhead = (MrcHeader *)malloc(sizeof(MrcHeader));
	mrc_read_head(fin,inhead);
	int nx = inhead->nx;
	int ny = inhead->ny;
	int nz = inhead->nz;
	DATATYPE * data2d  = (DATATYPE *)malloc(nx*ny*sizeof(DATATYPE));
	DATATYPE * eHeced_data2d  = (DATATYPE *)malloc(nx*ny*sizeof(DATATYPE));
	MrcHeader *outhead = (MrcHeader *)malloc(sizeof(MrcHeader));
	FILE * fout = fopen(output,"w+");
	memcpy(outhead,inhead,sizeof(MrcHeader));
	int wholeNum = int(nz/frameNum);
	if ((nz - wholeNum*frameNum) != 0)
		wholeNum++;
	outhead->nz = wholeNum;
	DATATYPE *result2d = (DATATYPE *)malloc(nx*ny*sizeof(DATATYPE));  
	mrc_write_head(fout,outhead);
	DATATYPE *data2dCU,*eHeced_data2dCU,*result2dCU,*paramCU; 
	cudaMalloc((void**)&(paramCU), sizeof(DATATYPE)*9);
	cudaMalloc((void**)&(data2dCU), sizeof(DATATYPE)*nx*ny);
	cudaMalloc((void**)&(result2dCU), sizeof(DATATYPE)*nx*ny);
	cudaMalloc((void**)&(eHeced_data2dCU), sizeof(DATATYPE)*nx*ny);
    
	cudaBindTexture(0,texData,data2dCU);
	DATATYPE *param = (DATATYPE *)malloc(9*sizeof(DATATYPE));
	for (i = 0 ; i <= 4 ; i++)
		param[i] = 0;
	param[5] = PARAM5;
	param[6] = PARAM6;
	param[7] = PARAM7;
	param[8] = PARAM8;

	//time start
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	float msecTotal1;
	//time end

	cudaMemcpy(paramCU, param, sizeof(DATATYPE)*9, cudaMemcpyHostToDevice);
	int wi;
	for (wi = 0 ; wi < wholeNum ; wi++){
		memset(result2d,0,nx*ny*sizeof(DATATYPE));
		cudaMemset(result2dCU,0,sizeof(DATATYPE)*nx*ny);
		float sum = 0;
		float std = 0;
		int size;
     	for (i = frameNum*wi ; i < nz && i < frameNum*wi+frameNum; i++){
     		printf("eHec %d frames\n",i);
			mrc_read_slice(fin, inhead, i, 'z', data2d);
			//time start
			cudaEventRecord(start1, NULL);
			//time end
			cudaMemcpy(data2dCU, data2d, sizeof(DATATYPE)*nx*ny, cudaMemcpyHostToDevice);
			//time start
			cudaDeviceSynchronize();
			cudaEventRecord(stop1, NULL);
			cudaEventSynchronize(stop1);
    			msecTotal1 = 0.0f;
    			cudaEventElapsedTime(&msecTotal1, start1, stop1);
    			//printf("data transfer CPU->GPU : %f ms\n",msecTotal1);
			//time end

			//time start
			cudaEventRecord(start1, NULL);
			//time end
			cudaMemcpy(eHeced_data2dCU, data2dCU, sizeof(DATATYPE)*nx*ny, cudaMemcpyDeviceToDevice);
			//calculate std
			{
				size = nx*ny;
    			while (size > 1)
    			{
					matrixAdd<<<dim3(ALPHAGRIDDIMX,((size - 1)/ALPHABLOCKDIM)/ALPHAGRIDDIMX+1),dim3(ALPHABLOCKDIMX,ALPHABLOCKDIMY)>>>(eHeced_data2dCU,size);
					if (size%ALPHABLOCKDIM != 0)
						size = size/ALPHABLOCKDIM+1;
					else
						size = size/ALPHABLOCKDIM;
    			}
				cudaMemcpy(&sum, eHeced_data2dCU, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
				sum /= nx*ny;
				//cudaMemcpy(data2dCU, data2d, sizeof(DATATYPE)*nx*ny, cudaMemcpyHostToDevice);
				minusCU<<<dim3(nx/BLOCKDIMX,ny/BLOCKDIMY),dim3(BLOCKDIMX,BLOCKDIMY)>>>(data2dCU,eHeced_data2dCU,sum,nx*ny);
				
				size = nx*ny;
    			while (size > 1)
    			{
					matrixAdd<<<dim3(ALPHAGRIDDIMX,((size - 1)/ALPHABLOCKDIM)/ALPHAGRIDDIMX+1),dim3(ALPHABLOCKDIMX,ALPHABLOCKDIMY)>>>(eHeced_data2dCU,size);
					if (size%ALPHABLOCKDIM != 0)
						size = size/ALPHABLOCKDIM+1;
					else
						size = size/ALPHABLOCKDIM;
    			}
				cudaMemcpy(&std, eHeced_data2dCU, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
				std = sqrt(std);
				printf("Std = %f\n",std);
			}
			eHecCU<<<dim3(nx/BLOCKDIMX,ny/BLOCKDIMY),dim3(BLOCKDIMX,BLOCKDIMY)>>>(eHeced_data2dCU,nx,ny,paramCU,std,bg_coef);
			AddCU<<<dim3(nx/BLOCKDIMX,ny/BLOCKDIMY),dim3(BLOCKDIMX,BLOCKDIMY)>>>(result2dCU,eHeced_data2dCU,nx*ny);
			cudaDeviceSynchronize();
		//time start
    		cudaEventRecord(stop1, NULL);
    		cudaEventSynchronize(stop1);
    		msecTotal1 = 0.0f;
    		cudaEventElapsedTime(&msecTotal1, start1, stop1);
    		//printf("eHec : %f ms\n",msecTotal1);
		//time end
    	}
		//time start
		cudaEventRecord(start1, NULL);
		//time end
		cudaMemcpy(result2d, result2dCU, sizeof(DATATYPE)*nx*ny, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		//time start
    		cudaEventRecord(stop1, NULL);
    		cudaEventSynchronize(stop1);
    		msecTotal1 = 0.0f;
    		cudaEventElapsedTime(&msecTotal1, start1, stop1);
    		//printf("data transform GPU->CPU : %f ms\n",msecTotal1);
		//time end		

		mrc_add_slice(fout,outhead,result2d);
		printf("save eHeced frame %d\n",wi);
    }
    fclose(fout);
    mrc_update_head(output);
    free(data2d);
    free(param);
    cudaFree(data2dCU);
    cudaFree(result2dCU);
    cudaFree(paramCU);
    cudaFree(eHeced_data2dCU);
    free(eHeced_data2d);
    free(result2d);
    free(inhead);
    free(outhead);
    fclose(fin);
    return 0;
}

