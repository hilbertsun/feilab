#include "nufft_gpu_v8.cuh"


__global__ void nufft_ck2d(DATATYPE *res ,int NX,int NY,int m, DATATYPE thita)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < NX*NY)
    {
        int x = pn%NX-NX/2;
        int y = pn/NX-NY/2;
        DATATYPE nx = ceil(NX*thita);
        DATATYPE ny = ceil(NY*thita);
        DATATYPE b = 2*thita*m/((2*thita-1)*PI);
        DATATYPE resx = exp(-b*((PI*x/nx)*(PI*x/nx)));
        DATATYPE resy = exp(-b*((PI*y/ny)*(PI*y/ny)));
        res[pn] = resx*resy;
    }
}

__global__ void nufft_ck2d2(DATATYPE *res ,int NX,int NY,int m, DATATYPE thita0,DATATYPE thita1)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < NX*NY)
    {
        //int x = pn%NX-NX/2;
        //int y = pn/NX-NY/2;
	int x = pn/NY-NX/2;
        int y = pn%NY-NY/2;
        DATATYPE nx = ceil(NX*thita0);
        DATATYPE ny = ceil(NY*thita1);
        DATATYPE b[2];
        b[0] = 2*thita0*m/((2*thita0-1)*PI);
        b[1] = 2*thita1*m/((2*thita1-1)*PI);
        DATATYPE resx = exp(-b[0]*((PI*x/nx)*(PI*x/nx)));
        DATATYPE resy = exp(-b[1]*((PI*y/ny)*(PI*y/ny)));
        res[pn] = resx*resy;
    }
}

void genPhi2dMark_adjoint(int *phi2dMark,int *phix_index,int *phiy_index,int m,int M,int nx,int ny)
{
	int i,j,k;
	int tmp;
	int m2 = 2*m+2;
	for (k = 0 ; k < M ; k++)
		for (j = k*m2 ; j < k*m2+m2 ; j++)
			for (i = k*m2 ; i < k*m2+m2 ; i++)
			{
				tmp = phix_index[i]*ny+phiy_index[j];
				phi2dMark[tmp]++;
			}
}

void genPhi1dMark_adjoint(int *phi1dMark,int *phi_index,int m,int M,int N)
{
	int i,k;
	int tmp;
	int m2 = 2*m+2;
	for (k = 0 ; k < M ; k++)
		for (i = k*m2 ; i < k*m2+m2 ; i++)
		{
			tmp = phi_index[i];
			phi1dMark[tmp]++;
		}
}

void preGenPhi2dSparseMatrix_adjoint(int *phi2dMark,int *phi2dSparseMatrixIndex,int *phi2dSparseMatrixNum,int *phi2dSparseMatrixPara,int nx,int ny)
{
	phi2dSparseMatrixPara[1] = phi2dSparseMatrixPara[0] = 0;
	int i;
	for (i = 0 ; i < nx*ny ; i++)
	{
		if (phi2dMark[i] > 0)
		{
			phi2dSparseMatrixIndex[phi2dSparseMatrixPara[1]] = i;
			phi2dSparseMatrixPara[0] += phi2dMark[i];
			phi2dSparseMatrixNum[phi2dSparseMatrixPara[1]++] = phi2dMark[i];
		}
	}
}

void preGenPhi1dSparseMatrix_adjoint(int *phi1dMark,int *phi1dSparseMatrixIndex,int *phi1dSparseMatrixNum,int *phi1dSparseMatrixPara,int N)
{
	phi1dSparseMatrixPara[1] = phi1dSparseMatrixPara[0] = 0;
	int i;
	for (i = 0 ; i < N ; i++)
	{
		if (phi1dMark[i] > 0)
		{
			phi1dSparseMatrixIndex[phi1dSparseMatrixPara[1]] = i;
			phi1dSparseMatrixPara[0] += phi1dMark[i];
			phi1dSparseMatrixNum[phi1dSparseMatrixPara[1]++] = phi1dMark[i];
		}
	}
}

void preGenPhi2dSparseMatrix2_adjoint(int *phi2dSparseMatrixNum,int *phi2dSparseMatrixLocate,int phi2dSparseMatrixPara1)
{
	int i;
	phi2dSparseMatrixLocate[0] = 0;
	for (i = 1 ; i < phi2dSparseMatrixPara1 ; i++)
	{
		phi2dSparseMatrixLocate[i] = phi2dSparseMatrixLocate[i-1]+phi2dSparseMatrixNum[i-1];
	}
}

void preGenPhi1dSparseMatrix2_adjoint(int *phi1dSparseMatrixNum,int *phi1dSparseMatrixLocate,int phi1dSparseMatrixPara1)
{
	int i;
	phi1dSparseMatrixLocate[0] = 0;
	for (i = 1 ; i < phi1dSparseMatrixPara1 ; i++)
	{
		phi1dSparseMatrixLocate[i] = phi1dSparseMatrixLocate[i-1]+phi1dSparseMatrixNum[i-1];
	}
}

void GenPhi2dSparseMatrix_adjoint(int *phi2dSparseMatrixJ,DATATYPE *phi2dSparseMatrix,DATATYPE *phix,int *phix_index,DATATYPE *phiy,int *phiy_index,int *phi2dSparseMatrixNum,int *phi2dSparseMatrixIndex,int* phi2dSparseMatrixPara,int nx,int ny,int m,int M)
{
	DATATYPE **phi2dSparseMatrixTmp = (DATATYPE **)malloc(nx*ny*sizeof(DATATYPE *));
    	int **phi2dSparseMatrixJTmp = (int **)malloc(nx*ny*sizeof(int *));
	int *numTmp = (int *)malloc(nx*ny*sizeof(int));
	int i,j,k;
	int m2 = 2*m+2;
	int tmp;
	DATATYPE phitmp;
	int phi2dSparseMatrixPara1 = phi2dSparseMatrixPara[1];
	for (i = 0 ; i < phi2dSparseMatrixPara1 ; i++)
	{
		phi2dSparseMatrixTmp[phi2dSparseMatrixIndex[i]] = (DATATYPE *)malloc(sizeof(DATATYPE)*phi2dSparseMatrixNum[i]);
		phi2dSparseMatrixJTmp[phi2dSparseMatrixIndex[i]] = (int *)malloc(sizeof(int)*phi2dSparseMatrixNum[i]);
	}
	memset(numTmp,0,nx*ny*sizeof(int));
	for (k = 0 ; k < M ; k++)
		for (j = k*m2 ; j < k*m2+m2 ; j++)
			for (i = k*m2 ; i < k*m2+m2 ; i++)
			{
				phitmp = phix[i]*phiy[j];
				tmp = phix_index[i]*ny+phiy_index[j];
				phi2dSparseMatrixJTmp[tmp][numTmp[tmp]] = k;
				phi2dSparseMatrixTmp[tmp][numTmp[tmp]++] = phitmp;
			}
	tmp = 0;
	for (i = 0 ; i < phi2dSparseMatrixPara1 ; i++)
	{
		for (j = 0 ; j < phi2dSparseMatrixNum[i] ; j++,tmp++)
		{
			phi2dSparseMatrixJ[tmp] = phi2dSparseMatrixJTmp[phi2dSparseMatrixIndex[i]][j];
			phi2dSparseMatrix[tmp] = phi2dSparseMatrixTmp[phi2dSparseMatrixIndex[i]][j];
		}
	}
	for (i = 0 ; i < phi2dSparseMatrixPara1 ; i++)
	{
		free(phi2dSparseMatrixTmp[phi2dSparseMatrixIndex[i]]);
		free(phi2dSparseMatrixJTmp[phi2dSparseMatrixIndex[i]]);
	}
	free(numTmp);
	free(phi2dSparseMatrixTmp);
	free(phi2dSparseMatrixJTmp);
}

void GenPhi1dSparseMatrix_adjoint(int *phi1dSparseMatrixJ,DATATYPE *phi1dSparseMatrix,DATATYPE *phi,int *phi_index,int *phi1dSparseMatrixNum,int *phi1dSparseMatrixIndex,int *phi1dSparseMatrixPara,int N,int m,int M)
{
	DATATYPE **phi1dSparseMatrixTmp = (DATATYPE **)malloc(N*sizeof(DATATYPE *));
    	int **phi1dSparseMatrixJTmp = (int **)malloc(N*sizeof(int *));
	int *numTmp = (int *)malloc(N*sizeof(int));
	int i,j,k;
	int m2 = 2*m+2;
	int tmp;
	DATATYPE phitmp;
	int phi1dSparseMatrixPara1 = phi1dSparseMatrixPara[1];
	for (i = 0 ; i < phi1dSparseMatrixPara1 ; i++)
	{
		phi1dSparseMatrixTmp[phi1dSparseMatrixIndex[i]] = (DATATYPE *)malloc(sizeof(DATATYPE)*phi1dSparseMatrixNum[i]);
		phi1dSparseMatrixJTmp[phi1dSparseMatrixIndex[i]] = (int *)malloc(sizeof(int)*phi1dSparseMatrixNum[i]);
	}
	memset(numTmp,0,N*sizeof(int));
	for (k = 0 ; k < M ; k++)
		for (j = k*m2 ; j < k*m2+m2 ; j++)
			{
				phitmp = phi[j];
				tmp = phi_index[j];
				phi1dSparseMatrixJTmp[tmp][numTmp[tmp]] = k;
				phi1dSparseMatrixTmp[tmp][numTmp[tmp]++] = phitmp;
			}
	tmp = 0;
	for (i = 0 ; i < phi1dSparseMatrixPara1 ; i++)
		for (j = 0 ; j < phi1dSparseMatrixNum[i] ; j++,tmp++)
		{
			phi1dSparseMatrixJ[tmp] = phi1dSparseMatrixJTmp[phi1dSparseMatrixIndex[i]][j];
			phi1dSparseMatrix[tmp] = phi1dSparseMatrixTmp[phi1dSparseMatrixIndex[i]][j];
		}
	for (i = 0 ; i < phi1dSparseMatrixPara1 ; i++)
	{
		free(phi1dSparseMatrixTmp[phi1dSparseMatrixIndex[i]]);
		free(phi1dSparseMatrixJTmp[phi1dSparseMatrixIndex[i]]);
	}
	free(numTmp);
	free(phi1dSparseMatrixTmp);
	free(phi1dSparseMatrixJTmp);
}

void GenPhi2dSparseMatrix_trafo(int *phi2dSparseMatrixL,DATATYPE *phi2dSparseMatrix,DATATYPE *phix,int *phix_index,DATATYPE *phiy,int *phiy_index,int nx,int ny,int m,int M)
{
	int k,j,i;
	int m2 = 2*m+2;
	int tmp = 0;
	for (k = 0 ; k < M ; k++)
		for (j = k*m2 ; j < k*m2+m2 ; j++)
			for (i = k*m2 ; i < k*m2+m2 ; i++,tmp++)
			{
				phi2dSparseMatrixL[tmp] = phix_index[i]*ny+phiy_index[j];
				phi2dSparseMatrix[tmp] = phix[i]*phiy[j];
			}
}


__global__ void nufft_Phi2d_1dCU(DATATYPE *phix ,int *phix_index,DATATYPE *pdx2d,int N,int m,int M,DATATYPE thita,DATATYPE b,DATATYPE bf,int xy)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < M*(2*m+2))
	{
		int j = pn/(2*m+2);
		int n = ceil(N*thita);
		int l = floor(n*pdx2d[2*j+xy]) - m + pn%(2*m+2);
		DATATYPE x1 = pdx2d[2*j+xy] - (DATATYPE)l/(DATATYPE)n;
		phix[pn] = bf*exp(-((n*x1)*(n*x1)/b));
		phix_index[pn] = (l + n) % n;
	}
}

void cunfft_mallocPlan_2d(cunfftplan2d *plan,int NX,int NY,DATATYPE thita,int M,int m,int flags)
{
	plan->m = m;
	plan->N[0] = NX;
	plan->N[1] = NY;
	plan->thita = thita;
	plan->M = M;

	int nx = ceil(NX*thita);
	int ny = ceil(NY*thita);
	
	plan->flags = flags;
	//deal with ckCU
	if ((flags & CUNFFTPLAN_NO_MALLOC_CK) != 0)
	{
		plan->ckCU = NULL;
	}
	else
	{
		cudaMalloc((void**)&(plan->ckCU), sizeof(DATATYPE)*NX*NY);
		printf("malloc ck\n");
	}
	
	//deal with gCU
	if ((flags & CUNFFTPLAN_NO_MALLOC_G) != 0)
	{
		plan->ghatCU = plan->gCU = NULL;
	}
	else
	{
		cudaMalloc((void**)&(plan->gCU),nx*ny*sizeof(CUFFTDATATYPE));
		plan->ghatCU = plan->gCU;
		printf("malloc g\n");
	}

	//deal with planTmp
	if ((flags & CUNFFTPLAN_NO_MALLOC_PLAN) == 0)
	{
		cufftPlan2d(&(plan->planTmp),nx,ny,CUFFTTYPE);
		printf("malloc plan\n");
	}
	
}

void cunfft_mallocPlan_2d2(cunfftplan2d2 *plan,int NX,int NY,DATATYPE* thita,int M,int m,int flags)
{
	plan->m = m;
	plan->N[0] = NX;
	plan->N[1] = NY;
	plan->thita[0] = thita[0];
	plan->thita[1] = thita[1];
	plan->M = M;

	int nx = ceil(NX*thita[0]);
	int ny = ceil(NY*thita[1]);
	
	plan->flags = flags;
	//deal with ckCU
	if ((flags & CUNFFTPLAN_NO_MALLOC_CK) != 0)
	{
		plan->ckCU = NULL;
	}
	else
	{
		cudaMalloc((void**)&(plan->ckCU), sizeof(DATATYPE)*NX*NY);
		printf("malloc ck\n");
	}
	
	//deal with gCU
	if ((flags & CUNFFTPLAN_NO_MALLOC_G) != 0)
	{
		plan->ghatCU = plan->gCU = NULL;
	}
	else
	{
		cudaMalloc((void**)&(plan->gCU),nx*ny*sizeof(CUFFTDATATYPE));
		plan->ghatCU = plan->gCU;
		printf("malloc g\n");
	}

	//deal with planTmp
	if ((flags & CUNFFTPLAN_NO_MALLOC_PLAN) == 0)
	{
		cufftPlan2d(&(plan->planTmp),nx,ny,CUFFTTYPE);
		printf("malloc plan\n");
	}
	
}


void cunfft_initPlan_2d(cunfftplan2d *plan,int NX,int NY,DATATYPE thita,int M,int m,DATATYPE *pdx2d)
{

	/*plan->m = m;
	plan->N[0] = NX;
	plan->N[1] = NY;
	plan->thita = thita;
	plan->M = M;*/

	int nx = ceil(NX*thita);
	int ny = ceil(NY*thita);

	/*cufftPlan2d(&(plan->planTmp),nx,ny,CUFFTTYPE);

	cudaMalloc((void**)&(plan->gCU),nx*ny*sizeof(CUFFTDATATYPE));
	//cudaMalloc((void**)&(plan->ghatCU),nx*ny*sizeof(CUFFTDATATYPE));

	//calculate ck
	cudaMalloc((void**)&(plan->ckCU), sizeof(DATATYPE)*NX*NY);*/
	nufft_ck2d<<<dim3(GRIDDIMX,((NX*NY - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan->ckCU,NX,NY,m,thita);

	cudaMalloc((void**)&(plan->f1CU),M*(2*m+2)*sizeof(CUFFTDATATYPE));
	plan->ghatCU = plan->gCU;
	//--------------calculate ck

	//calculate phi2d
	DATATYPE *pdx2dCU;
	cudaMalloc((void**)&(plan->phixCU),sizeof(DATATYPE)*M*(2*m+2));
    	cudaMalloc((void**)&(plan->phiyCU),sizeof(DATATYPE)*M*(2*m+2));
    	cudaMalloc((void**)&(plan->phix_indexCU),sizeof(int)*M*(2*m+2));
    	cudaMalloc((void**)&(plan->phiy_indexCU),sizeof(int)*M*(2*m+2));
	DATATYPE * phix = (DATATYPE *)malloc(sizeof(DATATYPE)*M*(2*m+2));
    	DATATYPE * phiy = (DATATYPE *)malloc(sizeof(DATATYPE)*M*(2*m+2));
    	int *  phix_index = (int *)malloc(sizeof(int)*M*(2*m+2));
    	int * phiy_index = (int *)malloc(sizeof(int)*M*(2*m+2));
		//calculate phi1d
	cudaMalloc((void**)&pdx2dCU,sizeof(DATATYPE)*2*M);
	cudaMemcpy(pdx2dCU, pdx2d, sizeof(DATATYPE)*2*M, cudaMemcpyHostToDevice);
	DATATYPE b = 2*thita*m/((2*thita-1)*PI);
    	DATATYPE bf = 1.0/sqrt(PI*b);
	nufft_Phi2d_1dCU<<<dim3(GRIDDIMX,((M*(2*m+2) - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan->phixCU,plan->phix_indexCU,pdx2dCU,NX,m,M,thita,b,bf,0);
    	nufft_Phi2d_1dCU<<<dim3(GRIDDIMX,((M*(2*m+2) - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan->phiyCU,plan->phiy_indexCU,pdx2dCU,NY,m,M,thita,b,bf,1);
	cudaMemcpy(phix, plan->phixCU, sizeof(DATATYPE)*M*(2*m+2), cudaMemcpyDeviceToHost);
    	cudaMemcpy(phiy, plan->phiyCU, sizeof(DATATYPE)*M*(2*m+2), cudaMemcpyDeviceToHost);
    	cudaMemcpy(phix_index, plan->phix_indexCU, sizeof(int)*M*(2*m+2), cudaMemcpyDeviceToHost);
   	cudaMemcpy(phiy_index, plan->phiy_indexCU, sizeof(int)*M*(2*m+2), cudaMemcpyDeviceToHost);
		//--------------calculate phi1d
        if ((plan->flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		cudaFree(plan->phixCU);
    		cudaFree(plan->phiyCU);
    		cudaFree(plan->phix_indexCU);
    		cudaFree(plan->phiy_indexCU);
	}
	cudaFree(pdx2dCU);

		//calculate adjoint phi2d in CPU
	int *phi2dMark = (int *)malloc(nx*ny*sizeof(int));
    	memset(phi2dMark,0,sizeof(int)*nx*ny);
    	genPhi2dMark_adjoint(phi2dMark,phix_index,phiy_index,m,M,nx,ny);
    	int phi2dSparseMatrixPara[2];
    	int *phi2dSparseMatrixIndex = (int *)malloc(M*(2*m+2)*(2*m+2)*sizeof(int));
    	int *phi2dSparseMatrixNum = (int *)malloc(M*(2*m+2)*(2*m+2)*sizeof(int));
    	preGenPhi2dSparseMatrix_adjoint(phi2dMark,phi2dSparseMatrixIndex,phi2dSparseMatrixNum,phi2dSparseMatrixPara,nx,ny);
    	int *phi2dSparseMatrixLocate = (int *)malloc(phi2dSparseMatrixPara[1]*sizeof(int));
    	preGenPhi2dSparseMatrix2_adjoint(phi2dSparseMatrixNum,phi2dSparseMatrixLocate,phi2dSparseMatrixPara[1]);
    	int *phi2dSparseMatrixJ = (int *)malloc(phi2dSparseMatrixPara[0]*sizeof(int));
    	DATATYPE *phi2dSparseMatrix = (DATATYPE *)malloc(phi2dSparseMatrixPara[0]*sizeof(DATATYPE));
    	GenPhi2dSparseMatrix_adjoint(phi2dSparseMatrixJ,phi2dSparseMatrix,phix,phix_index,phiy,phiy_index,phi2dSparseMatrixNum,phi2dSparseMatrixIndex,phi2dSparseMatrixPara,nx,ny,m,M);
	cudaMalloc((void**)&(plan->adjoint_phiSparM_LocaCU),phi2dSparseMatrixPara[1]*sizeof(int));
    	cudaMalloc((void**)&(plan->adjoint_phiSparM_indexCU),phi2dSparseMatrixPara[1]*sizeof(int));
    	cudaMalloc((void**)&(plan->adjoint_phiSparMCU),phi2dSparseMatrixPara[0]*sizeof(DATATYPE));
    	cudaMalloc((void**)&(plan->adjoint_phiSparM_jCU),phi2dSparseMatrixPara[0]*sizeof(int));

    	cudaMemcpy(plan->adjoint_phiSparM_LocaCU, phi2dSparseMatrixLocate, phi2dSparseMatrixPara[1]*sizeof(int), cudaMemcpyHostToDevice);

    	cudaMemcpy(plan->adjoint_phiSparM_indexCU, phi2dSparseMatrixIndex, phi2dSparseMatrixPara[1]*sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(plan->adjoint_phiSparMCU,phi2dSparseMatrix, phi2dSparseMatrixPara[0]*sizeof(DATATYPE), cudaMemcpyHostToDevice);
    	cudaMemcpy(plan->adjoint_phiSparM_jCU, phi2dSparseMatrixJ, phi2dSparseMatrixPara[0]*sizeof(int), cudaMemcpyHostToDevice);
	plan->adjoint_phiSparM_para[0] = phi2dSparseMatrixPara[0];
	plan->adjoint_phiSparM_para[1] = phi2dSparseMatrixPara[1];

	//printf("--------------%d %d\n",plan->adjoint_phiSparM_para[0],plan->adjoint_phiSparM_para[1]);
	//printf("phi2dSparseMatrixPara  %d %d\n",phi2dSparseMatrixPara[0],phi2dSparseMatrixPara[1]);
	free(phi2dMark);
	free(phi2dSparseMatrixIndex);
	free(phi2dSparseMatrixNum);
	free(phi2dSparseMatrixLocate);
	free(phi2dSparseMatrixJ);
	free(phi2dSparseMatrix);
		//--------------calculate adjoint phi2d in CPU
		//calculate trofa phi2d in CPU
	plan->trafo_phiSparM_para[0] = M*(2*m+2)*(2*m+2);
	plan->trafo_phiSparM_para[1] = M;
	if ((plan->flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		printf("trafo precompute!\n");
		phi2dSparseMatrixJ = (int *)malloc(plan->trafo_phiSparM_para[0]*sizeof(int));
		phi2dSparseMatrix = (DATATYPE *)malloc(plan->trafo_phiSparM_para[0]*sizeof(DATATYPE));
		GenPhi2dSparseMatrix_trafo(phi2dSparseMatrixJ,phi2dSparseMatrix,phix,phix_index,phiy,phiy_index,nx,ny,m,M);

		cudaMalloc((void**)&(plan->trafo_phiSparMCU),plan->trafo_phiSparM_para[0]*sizeof(DATATYPE));
    		cudaMalloc((void**)&(plan->trafo_phiSparM_lCU),plan->trafo_phiSparM_para[0]*sizeof(int));

		cudaMemcpy(plan->trafo_phiSparMCU,phi2dSparseMatrix, plan->trafo_phiSparM_para[0]*sizeof(DATATYPE), cudaMemcpyHostToDevice);
    		cudaMemcpy(plan->trafo_phiSparM_lCU, phi2dSparseMatrixJ, plan->trafo_phiSparM_para[0]*sizeof(int), cudaMemcpyHostToDevice);

		free(phi2dSparseMatrixJ);
		free(phi2dSparseMatrix);
		//--------------calculate trofa phi2d in CPU
	}
	free(phix);
    	free(phiy);
    	free(phix_index);
    	free(phiy_index);
		//--------------calculate phi2d
}

void cunfft_initPlan_2d2(cunfftplan2d2 *plan,int NX,int NY,DATATYPE* thita,int M,int m,DATATYPE *pdx2d)
{


	int nx = ceil(NX*thita[0]);
	int ny = ceil(NY*thita[1]);

	nufft_ck2d2<<<dim3(GRIDDIMX,((NX*NY - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan->ckCU,NX,NY,m,thita[0],thita[1]);

	cudaMalloc((void**)&(plan->f1CU),M*(2*m+2)*sizeof(CUFFTDATATYPE));
	plan->ghatCU = plan->gCU;
	//--------------calculate ck

	//calculate phi2d
	DATATYPE *pdx2dCU;
	cudaMalloc((void**)&(plan->phixCU),sizeof(DATATYPE)*M*(2*m+2));
    	cudaMalloc((void**)&(plan->phiyCU),sizeof(DATATYPE)*M*(2*m+2));
    	cudaMalloc((void**)&(plan->phix_indexCU),sizeof(int)*M*(2*m+2));
    	cudaMalloc((void**)&(plan->phiy_indexCU),sizeof(int)*M*(2*m+2));
	DATATYPE * phix = (DATATYPE *)malloc(sizeof(DATATYPE)*M*(2*m+2));
    	DATATYPE * phiy = (DATATYPE *)malloc(sizeof(DATATYPE)*M*(2*m+2));
    	int *  phix_index = (int *)malloc(sizeof(int)*M*(2*m+2));
    	int * phiy_index = (int *)malloc(sizeof(int)*M*(2*m+2));
		//calculate phi1d
	cudaMalloc((void**)&pdx2dCU,sizeof(DATATYPE)*2*M);
	cudaMemcpy(pdx2dCU, pdx2d, sizeof(DATATYPE)*2*M, cudaMemcpyHostToDevice);
	DATATYPE b = 2*thita[0]*m/((2*thita[0]-1)*PI);
    	DATATYPE bf = 1.0/sqrt(PI*b);
	nufft_Phi2d_1dCU<<<dim3(GRIDDIMX,((M*(2*m+2) - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan->phixCU,plan->phix_indexCU,pdx2dCU,NX,m,M,thita[0],b,bf,0);
        b = 2*thita[1]*m/((2*thita[1]-1)*PI);
	bf = 1.0/sqrt(PI*b);
    	nufft_Phi2d_1dCU<<<dim3(GRIDDIMX,((M*(2*m+2) - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan->phiyCU,plan->phiy_indexCU,pdx2dCU,NY,m,M,thita[1],b,bf,1);
	cudaMemcpy(phix, plan->phixCU, sizeof(DATATYPE)*M*(2*m+2), cudaMemcpyDeviceToHost);
    	cudaMemcpy(phiy, plan->phiyCU, sizeof(DATATYPE)*M*(2*m+2), cudaMemcpyDeviceToHost);
    	cudaMemcpy(phix_index, plan->phix_indexCU, sizeof(int)*M*(2*m+2), cudaMemcpyDeviceToHost);
   	cudaMemcpy(phiy_index, plan->phiy_indexCU, sizeof(int)*M*(2*m+2), cudaMemcpyDeviceToHost);
		//--------------calculate phi1d
	//printf("ooooo1\n");
        if ((plan->flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		cudaFree(plan->phixCU);
    		cudaFree(plan->phiyCU);
    		cudaFree(plan->phix_indexCU);
    		cudaFree(plan->phiy_indexCU);
	}
	cudaFree(pdx2dCU);
	//printf("ooooo1.5\n");
		//calculate adjoint phi2d in CPU
	int *phi2dMark = (int *)malloc(nx*ny*sizeof(int));
    	memset(phi2dMark,0,sizeof(int)*nx*ny);
    	genPhi2dMark_adjoint(phi2dMark,phix_index,phiy_index,m,M,nx,ny);
	//printf("ooooo2\n");
    	int phi2dSparseMatrixPara[2];
    	int *phi2dSparseMatrixIndex = (int *)malloc(M*(2*m+2)*(2*m+2)*sizeof(int));
    	int *phi2dSparseMatrixNum = (int *)malloc(M*(2*m+2)*(2*m+2)*sizeof(int));
    	preGenPhi2dSparseMatrix_adjoint(phi2dMark,phi2dSparseMatrixIndex,phi2dSparseMatrixNum,phi2dSparseMatrixPara,nx,ny);
	//printf("ooooo3\n");
    	int *phi2dSparseMatrixLocate = (int *)malloc(phi2dSparseMatrixPara[1]*sizeof(int));
    	preGenPhi2dSparseMatrix2_adjoint(phi2dSparseMatrixNum,phi2dSparseMatrixLocate,phi2dSparseMatrixPara[1]);
	//printf("ooooo4\n");
    	int *phi2dSparseMatrixJ = (int *)malloc(phi2dSparseMatrixPara[0]*sizeof(int));
    	DATATYPE *phi2dSparseMatrix = (DATATYPE *)malloc(phi2dSparseMatrixPara[0]*sizeof(DATATYPE));
    	GenPhi2dSparseMatrix_adjoint(phi2dSparseMatrixJ,phi2dSparseMatrix,phix,phix_index,phiy,phiy_index,phi2dSparseMatrixNum,phi2dSparseMatrixIndex,phi2dSparseMatrixPara,nx,ny,m,M);
	//printf("ooooo5\n");
	cudaMalloc((void**)&(plan->adjoint_phiSparM_LocaCU),phi2dSparseMatrixPara[1]*sizeof(int));
    	cudaMalloc((void**)&(plan->adjoint_phiSparM_indexCU),phi2dSparseMatrixPara[1]*sizeof(int));
    	cudaMalloc((void**)&(plan->adjoint_phiSparMCU),phi2dSparseMatrixPara[0]*sizeof(DATATYPE));
    	cudaMalloc((void**)&(plan->adjoint_phiSparM_jCU),phi2dSparseMatrixPara[0]*sizeof(int));

    	cudaMemcpy(plan->adjoint_phiSparM_LocaCU, phi2dSparseMatrixLocate, phi2dSparseMatrixPara[1]*sizeof(int), cudaMemcpyHostToDevice);

    	cudaMemcpy(plan->adjoint_phiSparM_indexCU, phi2dSparseMatrixIndex, phi2dSparseMatrixPara[1]*sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(plan->adjoint_phiSparMCU,phi2dSparseMatrix, phi2dSparseMatrixPara[0]*sizeof(DATATYPE), cudaMemcpyHostToDevice);
    	cudaMemcpy(plan->adjoint_phiSparM_jCU, phi2dSparseMatrixJ, phi2dSparseMatrixPara[0]*sizeof(int), cudaMemcpyHostToDevice);
	plan->adjoint_phiSparM_para[0] = phi2dSparseMatrixPara[0];
	plan->adjoint_phiSparM_para[1] = phi2dSparseMatrixPara[1];

	//printf("--------------%d %d\n",plan->adjoint_phiSparM_para[0],plan->adjoint_phiSparM_para[1]);
	//printf("phi2dSparseMatrixPara  %d %d\n",phi2dSparseMatrixPara[0],phi2dSparseMatrixPara[1]);
	free(phi2dMark);
	free(phi2dSparseMatrixIndex);
	free(phi2dSparseMatrixNum);
	free(phi2dSparseMatrixLocate);
	free(phi2dSparseMatrixJ);
	free(phi2dSparseMatrix);
		//--------------calculate adjoint phi2d in CPU
		//calculate trofa phi2d in CPU
	plan->trafo_phiSparM_para[0] = M*(2*m+2)*(2*m+2);
	plan->trafo_phiSparM_para[1] = M;
	if ((plan->flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		printf("trafo precompute!\n");
		phi2dSparseMatrixJ = (int *)malloc(plan->trafo_phiSparM_para[0]*sizeof(int));
		phi2dSparseMatrix = (DATATYPE *)malloc(plan->trafo_phiSparM_para[0]*sizeof(DATATYPE));
		GenPhi2dSparseMatrix_trafo(phi2dSparseMatrixJ,phi2dSparseMatrix,phix,phix_index,phiy,phiy_index,nx,ny,m,M);

		cudaMalloc((void**)&(plan->trafo_phiSparMCU),plan->trafo_phiSparM_para[0]*sizeof(DATATYPE));
    		cudaMalloc((void**)&(plan->trafo_phiSparM_lCU),plan->trafo_phiSparM_para[0]*sizeof(int));

		cudaMemcpy(plan->trafo_phiSparMCU,phi2dSparseMatrix, plan->trafo_phiSparM_para[0]*sizeof(DATATYPE), cudaMemcpyHostToDevice);
    		cudaMemcpy(plan->trafo_phiSparM_lCU, phi2dSparseMatrixJ, plan->trafo_phiSparM_para[0]*sizeof(int), cudaMemcpyHostToDevice);

		free(phi2dSparseMatrixJ);
		free(phi2dSparseMatrix);
		//--------------calculate trofa phi2d in CPU
	}
	free(phix);
    	free(phiy);
    	free(phix_index);
    	free(phiy_index);
		//--------------calculate phi2d
}

void cunfft_destroyPlan_2d(cunfftplan2d *plan)
{
	//create in init
	if ((plan->flags & CUNFFTPLAN_NO_MALLOC_CK) == 0)
	{
        	cudaFree(plan->ckCU);
	}
	cudaFree(plan->adjoint_phiSparM_LocaCU);
	cudaFree(plan->adjoint_phiSparM_jCU);
	cudaFree(plan->adjoint_phiSparMCU);
	cudaFree(plan->adjoint_phiSparM_indexCU);

	if ((plan->flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		cudaFree(plan->trafo_phiSparM_lCU);
		cudaFree(plan->trafo_phiSparMCU);
	}
	else
	{
		cudaFree(plan->phixCU);
    		cudaFree(plan->phiyCU);
    		cudaFree(plan->phix_indexCU);
    		cudaFree(plan->phiy_indexCU);
	}
	if ((plan->flags & CUNFFTPLAN_NO_MALLOC_G) == 0)
	{
		cudaFree(plan->gCU);
	}
	//cudaFree(plan->ghatCU);
	cudaFree(plan->f1CU);
	if ((plan->flags & CUNFFTPLAN_NO_MALLOC_PLAN) == 0)
	{
		cufftDestroy(plan->planTmp);
	}
}

void cunfft_destroyPlan_2d2(cunfftplan2d2 *plan)
{
	//create in init
	if ((plan->flags & CUNFFTPLAN_NO_MALLOC_CK) == 0)
	{
        	cudaFree(plan->ckCU);
	}
	cudaFree(plan->adjoint_phiSparM_LocaCU);
	cudaFree(plan->adjoint_phiSparM_jCU);
	cudaFree(plan->adjoint_phiSparMCU);
	cudaFree(plan->adjoint_phiSparM_indexCU);

	if ((plan->flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		cudaFree(plan->trafo_phiSparM_lCU);
		cudaFree(plan->trafo_phiSparMCU);
	}
	else
	{
		cudaFree(plan->phixCU);
    		cudaFree(plan->phiyCU);
    		cudaFree(plan->phix_indexCU);
    		cudaFree(plan->phiy_indexCU);
	}
	if ((plan->flags & CUNFFTPLAN_NO_MALLOC_G) == 0)
	{
		cudaFree(plan->gCU);
	}
	//cudaFree(plan->ghatCU);
	cudaFree(plan->f1CU);
	if ((plan->flags & CUNFFTPLAN_NO_MALLOC_PLAN) == 0)
	{
		cufftDestroy(plan->planTmp);
	}
}

__global__ void nufft_fhat_div_ckCU(CUFFTDATATYPE *f ,DATATYPE *ck,int NX,int NY,DATATYPE thita)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < NX*NY)
    {
        f[pn].x = f[pn].x/(ck[pn]);
        f[pn].y = f[pn].y/(ck[pn]);
    }
}

//search x with a fix y
__global__ void nufft_GPhi_p1CU(CUFFTDATATYPE *f1,CUFFTDATATYPE *g,DATATYPE *trafo_phiSparM,int *trafo_phiSparM_l,int NX,int NY,int m,int M,DATATYPE thita)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < M*(2*m+2))
	{
		int x,y;
		int m2 = 2*m+2;
		int j = pn/m2;
		y = pn%m2;
		f1[pn].x = 0;
		f1[pn].y = 0;
		int index2;
		DATATYPE phi;
		int li = j*m2*m2+y*m2;
		for (x = 0 ; x < m2 ; x++)
		{
			index2 = trafo_phiSparM_l[li+x];
			phi = trafo_phiSparM[li+x];
			f1[pn].x += g[index2].x*phi;
			f1[pn].y += g[index2].y*phi;
		}
	}
}

__global__ void nufft_GPhi_p1CU_noPreCompute(CUFFTDATATYPE *f1,CUFFTDATATYPE *g,DATATYPE *phix,DATATYPE *phiy,int *phix_index,int *phiy_index,int NX,int NY,int m,int M,DATATYPE thita1)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < M*(2*m+2))
	{
		int m2 = 2*m+2;
		int k = pn/m2;
		int j = k*m2+pn%m2;
		int i = k*m2;
		DATATYPE f1tmpx,f1tmpy;
		f1tmpx = 0;
		f1tmpy = 0;
		
		int index2;
		int ny = ceil(NY*thita1);
		DATATYPE phi;
		int l = k*m2+m2;
		int phiy_indexj = phiy_index[j];
		DATATYPE phiyj = phiy[j];
		for (i = k*m2 ; i < l ; i++)
		{
			index2 = phix_index[i]*ny+phiy_indexj;
			phi = phix[i]*phiyj;
			f1tmpx += g[index2].x*phi;
			f1tmpy += g[index2].y*phi;
		}
		f1[pn].x = f1tmpx;
		f1[pn].y = f1tmpy;
	}
}


//combine f1 into f
__global__ void nufft_GPhi_p2CU(CUFFTDATATYPE *f,CUFFTDATATYPE *f1,int m,int M)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < M)
	{
		int m2 = 2*m+2;
		DATATYPE fx,fy;
		int x,index;
		fx = 0;
		fy = 0;
		for (x = 0 ; x < m2 ; x++)
		{
			index = pn*m2+x;
			fx += f1[index].x;
			fy += f1[index].y;
		}
		f[pn].x = fx;
		f[pn].y = fy;
		/*if (pn == 0)
			printf("inside %.20lf %.20lf f1[0] %.20lf %.20lf\n",fx,fy,f1[0].x,f1[0].y);*/
	}
}


__global__ void nufft_assignG_HAT(CUFFTDATATYPE *f ,CUFFTDATATYPE *g,int NX,int NY,int nx,int ny)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < nx*ny)
    {
        int x = pn%nx;
        int y = pn/nx;
        int X = (x+NX/2)%nx;
        int Y = (y+NY/2)%ny;
        if (X >= 0 && X < NX && Y >=0 && Y < NY)
        {
            int i = Y*NX+X;
            g[pn].x = f[i].x;
            g[pn].y = f[i].y;
        }
        else
        {
            g[pn].x = g[pn].y = 0;
        }
    }
}

__global__ void nufft_fhatDivCk_assignGhat(CUFFTDATATYPE *f ,CUFFTDATATYPE *g,DATATYPE *ck,int NX,int NY,int nx,int ny)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < nx*ny)
    {
        //int x = pn%nx;
        //int y = pn/nx;
	int x = pn/ny;
	int y = pn%ny;
        int X = (x+NX/2)%nx;
        int Y = (y+NY/2)%ny;
	DATATYPE cktmp;
        if (X >= 0 && X < NX && Y >=0 && Y < NY)
        {
            //int i = Y*NX+X;
	    int i = X*NY+Y;
	    cktmp = ck[i];
            g[pn].x = f[i].x/cktmp;
            g[pn].y = f[i].y/cktmp;
        }
        else
        {
            g[pn].x = g[pn].y = 0;
        }
    }
}


__global__ void nufft_fhatDivCk_assignGhat_R2C(DATATYPE *f ,CUFFTDATATYPE *g,DATATYPE *ck,int NX,int NY,int nx,int ny)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < nx*ny)
    {
        //int x = pn%nx;
        //int y = pn/nx;
	int x = pn/ny;
	int y = pn%ny;
        int X = (x+NX/2)%nx;
        int Y = (y+NY/2)%ny;
        if (X >= 0 && X < NX && Y >=0 && Y < NY)
        {
            //int i = Y*NX+X;
	    int i = X*NY+Y;
            g[pn].x = f[i]/ck[i];
            g[pn].y = 0;
        }
        else
        {
            g[pn].x = g[pn].y = 0;
        }
    }
}

void cunfft_trafo_2d(cunfftplan2d plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat)
{
	int NX = plan.N[0];
	int NY = plan.N[1];
	int m = plan.m;
	int M = plan.M;
	DATATYPE thita = plan.thita;
	int nx = ceil(NX*thita);
	int ny = ceil(NY*thita);

	//step 1
	//ghat <- fhat/(ck*n)
    nufft_fhatDivCk_assignGhat<<<dim3(GRIDDIMX,((nx*ny - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhat,plan.ghatCU,plan.ckCU,NX,NY,nx,ny);
	//--------------step 1

	//step 2
    	CUFFTEXEC(plan.planTmp, plan.ghatCU,plan.gCU,CUFFT_FORWARD);
	//--------------step 2

	//step 3
	if ((plan.flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		nufft_GPhi_p1CU<<<dim3(GRIDDIMX,((M*(2*m+2)- 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.f1CU,plan.gCU,plan.trafo_phiSparMCU,plan.trafo_phiSparM_lCU,NX,NY,m,M,thita);
    		nufft_GPhi_p2CU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.f1CU,m,M);
	}
	else
	{
		nufft_GPhi_p1CU_noPreCompute<<<dim3(GRIDDIMX,((M*(2*m+2)- 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.f1CU,plan.gCU,plan.phixCU,plan.phiyCU,plan.phix_indexCU,plan.phiy_indexCU,NX,NY,m,M,thita);
		nufft_GPhi_p2CU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.f1CU,m,M);
	}
	//--------------step 3
}

void cunfft_trafo_2d2(cunfftplan2d2 plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat)
{
	int NX = plan.N[0];
	int NY = plan.N[1];
	int m = plan.m;
	int M = plan.M;
	DATATYPE *thita = plan.thita;
	int nx = ceil(NX*thita[0]);
	int ny = ceil(NY*thita[1]);

	//step 1
	//ghat <- fhat/(ck*n)
	cudaMemset(plan.ghatCU,0,nx*ny*sizeof(CUFFTDATATYPE));
        nufft_fhatDivCk_assignGhat<<<dim3(GRIDDIMX,((nx*ny - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhat,plan.ghatCU,plan.ckCU,NX,NY,nx,ny);
	//--------------step 1
	/*{
	DATATYPE * f2 = (DATATYPE*)malloc(nx*ny*sizeof(DATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.ckCU,NX*NY*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("ck %.20lf\n",f2[ii]);
	free(f2);
	}*/
	/*{
	CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.ghatCU,nx*ny*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("ghat %.20lf %.20lf\n",f2[ii].x,f2[ii].y);
	free(f2);
	}*/

	//step 2
    	CUFFTEXEC(plan.planTmp, plan.ghatCU,plan.gCU,CUFFT_FORWARD);
	//--------------step 2

	/*{
	CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.gCU,nx*ny*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("g %.20lf %.20lf\n",f2[ii].x,f2[ii].y);
	free(f2);
	}*/

	//step 3
	if ((plan.flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		nufft_GPhi_p1CU<<<dim3(GRIDDIMX,((M*(2*m+2)- 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.f1CU,plan.gCU,plan.trafo_phiSparMCU,plan.trafo_phiSparM_lCU,NX,NY,m,M,thita[1]);
    		nufft_GPhi_p2CU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.f1CU,m,M);
	}
	else
	{
		nufft_GPhi_p1CU_noPreCompute<<<dim3(GRIDDIMX,((M*(2*m+2)- 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.f1CU,plan.gCU,plan.phixCU,plan.phiyCU,plan.phix_indexCU,plan.phiy_indexCU,NX,NY,m,M,thita[1]);
		nufft_GPhi_p2CU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.f1CU,m,M);
	}
	/*{
	int * phix_index = (int*)malloc(M*(2*m+2)*sizeof(int));
	int * phiy_index = (int*)malloc(M*(2*m+2)*sizeof(int));
	DATATYPE * phix = (DATATYPE*)malloc(M*(2*m+2)*sizeof(DATATYPE));
	DATATYPE * phiy = (DATATYPE*)malloc(M*(2*m+2)*sizeof(DATATYPE));
	CUFFTDATATYPE * g = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
	CUFFTDATATYPE * f1 = (CUFFTDATATYPE*)malloc(M*(2*m+2)*sizeof(CUFFTDATATYPE));
	CUFFTDATATYPE * f3 = (CUFFTDATATYPE*)malloc(M*sizeof(CUFFTDATATYPE));
	
    	int ii,jj;
	cudaMemcpy(phix_index,plan.phix_indexCU,M*(2*m+2)*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(phiy_index,plan.phiy_indexCU,M*(2*m+2)*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(phix,plan.phixCU,M*(2*m+2)*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	cudaMemcpy(phiy,plan.phiyCU,M*(2*m+2)*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	cudaMemcpy(g,plan.gCU,nx*ny*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	cudaMemcpy(f1,plan.f1CU,M*(2*m+2)*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	cudaMemcpy(f3,f,M*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	CUFFTDATATYPE tmp,tmp2,tmp3;
	tmp.x = tmp.y=0;
	tmp2.x = tmp2.y = 0;
	tmp3.x = tmp3.y = 0;
	for (ii = 0 ; ii < 2*m+2 ; ii++)
	{
		tmp2.x = tmp2.y = 0;
		for (jj = 0 ; jj < 2*m+2 ; jj++)
		{
			tmp.x += g[phix_index[jj]*ny+phiy_index[ii]].x*phiy[ii]*phix[jj];
			tmp.y += g[phix_index[jj]*ny+phiy_index[ii]].y*phiy[ii]*phix[jj];
			tmp2.x += g[phix_index[jj]*ny+phiy_index[ii]].x*phiy[ii]*phix[jj];
			tmp2.y += g[phix_index[jj]*ny+phiy_index[ii]].y*phiy[ii]*phix[jj];
		}
		tmp3.x += f1[ii].x;
		tmp3.y += f1[ii].y;
		printf("f1 %.20lf %.20lf\ntmp2 %.20lf %.20lf\ntmp3 %.20lf %.20lf\n",f1[ii].x,f1[ii].y,tmp2.x,tmp2.y,tmp3.x,tmp3.y);
	}
		printf("tmp %.20lf %.20lf\n",tmp.x,tmp.y);
		//printf("phix %.20lf phiy %.20lf phix_index %d phiy_index %d gx %.20lf gy %.20lf\n",phix[0],phiy[ii],phix_index[0],phiy_index[ii],g[phix_index[0]*ny+phiy_index[ii]].x,g[phix_index[0]*ny+phiy_index[ii]].y);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("f %.20lf %.20lf\n",f3[ii].x,f3[ii].y);
	free(phix_index);
	free(phiy_index);
	free(phix);
	free(phiy);
	free(g);
	free(f1);
	free(f3);
	}*/

	//--------------step 3
}

void cunfft_trafo_2d_R2C(cunfftplan2d plan,CUFFTDATATYPE* f,DATATYPE* fhat)
{
	int NX = plan.N[0];
	int NY = plan.N[1];
	int m = plan.m;
	int M = plan.M;
	DATATYPE thita = plan.thita;
	int nx = ceil(NX*thita);
	int ny = ceil(NY*thita);

	//step 1
	//ghat <- fhat/(ck*n)
    nufft_fhatDivCk_assignGhat_R2C<<<dim3(GRIDDIMX,((nx*ny - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhat,plan.ghatCU,plan.ckCU,NX,NY,nx,ny);
	//--------------step 1

	//step 2
    	CUFFTEXEC(plan.planTmp, plan.ghatCU,plan.gCU,CUFFT_FORWARD);
	//--------------step 2

	//step 3
	if ((plan.flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		nufft_GPhi_p1CU<<<dim3(GRIDDIMX,((M*(2*m+2)- 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.f1CU,plan.gCU,plan.trafo_phiSparMCU,plan.trafo_phiSparM_lCU,NX,NY,m,M,thita);
    		nufft_GPhi_p2CU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.f1CU,m,M);
	}
	else
	{
		nufft_GPhi_p1CU_noPreCompute<<<dim3(GRIDDIMX,((M*(2*m+2)- 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.f1CU,plan.gCU,plan.phixCU,plan.phiyCU,plan.phix_indexCU,plan.phiy_indexCU,NX,NY,m,M,thita);
		nufft_GPhi_p2CU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.f1CU,m,M);
	}
	//--------------step 3
}

void cunfft_trafo_2d_R2C2(cunfftplan2d2 plan,CUFFTDATATYPE* f,DATATYPE* fhat)
{
	int NX = plan.N[0];
	int NY = plan.N[1];
	int m = plan.m;
	int M = plan.M;
	DATATYPE *thita = plan.thita;
	int nx = ceil(NX*thita[0]);
	int ny = ceil(NY*thita[1]);

        /*cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    float msecTotal1,msecTotal2;*/

	//step 1
	//ghat <- fhat/(ck*n)
	//cudaEventRecord(start1, NULL);
    	nufft_fhatDivCk_assignGhat_R2C<<<dim3(GRIDDIMX,((nx*ny - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhat,plan.ghatCU,plan.ckCU,NX,NY,nx,ny);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("nufft_fhatDivCk_assignGhat_R2C : %f\n ",msecTotal1);*/
	//--------------step 1
	/*{
	DATATYPE * f2 = (DATATYPE*)malloc(nx*ny*sizeof(DATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.ckCU,NX*NY*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("R2C ck %.20lf\n",f2[ii]);
	free(f2);
	}*/
	/*{
	CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.ghatCU,nx*ny*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("R2C ghat %.20lf %.20lf\n",f2[ii].x,f2[ii].y);
	free(f2);
	}*/

	//step 2
	//cudaEventRecord(start1, NULL);
    	CUFFTEXEC(plan.planTmp, plan.ghatCU,plan.gCU,CUFFT_FORWARD);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("CUFFTEXEC : %f\n ",msecTotal1);*/
	//--------------step 2

	/*{
	CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.gCU,nx*ny*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("R2C g %.20lf %.20lf\n",f2[ii].x,f2[ii].y);
	free(f2);
	}*/

	//step 3
	//cudaEventRecord(start1, NULL);
	if ((plan.flags & CUNFFTPLAN_NO_PRECOMPUTE) == 0)
	{
		nufft_GPhi_p1CU<<<dim3(GRIDDIMX,((M*(2*m+2)- 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.f1CU,plan.gCU,plan.trafo_phiSparMCU,plan.trafo_phiSparM_lCU,NX,NY,m,M,thita[1]);
    		nufft_GPhi_p2CU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.f1CU,m,M);
	}
	else
	{
		nufft_GPhi_p1CU_noPreCompute<<<dim3(GRIDDIMX,((M*(2*m+2)- 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.f1CU,plan.gCU,plan.phixCU,plan.phiyCU,plan.phix_indexCU,plan.phiy_indexCU,NX,NY,m,M,thita[1]);
		nufft_GPhi_p2CU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.f1CU,m,M);
	}
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("nufft_GPhi_p1CU_noPreCompute : %f\n ",msecTotal1);*/
	//--------------step 3

	/*{
	CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
    	int ii;
	cudaMemcpy(f2,f,M*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("R2C f %.20lf %.20lf\n",f2[ii].x,f2[ii].y);
	free(f2);
	}*/
}

__global__ void nufft_adjoint_fullpre_FPhi(CUFFTDATATYPE *g,CUFFTDATATYPE* f,int *phi2dSparseMatrixLocate,int *phi2dSparseMatrixIndex,DATATYPE *phi2dSparseMatrix,int *phi2dSparseMatrixJ,int phi2dSparseMatrixPara0,int phi2dSparseMatrixPara1)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    	int pn = x + y * blockDim.x * gridDim.x;
	if (pn < phi2dSparseMatrixPara1)
	{
		int gindex = phi2dSparseMatrixIndex[pn];
		int begin = phi2dSparseMatrixLocate[pn];
		int end = pn < phi2dSparseMatrixPara1 - 1 ? phi2dSparseMatrixLocate[pn+1]:phi2dSparseMatrixPara0;
		int i,j;
		CUFFTDATATYPE gtmp;
		DATATYPE phi;
		gtmp.x = gtmp.y = 0;
		for (i = begin ; i < end; i++)
		{
			j = phi2dSparseMatrixJ[i];
			phi = phi2dSparseMatrix[i];
			gtmp.x += f[j].x*phi;
			gtmp.y += f[j].y*phi;
		}
		g[gindex].x = gtmp.x;
		g[gindex].y = gtmp.y;
 	}
}

__global__ void nufft_assignF_HAT(CUFFTDATATYPE *f_hat,CUFFTDATATYPE *g_hat,DATATYPE *ck,int NX,int NY,int nx,int ny)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < NX*NY)
    {
	int X = pn/NY;
        int Y = pn%NY;
        int x = (X+nx-NX/2)%nx;
        int y = (Y+ny-NY/2)%ny;
	int i = x*ny+y;
        f_hat[pn].x = g_hat[i].x/(ck[pn]);
        f_hat[pn].y = g_hat[i].y/(ck[pn]);
	/*if (pn == 0)
		printf("inside %f %f %f %f %f\n",f_hat[pn].x,f_hat[pn].y,g_hat[i].x,g_hat[i].y,ck[pn]);*/
    }
}

__global__ void nufft_assignF_HAT_1d(CUFFTDATATYPE *f_hat,CUFFTDATATYPE *g_hat,DATATYPE *ck,int N,int n)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < N)
    {
	int X = pn;
        int x = (X+n-N/2)%n;
        f_hat[pn].x = g_hat[x].x/(ck[pn]);
        f_hat[pn].y = g_hat[x].y/(ck[pn]);
    }
}

void cunfft_adjoint_2d(cunfftplan2d plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat)
{
	int NX = plan.N[0];
	int NY = plan.N[1];
	DATATYPE thita = plan.thita;
	int nx = ceil(NX*thita);
	int ny = ceil(NY*thita);

	//step 1
	cudaMemset(plan.gCU,0,nx*ny*sizeof(CUFFTDATATYPE));
	nufft_adjoint_fullpre_FPhi<<<dim3(GRIDDIMX,((plan.adjoint_phiSparM_para[1] - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.gCU,f,plan.adjoint_phiSparM_LocaCU,plan.adjoint_phiSparM_indexCU,plan.adjoint_phiSparMCU,plan.adjoint_phiSparM_jCU,plan.adjoint_phiSparM_para[0],plan.adjoint_phiSparM_para[1]);
	//--------------step 1

	//step 2
    	CUFFTEXEC(plan.planTmp, plan.gCU, plan.ghatCU, CUFFT_INVERSE);
	//--------------step 2

	/*CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.ghatCU,nx*ny*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("%.20lf %.20lf\n",f2[ii].x,f2[ii].y);
	free(f2);
	return;*/

	//step 3
	nufft_assignF_HAT<<<dim3(GRIDDIMX,((NX*NY - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhat,plan.ghatCU,plan.ckCU,NX,NY,nx,ny);
	//--------------step 3
}

void cunfft_adjoint_2d2(cunfftplan2d2 plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat)
{
	int NX = plan.N[0];
	int NY = plan.N[1];
	DATATYPE* thita = plan.thita;
	int nx = ceil(NX*thita[0]);
	int ny = ceil(NY*thita[1]);

	/*cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    float msecTotal1,msecTotal2;*/

	//step 1
	//cudaEventRecord(start1, NULL);
	cudaMemset(plan.gCU,0,nx*ny*sizeof(CUFFTDATATYPE));
	nufft_adjoint_fullpre_FPhi<<<dim3(GRIDDIMX,((plan.adjoint_phiSparM_para[1] - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.gCU,f,plan.adjoint_phiSparM_LocaCU,plan.adjoint_phiSparM_indexCU,plan.adjoint_phiSparMCU,plan.adjoint_phiSparM_jCU,plan.adjoint_phiSparM_para[0],plan.adjoint_phiSparM_para[1]);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("nufft_adjoint_fullpre_FPhi : %f\n ",msecTotal1);*/
	//printf("plan.adjoint_phiSparM_para[0] %d ,plan.adjoint_phiSparM_para[1] %d\n",plan.adjoint_phiSparM_para[0],plan.adjoint_phiSparM_para[1]);
	//--------------step 1
	/*{
        CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.gCU,nx*ny*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("g %.20lf %.20lf\n",f2[ii].x,f2[ii].y);
	free(f2);
	}*/
	//step 2
        //cudaEventRecord(start1, NULL);
    	CUFFTEXEC(plan.planTmp, plan.gCU, plan.ghatCU, CUFFT_INVERSE);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("CUFFTEXEC : %f\n ",msecTotal1);*/
	//--------------step 2

	/*CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(nx*ny*sizeof(CUFFTDATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.ghatCU,nx*ny*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("g_hat %.20lf %.20lf\n",f2[ii].x,f2[ii].y);
	free(f2);
	//return;*/

	//step 3
	//cudaEventRecord(start1, NULL);
	nufft_assignF_HAT<<<dim3(GRIDDIMX,((NX*NY - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhat,plan.ghatCU,plan.ckCU,NX,NY,nx,ny);
	/*cudaDeviceSynchronize();
   	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
    	msecTotal1 = 0.0f;
    	cudaEventElapsedTime(&msecTotal1, start1, stop1);
    	printf("nufft_assignF_HAT : %f\n ",msecTotal1);*/
	/*{
	DATATYPE * f2 = (DATATYPE*)malloc(NX*NY*sizeof(DATATYPE));
    	int ii;
	cudaMemcpy(f2,plan.ckCU,NX*NY*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("ck %.20lf\n",f2[ii]);
	free(f2);
	}*/
	//--------------step 3
}

//----------------------------------------------------------------------------------------------------trafo 1d

__global__ void nufft_Phi1dCU(DATATYPE *phix ,int *phix_index,DATATYPE *pdx1d,int N,int m,int M,DATATYPE thita,DATATYPE b,DATATYPE bf)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < M*(2*m+2))
	{
		int j = pn/(2*m+2);
		int n = ceil(N*thita);
		int l = floor(n*pdx1d[j]) - m + pn%(2*m+2);
		DATATYPE x1 = pdx1d[j] - (DATATYPE)l/(DATATYPE)n;
		phix[pn] = bf*exp(-((n*x1)*(n*x1)/b));
		phix_index[pn] = (l + n) % n;
	}
}

__global__ void nufft_ck1d(DATATYPE *res ,int N,int m, DATATYPE thita)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < N)
    {
        DATATYPE x = pn-N/2;
        DATATYPE n = ceil(N*thita);
        DATATYPE b = 2*thita*m/((2*thita-1)*PI);
        DATATYPE resx = exp(-b*((PI*x/n)*(PI*x/n)));
        res[pn] = resx;
    }
}

__global__ void nufft_fhat_div_ck1dCU(CUFFTDATATYPE *f ,DATATYPE *ck1d,int N,DATATYPE thita)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < N)
    {
        f[pn].x = f[pn].x/(ck1d[pn]);
        f[pn].y = f[pn].y/(ck1d[pn]);
    }
}

void GenPhi1dSparseMatrix_trafo(int *phi1dSparseMatrixL,DATATYPE *phi1dSparseMatrix,DATATYPE *phi,int *phi_index,int n,int m,int M)
{
	int k,j;
	int m2 = 2*m+2;
	int tmp = 0;
	for (k = 0 ; k < M ; k++)
		for (j = k*m2 ; j < k*m2+m2 ; j++,tmp++)
		{
			phi1dSparseMatrixL[tmp] = phi_index[j];
			phi1dSparseMatrix[tmp] = phi[j];
		}
}

void cunfft_initPlan_1d(cunfftplan1d *plan,int N,DATATYPE thita,int M,int m,DATATYPE *pdx1d)
{
	plan->N = N;
	plan->thita = thita;
	plan->m = m;
	plan->M = M;

	int n=ceil(N*thita);
        //printf("%d %d %f\n",N,n,thita);

	cufftPlan1d(&(plan->planTmp),n,CUFFTTYPE,1);

	cudaMalloc((void**)&(plan->gCU),n*sizeof(CUFFTDATATYPE));
	//cudaMalloc((void**)&(plan->ghatCU),n*sizeof(CUFFTDATATYPE));
	plan->ghatCU = plan->gCU;

        cudaMalloc((void**)&(plan->ckCU), sizeof(DATATYPE)*N);
	nufft_ck1d<<<dim3(GRIDDIMX,((N - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan->ckCU,N,m,thita);

	//calculate phi1d
	DATATYPE *phiCU;
	int *phi_indexCU;
	DATATYPE *pdx1dCU;

	cudaMalloc((void**)&phiCU,sizeof(DATATYPE)*M*(2*m+2));
    	cudaMalloc((void**)&phi_indexCU,sizeof(int)*M*(2*m+2));
	DATATYPE * phi = (DATATYPE *)malloc(sizeof(DATATYPE)*M*(2*m+2));
    	int *  phi_index = (int *)malloc(sizeof(int)*M*(2*m+2));

		//calculate phi1d
	cudaMalloc((void**)&pdx1dCU,sizeof(DATATYPE)*M);
	cudaMemcpy(pdx1dCU, pdx1d, sizeof(DATATYPE)*M, cudaMemcpyHostToDevice);
	DATATYPE b = 2*thita*m/((2*thita-1)*PI);
    	DATATYPE bf = 1.0/sqrt(PI*b);
	nufft_Phi1dCU<<<dim3(GRIDDIMX,((M*(2*m+2) - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(phiCU,phi_indexCU,pdx1dCU,N,m,M,thita,b,bf);
	cudaMemcpy(phi, phiCU, sizeof(DATATYPE)*M*(2*m+2), cudaMemcpyDeviceToHost);
    	cudaMemcpy(phi_index, phi_indexCU, sizeof(int)*M*(2*m+2), cudaMemcpyDeviceToHost);
		//--------------calculate phi1d

	//calculate adjoint phi1d in CPU
	int phi1dSparseMatrixPara[2];
    	int *phi1dSparseMatrixIndex = (int *)malloc(M*(2*m+2)*sizeof(int));
    	int *phi1dSparseMatrixNum = (int *)malloc(M*(2*m+2)*sizeof(int));
	int *phi1dMark = (int *)malloc(n*sizeof(int));
    	memset(phi1dMark,0,sizeof(int)*n);
    	genPhi1dMark_adjoint(phi1dMark,phi_index,m,M,n);
    	preGenPhi1dSparseMatrix_adjoint(phi1dMark,phi1dSparseMatrixIndex,phi1dSparseMatrixNum,phi1dSparseMatrixPara,n);
    	int *phi1dSparseMatrixLocate = (int *)malloc(phi1dSparseMatrixPara[1]*sizeof(int));
    	preGenPhi1dSparseMatrix2_adjoint(phi1dSparseMatrixNum,phi1dSparseMatrixLocate,phi1dSparseMatrixPara[1]);
    	int *phi1dSparseMatrixJ = (int *)malloc(phi1dSparseMatrixPara[0]*sizeof(int));
    	DATATYPE *phi1dSparseMatrix = (DATATYPE *)malloc(phi1dSparseMatrixPara[0]*sizeof(DATATYPE));
    	GenPhi1dSparseMatrix_adjoint(phi1dSparseMatrixJ,phi1dSparseMatrix,phi,phi_index,phi1dSparseMatrixNum,phi1dSparseMatrixIndex,phi1dSparseMatrixPara,n,m,M);
	cudaMalloc((void**)&(plan->adjoint_phiSparM_LocaCU),phi1dSparseMatrixPara[1]*sizeof(int));
    	cudaMalloc((void**)&(plan->adjoint_phiSparM_indexCU),phi1dSparseMatrixPara[1]*sizeof(int));
    	cudaMalloc((void**)&(plan->adjoint_phiSparMCU),phi1dSparseMatrixPara[0]*sizeof(DATATYPE));
    	cudaMalloc((void**)&(plan->adjoint_phiSparM_jCU),phi1dSparseMatrixPara[0]*sizeof(int));

    	cudaMemcpy(plan->adjoint_phiSparM_LocaCU, phi1dSparseMatrixLocate, phi1dSparseMatrixPara[1]*sizeof(int), cudaMemcpyHostToDevice);

    	cudaMemcpy(plan->adjoint_phiSparM_indexCU, phi1dSparseMatrixIndex, phi1dSparseMatrixPara[1]*sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(plan->adjoint_phiSparMCU,phi1dSparseMatrix, phi1dSparseMatrixPara[0]*sizeof(DATATYPE), cudaMemcpyHostToDevice);
    	cudaMemcpy(plan->adjoint_phiSparM_jCU, phi1dSparseMatrixJ, phi1dSparseMatrixPara[0]*sizeof(int), cudaMemcpyHostToDevice);
	plan->adjoint_phiSparM_para[0] = phi1dSparseMatrixPara[0];
	plan->adjoint_phiSparM_para[1] = phi1dSparseMatrixPara[1];
	free(phi1dMark);
	free(phi1dSparseMatrixIndex);
	free(phi1dSparseMatrixNum);
	free(phi1dSparseMatrixLocate);
	free(phi1dSparseMatrixJ);
	free(phi1dSparseMatrix);
		//--------------calculate adjoint phi1d in CPU
		//calculate trofa phi1d in CPU
	plan->trafo_phiSparM_para[0] = M*(2*m+2);
	plan->trafo_phiSparM_para[1] = M;
	phi1dSparseMatrixJ = (int *)malloc(plan->trafo_phiSparM_para[0]*sizeof(int));
	phi1dSparseMatrix = (DATATYPE *)malloc(plan->trafo_phiSparM_para[0]*sizeof(DATATYPE));
	GenPhi1dSparseMatrix_trafo(phi1dSparseMatrixJ,phi1dSparseMatrix,phi,phi_index,n,m,M);

	cudaMalloc((void**)&(plan->trafo_phiSparMCU),plan->trafo_phiSparM_para[0]*sizeof(DATATYPE));
    	cudaMalloc((void**)&(plan->trafo_phiSparM_lCU),plan->trafo_phiSparM_para[0]*sizeof(int));

	cudaMemcpy(plan->trafo_phiSparMCU,phi1dSparseMatrix, plan->trafo_phiSparM_para[0]*sizeof(DATATYPE), cudaMemcpyHostToDevice);
    	cudaMemcpy(plan->trafo_phiSparM_lCU, phi1dSparseMatrixJ, plan->trafo_phiSparM_para[0]*sizeof(int), cudaMemcpyHostToDevice);

	free(phi1dSparseMatrixJ);
	free(phi1dSparseMatrix);
		//--------------calculate trofa phi1d in CPU
	free(phi);
    	free(phi_index);
	cudaFree(phiCU);
    	cudaFree(phi_indexCU);
	cudaFree(pdx1dCU);
	//--------------calculate phi1d
}

void cunfft_destroyPlan_1d(cunfftplan1d *plan)
{
	//create in init
        cudaFree(plan->ckCU);

	cudaFree(plan->adjoint_phiSparM_LocaCU);
	cudaFree(plan->adjoint_phiSparM_jCU);
	cudaFree(plan->adjoint_phiSparMCU);
	cudaFree(plan->adjoint_phiSparM_indexCU);

	cudaFree(plan->trafo_phiSparM_lCU);
	cudaFree(plan->trafo_phiSparMCU);
	cudaFree(plan->gCU);
	//cudaFree(plan->ghatCU);
	cufftDestroy(plan->planTmp);
}

__global__ void nufft_assignG_HAT_1d(CUFFTDATATYPE *f ,CUFFTDATATYPE *g,int N,int n)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < n)
    {
        int x = pn;
        int X = (x+N/2)%n;
        if (X >= 0 && X < N)
        {
            int i = X;
            g[pn].x = f[i].x;
            g[pn].y = f[i].y;
        }
        else
        {
            g[pn].x = g[pn].y = 0;
        }
    }
}

__global__ void nufft_fhatDivCk1d_assignGhat_1d(CUFFTDATATYPE *f ,CUFFTDATATYPE *g,DATATYPE* ck,int N,int n)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    if (pn < n)
    {
        int x = pn;
        int X = (x+N/2)%n;
        if (X >= 0 && X < N)
        {
            int i = X;
            g[pn].x = f[i].x/ck[i];
            g[pn].y = f[i].y/ck[i];
        }
        else
        {
            g[pn].x = g[pn].y = 0;
        }
    }
}

__global__ void nufft_GPhi_1dCU(CUFFTDATATYPE *f,CUFFTDATATYPE *g,DATATYPE *trafo_phiSparM,int *trafo_phiSparM_l,int N,int m,int M,DATATYPE thita)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < M)
	{
		int m2 = 2*m+2;
		int j = pn;
		f[pn].x = 0;
		f[pn].y = 0;
		int index2;
		DATATYPE phi;
		int li = j*m2;
		int x;
		for (x = 0 ; x < m2 ; x++)
		{
			index2 = trafo_phiSparM_l[li+x];
			phi = trafo_phiSparM[li+x];
			f[pn].x += g[index2].x*phi;
			f[pn].y += g[index2].y*phi;
		}
	}
}

void cunfft_trafo_1d(cunfftplan1d plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat)
{
	DATATYPE thita = plan.thita;
	int M = plan.M;
	int m = plan.m;
	int N = plan.N;
	int n = ceil(N*thita);

	//step 1
	//g_hat <- f_hat/(ck)
	nufft_fhatDivCk1d_assignGhat_1d<<<dim3(GRIDDIMX,((n - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhat ,plan.ghatCU,plan.ckCU,N,n);
	//--------------step 1

	//step 2
	/*CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(n*sizeof(CUFFTDATATYPE));
	cudaMemcpy(f2,plan.ghatCU,n*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	int ii;
	for (ii = 0 ; ii < 20 ; ii++)
		printf("xxxxx%.20g %.20g\n",f2[ii].x,f2[ii].y);*/
	//free(f2);
	//exit(0);
    	CUFFTEXEC(plan.planTmp,plan.ghatCU,plan.gCU,CUFFT_FORWARD);
	/*CUFFTDATATYPE * f2 = (CUFFTDATATYPE*)malloc(n*sizeof(CUFFTDATATYPE));
	cudaMemcpy(f2,plan.gCU,n*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	int ii;
	for (ii = 0 ; ii < 20 ; ii++)
		printf("yyyyyy%.20lf %.20f\n",f2[ii].x,f2[ii].y);*/
	//--------------step 2

	//step 3
	nufft_GPhi_1dCU<<<dim3(GRIDDIMX,((M - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(f,plan.gCU,plan.trafo_phiSparMCU,plan.trafo_phiSparM_lCU,N,m,M,thita);
	/*cudaMemcpy(f2,f,M*sizeof(CUFFTDATATYPE),cudaMemcpyDeviceToHost);
	for (ii = 0 ; ii < 20 ; ii++)
		printf("ttttttt%.20lf %.20f\n",f2[ii].x,f2[ii].y);
	free(f2);
	exit(0);*/
	//--------------step 3
}

void cunfft_adjoint_1d(cunfftplan1d plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat)
{
	DATATYPE thita = plan.thita;
	int N = plan.N;
	int n = ceil(N*thita);

	//step 1
	cudaMemset(plan.gCU,0,n*sizeof(CUFFTDATATYPE));	
	nufft_adjoint_fullpre_FPhi<<<dim3(GRIDDIMX,((plan.adjoint_phiSparM_para[1] - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(plan.gCU,f,plan.adjoint_phiSparM_LocaCU,plan.adjoint_phiSparM_indexCU,plan.adjoint_phiSparMCU,plan.adjoint_phiSparM_jCU,plan.adjoint_phiSparM_para[0],plan.adjoint_phiSparM_para[1]);
	//--------------step 1
	
	//step 2
    	CUFFTEXEC(plan.planTmp, plan.gCU, plan.ghatCU, CUFFT_INVERSE);
	//--------------step 2

	//step 3
	nufft_assignF_HAT_1d<<<dim3(GRIDDIMX,((N - 1)/BLOCKDIM)/GRIDDIMX+1),dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhat,plan.ghatCU,plan.ckCU,N,n);
	//--------------step 3
}

//-----------------------------------------------------------------only for ICON
__global__ void complexMulDoubleCU(CUFFTDATATYPE* f,DATATYPE *weight,int M)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < M)
	{
		f[pn].x = f[pn].x*weight[pn];
		f[pn].y = f[pn].y*weight[pn];
	}
}

__global__ void complexCopyCU(CUFFTDATATYPE* dest,CUFFTDATATYPE* src,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		dest[pn].x = src[pn].x;
		dest[pn].y = src[pn].y;
	}
}

__global__ void complexMinusCU(CUFFTDATATYPE* r,CUFFTDATATYPE* fhat,CUFFTDATATYPE* Ahwb,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		r[pn].x = fhat[pn].x - Ahwb[pn].x;
		r[pn].y = fhat[pn].y - Ahwb[pn].y;
	}
}

__global__ void ICONUpdateCU(CUFFTDATATYPE* m,CUFFTDATATYPE* r,ALPHADATATYPE *al,int dataType,DATATYPE threshold,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		m[pn].x = m[pn].x - al[0]*r[pn].x;
		m[pn].y = 0;
		if (dataType == 1)
		{
			if (m[pn].x > threshold)
				m[pn].x = threshold;
		}
		else
		{
			if (m[pn]. x < threshold)
				m[pn].x = threshold;
		}
	}
}

__global__ void ICONUpdateCU_R(DATATYPE* m,CUFFTDATATYPE* r,ALPHADATATYPE *al,int dataType,DATATYPE threshold,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		m[pn] = m[pn] - al[0]*r[pn].x;
		if (dataType == 1)
		{
			if (m[pn] > threshold)
				m[pn] = threshold;
		}
		else
		{
			if (m[pn] < threshold)
				m[pn] = threshold;
		}
	}
}

__global__ void INFRUpdateCU(CUFFTDATATYPE* m,CUFFTDATATYPE* r,ALPHADATATYPE *al,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		m[pn].x = m[pn].x - al[0]*r[pn].x;
		m[pn].y = 0;
	}
}

__global__ void INFRUpdateCU_R(DATATYPE* m,CUFFTDATATYPE* r,ALPHADATATYPE *al,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    	int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    	int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		m[pn] = m[pn] - al[0]*r[pn].x;
	}
}

__global__ void alpha0(ALPHADATATYPE *alpha0Tmp,CUFFTDATATYPE* r,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		alpha0Tmp[pn] = (ALPHADATATYPE)r[pn].x*(ALPHADATATYPE)r[pn].x + (ALPHADATATYPE)r[pn].y*(ALPHADATATYPE)r[pn].y;
	}
}

__global__ void alpha1(ALPHADATATYPE *alpha0Tmp,CUFFTDATATYPE* rhat,DATATYPE *weight,int size)
{
	int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
	if (pn < size)
	{
		alpha0Tmp[pn] = (ALPHADATATYPE)rhat[pn].x*(ALPHADATATYPE)rhat[pn].x*(ALPHADATATYPE)weight[pn] + (ALPHADATATYPE)rhat[pn].y*(ALPHADATATYPE)rhat[pn].y*(ALPHADATATYPE)weight[pn];
	}
}

__global__ void alpha2(ALPHADATATYPE *rtr,ALPHADATATYPE *rtar,ALPHADATATYPE * al)
{
	int pn = threadIdx.x;
	if (pn == 0)
	{
		if (rtar[0] != 0)
			al[0] = rtr[0]/rtar[0];
		else
			al[0] = 0;
	}
}

__device__ void addShared256(ALPHADATATYPE* data, int tid,int lowBound) {
    int i;
    for (i = 128 ; i >= 1 ; i/=2)
    {
	if (tid < i && tid + i < lowBound){
		data[tid] += data[tid + i];
	}
	__syncthreads();
    }
}

__global__ void matrixAdd(ALPHADATATYPE *matrix,int size)
{
    int blockid = blockIdx.x+blockIdx.y*gridDim.x;
    int threadid = threadIdx.x+threadIdx.y*blockDim.x;
    int pn = threadid + blockid*(blockDim.x*blockDim.y);
    __shared__ ALPHADATATYPE data[ALPHABLOCKDIM];
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

void initGICONPara(GICONPara *GP,int nx,int nz,DATATYPE *weight)
{
    //malloc for alpha
    cudaMalloc((void**)&(GP->alphaTmpCU),(nx*nx+1)*sizeof(ALPHADATATYPE));
    cudaMalloc((void**)&(GP->alCU),sizeof(ALPHADATATYPE));
    //end of malloc for alpha

    cudaMalloc((void**)&(GP->AhwbCU),nx*nx*sizeof(CUFFTDATATYPE));

    int M = nx*nz;
    int N[2];
    N[0] = nx;
    N[1] = nx;

    
    cudaMalloc((void**)&(GP->fhatCU),N[0]*N[1]*sizeof(CUFFTDATATYPE));
    GP->rhatCU = GP->fhatCU;
    //cudaMalloc((void**)&(GP->rCU),M*sizeof(CUFFTDATATYPE));
    //cudaMalloc((void**)&(GP->rhatCU),N[0]*N[1]*sizeof(CUFFTDATATYPE));
    cudaMalloc((void**)&(GP->mCU),N[0]*N[1]*sizeof(DATATYPE));
     

    cudaMalloc((void**)&(GP->fhat1dCU),N[0]*sizeof(CUFFTDATATYPE));

    cudaMalloc((void**)&(GP->fCU),M*sizeof(CUFFTDATATYPE));
    GP->rCU = GP->fCU;

    cudaMalloc((void**)&(GP->weightCU),nx*nz*sizeof(DATATYPE));
    cudaMemcpy(GP->weightCU,weight,nx*nz*sizeof(DATATYPE),cudaMemcpyHostToDevice);

    GP->reprojection = (CUFFTDATATYPE*)malloc(nx*nx*sizeof(CUFFTDATATYPE));
    GP->m = (DATATYPE*)malloc(nx*nx*sizeof(DATATYPE));
}

void initGICONPara2(GICONPara *GP,int nx,int thickness,int nz,DATATYPE *weight)
{
    //malloc for alpha
    cudaMalloc((void**)&(GP->alphaTmpCU),(nx*thickness+1)*sizeof(ALPHADATATYPE));
    cudaMalloc((void**)&(GP->alCU),sizeof(ALPHADATATYPE));
    //end of malloc for alpha

    cudaMalloc((void**)&(GP->AhwbCU),nx*thickness*sizeof(CUFFTDATATYPE));

    int M = nx*nz;
    int N[2];
    N[0] = nx;
    N[1] = thickness;

    
    cudaMalloc((void**)&(GP->fhatCU),N[0]*N[1]*sizeof(CUFFTDATATYPE));
    GP->rhatCU = GP->fhatCU;
    //cudaMalloc((void**)&(GP->rCU),M*sizeof(CUFFTDATATYPE));
    //cudaMalloc((void**)&(GP->rhatCU),N[0]*N[1]*sizeof(CUFFTDATATYPE));
    cudaMalloc((void**)&(GP->mCU),N[0]*N[1]*sizeof(DATATYPE));
     

    cudaMalloc((void**)&(GP->fhat1dCU),N[0]*sizeof(CUFFTDATATYPE));

    cudaMalloc((void**)&(GP->fCU),M*sizeof(CUFFTDATATYPE));
    GP->rCU = GP->fCU;

    cudaMalloc((void**)&(GP->weightCU),nx*nz*sizeof(DATATYPE));
    cudaMemcpy(GP->weightCU,weight,nx*nz*sizeof(DATATYPE),cudaMemcpyHostToDevice);

    GP->reprojection = (CUFFTDATATYPE*)malloc(nx*nx*sizeof(CUFFTDATATYPE));
    GP->m = (DATATYPE*)malloc(nx*thickness*sizeof(DATATYPE));
}


void destroyGICONPara(GICONPara *GP)
{
    cudaFree(GP->AhwbCU);
    cudaFree(GP->weightCU);
    cudaFree(GP->alphaTmpCU);
    cudaFree(GP->fCU);
    cudaFree(GP->fhatCU);
    cudaFree(GP->rCU);
    cudaFree(GP->rhatCU);
    cudaFree(GP->mCU);
    cudaFree(GP->fhat1dCU);
    free(GP->m);
    free(GP->reprojection);
}



