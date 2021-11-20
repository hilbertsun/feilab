
#include "cufft.h"
#include "stdio.h"

#define NFFTM 6

#define PI2 6.2831853071796
#define PI 3.1415926535898
#define BLOCKDIM 1024
#define BLOCKDIMX  32
#define BLOCKDIMY  32
#define GRIDDIMX  4

//only for ICON matrixAdd in the calculation of alpha 
#define ALPHABLOCKDIM 256
#define ALPHABLOCKDIMX  16
#define ALPHABLOCKDIMY  16
#define ALPHAGRIDDIMX  4


#define ALPHADATATYPE double

#define MAXGPUNUM 100

#define USE_FLOAT

#ifndef USE_FLOAT
#define DATATYPE double
#define CUFFTDATATYPE cufftDoubleComplex
#define CUFFTTYPE CUFFT_Z2Z
#define CUFFTEXEC cufftExecZ2Z
#else
#define DATATYPE float
#define CUFFTDATATYPE cufftComplex
#define CUFFTTYPE CUFFT_C2C
#define CUFFTEXEC cufftExecC2C
#endif


//flags used in cunfftplan2d
#define CUNFFTPLAN_MALLOCALL  (0)
#define CUNFFTPLAN_NO_MALLOC_CK  (1U << 0)
#define CUNFFTPLAN_NO_MALLOC_G   (1U << 1)
#define CUNFFTPLAN_NO_MALLOC_PLAN   (1U << 2)
#define CUNFFTPLAN_NO_PRECOMPUTE  (1U << 3)

#ifndef CUNFFTP2D
#define CUNFFTP2D
typedef struct{
	DATATYPE *phixCU,*phiyCU;
	int *phix_indexCU,*phiy_indexCU;
	int N[2];
	DATATYPE thita;
	int m;
	int M;
	DATATYPE *ckCU;
	int *trafo_phiSparM_lCU;
     	DATATYPE *trafo_phiSparMCU;
	int *adjoint_phiSparM_LocaCU;
	int *adjoint_phiSparM_jCU;
	int *adjoint_phiSparM_indexCU;
	DATATYPE *adjoint_phiSparMCU;
	int trafo_phiSparM_para[2],adjoint_phiSparM_para[2];
	CUFFTDATATYPE *gCU,*ghatCU,*f1CU;
	cufftHandle planTmp;
	int flags;
} cunfftplan2d;
#endif

#ifndef CUNFFTP2D2
#define CUNFFTP2D2
typedef struct{
	DATATYPE *phixCU,*phiyCU;
	int *phix_indexCU,*phiy_indexCU;
	int N[2];
	DATATYPE thita[2];
	int m;
	int M;
	DATATYPE *ckCU;
	int *trafo_phiSparM_lCU;
     	DATATYPE *trafo_phiSparMCU;
	int *adjoint_phiSparM_LocaCU;
	int *adjoint_phiSparM_jCU;
	int *adjoint_phiSparM_indexCU;
	DATATYPE *adjoint_phiSparMCU;
	int trafo_phiSparM_para[2],adjoint_phiSparM_para[2];
	CUFFTDATATYPE *gCU,*ghatCU,*f1CU;
	cufftHandle planTmp;
	int flags;
} cunfftplan2d2;
#endif

#ifndef CUNFFTP1D
#define CUNFFTP1D
typedef struct {
	int N;
	DATATYPE thita;
	int m;
	int M;
	DATATYPE *ckCU;
	int *adjoint_phiSparM_LocaCU;
	int *adjoint_phiSparM_jCU;
	int *adjoint_phiSparM_indexCU;
	DATATYPE *adjoint_phiSparMCU;
	int *trafo_phiSparM_lCU;
     	DATATYPE *trafo_phiSparMCU;
	int trafo_phiSparM_para[2],adjoint_phiSparM_para[2];
	CUFFTDATATYPE *gCU,*ghatCU;
	cufftHandle planTmp;
} cunfftplan1d;
#endif

#ifndef GICONP
#define GICONP
typedef struct{
	CUFFTDATATYPE *reprojection;
	DATATYPE *m;
    	DATATYPE *mCU;
	CUFFTDATATYPE *fCU;
	CUFFTDATATYPE *fhatCU;
	CUFFTDATATYPE *rCU;
	CUFFTDATATYPE *rhatCU;
	CUFFTDATATYPE *AhwbCU;
	CUFFTDATATYPE *fhat1dCU;
	DATATYPE *weightCU;
	ALPHADATATYPE *alphaTmpCU;
	ALPHADATATYPE *alCU;
} GICONPara;
#endif

__global__ void nufft_fhatDivCk1d_assignGhat_1d(CUFFTDATATYPE *f ,CUFFTDATATYPE *g,DATATYPE* ck,int N,int n);

__global__ void nufft_fhatDivCk_assignGhat(CUFFTDATATYPE *f ,CUFFTDATATYPE *g,DATATYPE *ck,int NX,int NY,int nx,int ny);

__global__ void nufft_fhatDivCk_assignGhat_R2C(DATATYPE *f ,CUFFTDATATYPE *g,DATATYPE *ck,int NX,int NY,int nx,int ny);

__global__ void nufft_ck2d(DATATYPE *res ,int NX,int NY,int m, DATATYPE thita);

__global__ void nufft_ck2d2(DATATYPE *res ,int NX,int NY,int m, DATATYPE thita0,DATATYPE thita1);

void genPhi2dMark_adjoint(int *phi2dMark,int *phix_index,int *phiy_index,int m,int M,int nx,int ny);

void preGenPhi2dSparseMatrix_adjoint(int *phi2dMark,int *phi2dSparseMatrixIndex,int *phi2dSparseMatrixNum,int *phi2dSparseMatrixPara,int nx,int ny);

void preGenPhi2dSparseMatrix2_adjoint(int *phi2dSparseMatrixNum,int *phi2dSparseMatrixLocate,int phi2dSparseMatrixPara1);

void GenPhi2dSparseMatrix_adjoint(int *phi2dSparseMatrixJ,DATATYPE *phi2dSparseMatrix,DATATYPE *phix,int *phix_index,DATATYPE *phiy,int *phiy_index,int *phi2dSparseMatrixNum,int *phi2dSparseMatrixIndex,int* phi2dSparseMatrixPara,int nx,int ny,int m,int M);

void GenPhi2dSparseMatrix_trafo(int *phi2dSparseMatrixL,DATATYPE *phi2dSparseMatrix,DATATYPE *phix,int *phix_index,DATATYPE *phiy,int *phiy_index,int nx,int ny,int m,int M);

__global__ void nufft_Phi2d_1dCU(DATATYPE *phix ,int *phix_index,DATATYPE *pdx2d,int N,int m,int M,DATATYPE thita,DATATYPE b,DATATYPE bf,int xy);

void cunfft_mallocPlan_2d(cunfftplan2d *plan,int NX,int NY,DATATYPE thita,int M,int m,int flags);

void cunfft_mallocPlan_2d2(cunfftplan2d2 *plan,int NX,int NY,DATATYPE* thita,int M,int m,int flags);

void cunfft_initPlan_2d(cunfftplan2d *plan,int NX,int NY,DATATYPE thita,int M,int m,DATATYPE *pdx2d);

void cunfft_initPlan_2d2(cunfftplan2d2 *plan,int NX,int NY,DATATYPE* thita,int M,int m,DATATYPE *pdx2d);

void cunfft_destroyPlan_2d(cunfftplan2d *plan);

void cunfft_destroyPlan_2d2(cunfftplan2d2 *plan);

__global__ void nufft_fhat_div_ckCU(CUFFTDATATYPE *f ,DATATYPE *ck,int NX,int NY,DATATYPE thita);

//search x with a fix y
__global__ void nufft_GPhi_p1CU(CUFFTDATATYPE *f1,CUFFTDATATYPE *g,DATATYPE *trafo_phiSparM,int *trafo_phiSparM_l,int NX,int NY,int m,int M,DATATYPE thita);

//combine f1 into f
__global__ void nufft_GPhi_p2CU(CUFFTDATATYPE *f,CUFFTDATATYPE *f1,int m,int M);

__global__ void nufft_assignG_HAT(CUFFTDATATYPE *f ,CUFFTDATATYPE *g,int NX,int NY,int nx,int ny);

void cunfft_trafo_2d(cunfftplan2d plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat);

void cunfft_trafo_2d2(cunfftplan2d2 plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat);

void cunfft_trafo_2d_R2C(cunfftplan2d plan,CUFFTDATATYPE* f,DATATYPE* fhat);

void cunfft_trafo_2d_R2C2(cunfftplan2d2 plan,CUFFTDATATYPE* f,DATATYPE* fhat);

__global__ void nufft_adjoint_fullpre_FPhi(CUFFTDATATYPE *g,CUFFTDATATYPE* f,int *phi2dSparseMatrixLocate,int *phi2dSparseMatrixIndex,DATATYPE *phi2dSparseMatrix,int *phi2dSparseMatrixJ,int phi2dSparseMatrixPara0,int phi2dSparseMatrixPara1);

__global__ void nufft_assignF_HAT(CUFFTDATATYPE *f_hat,CUFFTDATATYPE *g_hat,DATATYPE *ck,int NX,int NY,int nx,int ny);

void cunfft_adjoint_2d(cunfftplan2d plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat);

void cunfft_adjoint_2d2(cunfftplan2d2 plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat);

void genPhi1dMark_adjoint(int *phi1dMark,int *phi_index,int m,int M,int N);

void preGenPhi1dSparseMatrix_adjoint(int *phi1dMark,int *phi1dSparseMatrixIndex,int *phi1dSparseMatrixNum,int *phi1dSparseMatrixPara,int N);

void preGenPhi1dSparseMatrix2_adjoint(int *phi1dSparseMatrixNum,int *phi1dSparseMatrixLocate,int phi1dSparseMatrixPara1);

void GenPhi1dSparseMatrix_adjoint(int *phi1dSparseMatrixJ,DATATYPE *phi1dSparseMatrix,DATATYPE *phi,int *phi_index,int *phi1dSparseMatrixNum,int *phi1dSparseMatrixIndex,int *phi1dSparseMatrixPara,int N,int m,int M);

__global__ void nufft_assignF_HAT_1d(CUFFTDATATYPE *f_hat,CUFFTDATATYPE *g_hat,DATATYPE *ck,int N,int n);

__global__ void nufft_GPhi_p1CU_noPreCompute(CUFFTDATATYPE *f1,CUFFTDATATYPE *g,DATATYPE *phix,DATATYPE *phiy,int *phix_index,int *phiy_index,int NX,int NY,int m,int M,DATATYPE thita1);

//----------------------------------------------------------------------------------------------------trafo 1d

__global__ void nufft_Phi1dCU(DATATYPE *phix ,int *phix_index,DATATYPE *pdx1d,int N,int m,int M,DATATYPE thita,DATATYPE b,DATATYPE bf);

__global__ void nufft_ck1d(DATATYPE *res ,int N,int m, DATATYPE thita);

__global__ void nufft_fhat_div_ck1dCU(CUFFTDATATYPE *f ,DATATYPE *ck1d,int N,DATATYPE thita);

void GenPhi1dSparseMatrix_trafo(int *phi1dSparseMatrixL,DATATYPE *phi1dSparseMatrix,DATATYPE *phi,int *phi_index,int n,int m,int M);

void cunfft_initPlan_1d(cunfftplan1d *plan,int N,DATATYPE thita,int M,int m,DATATYPE *pdx1d);

void cunfft_destroyPlan_1d(cunfftplan1d *plan);

__global__ void nufft_assignG_HAT_1d(CUFFTDATATYPE *f ,CUFFTDATATYPE *g,int N,int n);

__global__ void nufft_GPhi_1dCU(CUFFTDATATYPE *f,CUFFTDATATYPE *g,DATATYPE *trafo_phiSparM,int *trafo_phiSparM_l,int N,int m,int M,DATATYPE thita);

void cunfft_trafo_1d(cunfftplan1d plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat);

void cunfft_adjoint_1d(cunfftplan1d plan,CUFFTDATATYPE* f,CUFFTDATATYPE* fhat);

//--------------------------------------only for ICON
__global__ void complexMulDoubleCU(CUFFTDATATYPE* f,DATATYPE *weight,int M);
__global__ void complexCopyCU(CUFFTDATATYPE* dest,CUFFTDATATYPE* src,int size);
__global__ void complexMinusCU(CUFFTDATATYPE* r,CUFFTDATATYPE* fhat,CUFFTDATATYPE* Ahwb,int size);
__global__ void ICONUpdateCU(CUFFTDATATYPE* m,CUFFTDATATYPE* r,ALPHADATATYPE *al,int dataType,DATATYPE threshold,int size);
__global__ void ICONUpdateCU_R(DATATYPE* m,CUFFTDATATYPE* r,ALPHADATATYPE *al,int dataType,DATATYPE threshold,int size);
__global__ void INFRUpdateCU(CUFFTDATATYPE* m,CUFFTDATATYPE* r,ALPHADATATYPE *al,int size);
__global__ void INFRUpdateCU_R(DATATYPE* m,CUFFTDATATYPE* r,ALPHADATATYPE *al,int size);
__global__ void alpha0(ALPHADATATYPE *alpha0Tmp,CUFFTDATATYPE* r,int size);
__global__ void alpha1(ALPHADATATYPE *alpha0Tmp,CUFFTDATATYPE* rhat,DATATYPE *weight,int size);
__global__ void alpha2(ALPHADATATYPE *rtr,ALPHADATATYPE *rtar,ALPHADATATYPE *al);
__device__ void addShared256(ALPHADATATYPE* data, int tid,int lowBound);
__global__ void matrixAdd(ALPHADATATYPE *matrix,int size);
void initGICONPara(GICONPara *GP,int nx,int nz,DATATYPE *weight);
void initGICONPara2(GICONPara *GP,int nx,int thickness,int nz,DATATYPE *weight);
void destroyGICONPara(GICONPara *GP);





