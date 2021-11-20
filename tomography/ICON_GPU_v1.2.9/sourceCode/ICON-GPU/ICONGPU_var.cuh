#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "mrcfile3.h"
#include "nufft_gpu_v8.cuh"
#include "../log/log.h"

#ifndef GICONP
#define GICONP
typedef struct{
	CUFFTDATATYPE *reprojection;
	CUFFTDATATYPE *m;
    	CUFFTDATATYPE *mCU;
	CUFFTDATATYPE *fCU;
	CUFFTDATATYPE *fhatCU;
	CUFFTDATATYPE *rCU;
	CUFFTDATATYPE *rhatCU;
	CUFFTDATATYPE *AhwbCU;
	CUFFTDATATYPE *fhat1dCU;
	DATATYPE *weightCU;
	ALPHADATATYPE *(alphaTmpCU[2]);
	ALPHADATATYPE *alCU;
} GICONPara;
#endif
