#include "readSlicePart.cuh"
#include "Iteration2.cuh"
#include "ICONGPU_var.cuh"
#include "weight.cuh"
#include "px2D.cuh"
#include "saveMRC.cuh"
#include "saveMRC_real.cuh"
#include "reProject_NFFT.cuh"

int Lib_2D2(int sliceBegin,int sliceEnd,int iternum,MrcHeader* inhead,DATATYPE *reprojection,FILE *infile,char *anglefilename,char* resultpath,int *before,float *betain);
