#include "ICONGPU_var.cuh"
#include "px2D.cuh"

void reProject_NFFT(cunfftplan1d plan1d,cunfftplan2d plan2d,GICONPara GP,DATATYPE *reprojection);

void reProject_NFFT2(cunfftplan1d plan1d,cunfftplan2d2 plan2d,GICONPara GP,DATATYPE *reprojection);
