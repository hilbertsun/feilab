#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ICONGPU_var.cuh"



int calFRC(DATATYPE *image1,DATATYPE* image2,DATATYPE* frc,int nx);
int calFRC_nonsqure_padOrClip(DATATYPE *image1_small,DATATYPE* image2_small,DATATYPE* frc,int nx,int ny);
int calFRC_nonsqure(DATATYPE *image1,DATATYPE* image2,DATATYPE* frc,int nx,int ny);
