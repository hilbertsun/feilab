#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "ICONGPU_var.cuh"
#include "mrcfile3.h"

#ifdef HAVE_COMPLEX_H
#include <complex.h>
#endif

#include "cufft.h"


void saveMRC(MrcHeader *inhead,FILE *outfile,MrcHeader *outhead,CUFFTDATATYPE *slice);
