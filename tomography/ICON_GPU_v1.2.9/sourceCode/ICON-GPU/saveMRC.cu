#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "saveMRC.cuh"


void saveMRC(MrcHeader *inhead,FILE *outfile,MrcHeader *outhead,CUFFTDATATYPE *slice)
{
    int i,j;
    int nx = inhead->nx;
    float *sliceTmp = (float *)malloc(sizeof(float)*nx*nx);
    memcpy(outhead,inhead,sizeof(MrcHeader));
    //mrc_init_head(outhead);
        //set apixel nx=ny=nz
    outhead->mz = inhead->mx;
    outhead->zlen = inhead->xlen;
    outhead->nx = nx;
    outhead->ny = nx;
    outhead->nz = 1;
    outhead->mode = MRC_MODE_FLOAT;
    mrc_write_head(outfile,outhead);


    //cut slice
    for(j = 0 ; j < nx; j++)
        for (i = 0 ; i < nx ; i++)
        {
            sliceTmp[j*nx+i] = slice[i*nx+j].x;
        }
    mrc_add_slice(outfile,outhead,sliceTmp);
    free(sliceTmp);
}
