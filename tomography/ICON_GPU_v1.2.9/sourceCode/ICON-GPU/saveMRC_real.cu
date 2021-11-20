#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "saveMRC_real.cuh"


void saveMRC_real(MrcHeader *inhead,FILE *outfile,MrcHeader *outhead,DATATYPE *slice,int nx,int ny)
{
    int i;
    memcpy(outhead,inhead,sizeof(MrcHeader));
    //mrc_init_head(outhead);
        //set apixel nx=ny=nz
    outhead->mz = inhead->mx;
    outhead->zlen = inhead->xlen;
    outhead->nx = nx;
    outhead->ny = ny;
    outhead->nz = 1;
    outhead->mode = MRC_MODE_FLOAT;
    mrc_write_head(outfile,outhead);

    float *slice1 = (float *)malloc(nx*ny*sizeof(float));
    for (i = 0 ; i < nx*ny ; i++)
	{
	slice1[i] = (float)slice[i];
	}
    //cut slice
    mrc_add_slice(outfile,outhead,slice1);
    free(slice1);
}

void saveMRC_real_inv(MrcHeader *inhead,FILE *outfile,MrcHeader *outhead,DATATYPE *slice,int nx,int ny)
{
    int i,j;
    memcpy(outhead,inhead,sizeof(MrcHeader));
    //mrc_init_head(outhead);
        //set apixel nx=ny=nz
    outhead->mz = inhead->mx;
    outhead->zlen = inhead->xlen;
    outhead->nx = nx;
    outhead->ny = ny;
    outhead->nz = 1;
    outhead->mode = MRC_MODE_FLOAT;
    mrc_write_head(outfile,outhead);
    float *sliceTmp = (float *)malloc(sizeof(float)*nx*ny);

    //cut slice
    for(j = 0 ; j < ny; j++)
        for (i = 0 ; i < nx ; i++)
        {
            sliceTmp[j*nx+i] = slice[i*ny+j];
        }
    mrc_add_slice(outfile,outhead,sliceTmp);
    free(sliceTmp);
}
