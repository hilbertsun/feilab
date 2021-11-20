#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "px2D.cuh"

int TFunction2(DATATYPE *x,float angle,int SIZE)
{
    DATATYPE x1=-1.0/2;
    DATATYPE step = 1.0/SIZE;
    DATATYPE *f = (DATATYPE*)malloc(SIZE*sizeof(DATATYPE));
    int i;
    f[0]=x1;
    for(i=1; i<SIZE; i++)
    {
        f[i]=f[i-1]+step;
    }
    for(i=0; i<SIZE; i++)
    {
        x[2*i]=f[i]*cos(angle*pi/180);
        x[2*i+1]=f[i]*sin(angle*pi/180);
    }
    free(f);
    return 0;
}


void px2D(DATATYPE *bpx,MrcHeader *inhead,float *thita)
{
    int i;

    DATATYPE *px1D;
    int SIZE = inhead->nx;
    px1D = (DATATYPE*)malloc(2*(inhead->nx)*sizeof(DATATYPE));
    for (i = 0 ; i < inhead->nz ; i++)
    {
        TFunction2(px1D,thita[i],inhead->nx);
        memcpy(bpx+i*SIZE*2,px1D,2*SIZE*sizeof(DATATYPE));

    }
    /*FILE *f = fopen("px2d","w+");
    for (i = 0 ; i < inhead->nx*inhead->nz; i++)
	fprintf(f,"%f %f\n",bpx[2*i],bpx[2*i+1]);
    fclose(f);*/
    free(px1D);
}

