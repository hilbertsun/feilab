#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "weight.cuh"

void Weight(DATATYPE* w,MrcHeader *inhead,float* thita)
{
    int i,j;
    int SIZE = inhead->nx;
    int center = ceil(SIZE/2);
    for(i = 0; i < inhead->nz ; i++)
    {
        if (i == 0)
            for(j = 0; j < SIZE ; j++)
            {
                if (j-center != 0)
                    w[i*SIZE+j] = 2.0*fabs((thita[i+1] - thita[i])*(j-center));
                else
                    w[i*SIZE+j]=(180.0/(4*inhead->nz));
            }
        if (i == inhead->nz-1)
            for(j = 0; j < SIZE ; j++)
            {
                if (j-center != 0)
                    w[i*SIZE+j] = 2.0*fabs((thita[i-1] - thita[i])*(j-center));
                else
                    w[i*SIZE+j]=(180.0/(4*inhead->nz));
            }
        if (i!=0&&i!=inhead->nz-1)
            for(j = 0; j < SIZE ; j++)
            {
                if(j!=center)
                {
                    w[i*SIZE+j]=1.0*fabs((thita[i-1] - thita[i+1])*(j-center));
                }
                else
                    w[i*SIZE+j]=(180.0/(4*inhead->nz));
            }
    }

	/*for(i = 0; i < inhead->nz ; i++)
		for(j = 0; j < SIZE ; j++)
			w[i*SIZE+j] =  w[i*SIZE+j]* w[i*SIZE+j]/1000; */

//    FILE *f = fopen("weight.txt","w+");
//    for ( i = 0 ; i < inhead->nz ; i++)
//    {
//        for(j=0; j<SIZE; j++)
//            fprintf(f,"%f ",w[i*SIZE+j]);
//        fprintf(f,"\n\n");
//    }
//    fclose(f);
}

