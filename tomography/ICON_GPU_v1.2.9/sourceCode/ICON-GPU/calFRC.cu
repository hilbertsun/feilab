#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "calFRC.cuh"


int calFRC(DATATYPE *image1,DATATYPE* image2,DATATYPE* frc,int nx)
{
	int i,j,i1,j1,r;
	CUFFTDATATYPE* image1F = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*nx);
	CUFFTDATATYPE* image2F = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*nx);
	CUFFTDATATYPE* image1ft = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*nx);
	CUFFTDATATYPE* image2ft = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*nx);
	CUFFTDATATYPE* image1FCU;
	CUFFTDATATYPE* image2FCU;
	CUFFTDATATYPE* image1ftCU;
	CUFFTDATATYPE* image2ftCU;
	cudaMalloc((void**)&(image1FCU),sizeof(CUFFTDATATYPE)*nx*nx);
	cudaMalloc((void**)&(image2FCU),sizeof(CUFFTDATATYPE)*nx*nx);
	cudaMalloc((void**)&(image1ftCU),sizeof(CUFFTDATATYPE)*nx*nx);
	cudaMalloc((void**)&(image2ftCU),sizeof(CUFFTDATATYPE)*nx*nx);
	for (i = 0 ; i < nx*nx ; i++)
	{
		image1F[i].x = image1[i];
		image2F[i].x = image2[i];
		image1F[i].y = image2F[i].y = 0;
	}
	cudaMemcpy(image1FCU,image1F,sizeof(CUFFTDATATYPE)*nx*nx, cudaMemcpyHostToDevice);
	cudaMemcpy(image2FCU,image2F,sizeof(CUFFTDATATYPE)*nx*nx, cudaMemcpyHostToDevice);
	
	cufftHandle plan;
	cufftPlan2d(&plan,nx,nx,CUFFTTYPE);
	CUFFTEXEC(plan,image1FCU,image1ftCU,CUFFT_FORWARD);
     	CUFFTEXEC(plan,image2FCU,image2ftCU,CUFFT_FORWARD);
  	cufftDestroy(plan);
	
	cudaMemcpy(image1ft,image1ftCU,sizeof(CUFFTDATATYPE)*nx*nx, cudaMemcpyDeviceToHost);
	cudaMemcpy(image2ft,image2ftCU,sizeof(CUFFTDATATYPE)*nx*nx, cudaMemcpyDeviceToHost);
	
	
	double* upper =  (double*)malloc(sizeof(double)*(nx/2+1));
	double* lower1 = (double*)malloc(sizeof(double)*(nx/2+1));
	double* lower2 = (double*)malloc(sizeof(double)*(nx/2+1));
	int nx_2 = nx/2;
	int index;
	double tmp2;
	memset(lower1,0,sizeof(double)*(nx/2+1));
	memset(lower2,0,sizeof(double)*(nx/2+1));
	memset(upper,0,sizeof(double)*(nx/2+1));
	for (j = 0 ; j < nx ; j++)
		for (i = 0 ; i < nx ;i++)
		{
			     i1 = (i + nx_2)%nx;
                	     j1 = (j + nx_2)%nx;
			     r = floor(sqrt((i1-nx_2)*(i1-nx_2)+(j1-nx_2)*(j1-nx_2)));
			     if (r < nx/2)
			     {
				/*if (r == 240)
					printf("%d\n",r);*/
			     	index = j*nx+i;
				upper[r] += image1ft[index].x*image2ft[index].x+image1ft[index].y*image2ft[index].y;
			     	lower1[r] += image1ft[index].x*image1ft[index].x+image1ft[index].y*image1ft[index].y;
			     	lower2[r] += image2ft[index].x*image2ft[index].x+image2ft[index].y*image2ft[index].y;
			     }
		}
	
	frc[0] = 1;
	for (i = 1 ; i < nx_2 ; i++)
	{
		tmp2 = sqrt(lower1[i])*sqrt(lower2[i]);
		if (tmp2 != 0)
			frc[i] = upper[i]/tmp2;
		else
			frc[i] = 0;
	}
	free(image1F);
	free(image2F);
	free(image1ft);
	free(image2ft);
	cudaFree(image1FCU);
	cudaFree(image2FCU);
	cudaFree(image1ftCU);
	cudaFree(image2ftCU);
	free(upper);
	free(lower1);
	free(lower2);
	printf("calFRC finish!\n");
	return 0;
}

//only for nx > ny
int calFRC_nonsqure(DATATYPE *image1,DATATYPE* image2,DATATYPE* frc,int nx,int ny)
{
	int i,j,i1,j1,r;
	CUFFTDATATYPE* image1F = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*ny);
	CUFFTDATATYPE* image2F = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*ny);
	CUFFTDATATYPE* image1ft = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*ny);
	CUFFTDATATYPE* image2ft = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*ny);
	CUFFTDATATYPE* image1FCU;
	CUFFTDATATYPE* image2FCU;
	CUFFTDATATYPE* image1ftCU;
	CUFFTDATATYPE* image2ftCU;
	cudaMalloc((void**)&(image1FCU),sizeof(CUFFTDATATYPE)*nx*ny);
	cudaMalloc((void**)&(image2FCU),sizeof(CUFFTDATATYPE)*nx*ny);
	cudaMalloc((void**)&(image1ftCU),sizeof(CUFFTDATATYPE)*nx*ny);
	cudaMalloc((void**)&(image2ftCU),sizeof(CUFFTDATATYPE)*nx*ny);
	for (i = 0 ; i < nx*ny ; i++)
	{
		image1F[i].x = image1[i];
		image2F[i].x = image2[i];
		image1F[i].y = image2F[i].y = 0;
	}
	cudaMemcpy(image1FCU,image1F,sizeof(CUFFTDATATYPE)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(image2FCU,image2F,sizeof(CUFFTDATATYPE)*nx*ny, cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan2d(&plan,ny,nx,CUFFTTYPE);
	CUFFTEXEC(plan,image1FCU,image1ftCU,CUFFT_FORWARD);
     	CUFFTEXEC(plan,image2FCU,image2ftCU,CUFFT_FORWARD);
  	cufftDestroy(plan);

	cudaMemcpy(image1ft,image1ftCU,sizeof(CUFFTDATATYPE)*nx*ny, cudaMemcpyDeviceToHost);
	cudaMemcpy(image2ft,image2ftCU,sizeof(CUFFTDATATYPE)*nx*ny, cudaMemcpyDeviceToHost);


	double* upper =  (double*)malloc(sizeof(double)*(nx/2+1));
	double* lower1 = (double*)malloc(sizeof(double)*(nx/2+1));
	double* lower2 = (double*)malloc(sizeof(double)*(nx/2+1));
	int nx_2 = nx/2;
        int ny_2 = ny/2;
	int index;
	double tmp2;
	memset(lower1,0,sizeof(double)*(nx/2+1));
	memset(lower2,0,sizeof(double)*(nx/2+1));
	memset(upper,0,sizeof(double)*(nx/2+1));
	float dampFactor2 = ((float)nx/(float)ny)*((float)nx/(float)ny);
	for (j = 0 ; j < ny ; j++)
		for (i = 0 ; i < nx ;i++)
		{
			     i1 = (i + nx_2)%nx;
                	     j1 = (j + ny_2)%ny;
			     r = floor(sqrt((i1-nx_2)*(i1-nx_2)+(j1-ny_2)*(j1-ny_2)*dampFactor2));
			     if (r < nx/2)
			     {
			     	index = j*nx+i;
				upper[r] += image1ft[index].x*image2ft[index].x+image1ft[index].y*image2ft[index].y;
			     	lower1[r] += image1ft[index].x*image1ft[index].x+image1ft[index].y*image1ft[index].y;
			     	lower2[r] += image2ft[index].x*image2ft[index].x+image2ft[index].y*image2ft[index].y;
			     }
		}

	frc[0] = 1;
	for (i = 1 ; i < nx_2 ; i++)
	{
		tmp2 = sqrt(lower1[i])*sqrt(lower2[i]);
		if (tmp2 != 0)
			frc[i] = upper[i]/tmp2;
		else
			frc[i] = 0;
	}
	free(image1F);
	free(image2F);
	free(image1ft);
	free(image2ft);
	cudaFree(image1FCU);
	cudaFree(image2FCU);
	cudaFree(image1ftCU);
	cudaFree(image2ftCU);
	free(upper);
	free(lower1);
	free(lower2);
	return 0;
}

//for nx > ny , zeros pad ny == nx
//for nx < ny , clip ny == nx
int calFRC_nonsqure_padOrClip(DATATYPE *image1_small,DATATYPE* image2_small,DATATYPE* frc,int nx,int ny)
{
	int i,j,i1,j1,r;
	DATATYPE *image1 = (DATATYPE *)malloc(nx*nx*sizeof(DATATYPE));
        DATATYPE *image2 = (DATATYPE *)malloc(nx*nx*sizeof(DATATYPE));
	
	memset(image1,0,nx*nx*sizeof(DATATYPE));	
	memset(image2,0,nx*nx*sizeof(DATATYPE));

        if (nx >= ny){
		for (j1 = 0,j=nx/2-ny/2;j1<ny;j1++,j++)
			for (i = 0 ; i < nx ; i++)
			{	
				image1[j*nx+i] = image1_small[j1*nx+i];
				image2[j*nx+i] = image2_small[j1*nx+i];
			}
	}
        if (nx < ny){
        	for (j1 = 0,j=ny/2 - nx/2;j1 < nx;j1++,j++)
			for (i = 0 ; i < nx ; i++)
			{	
				image1[j1*nx+i] = image1_small[j*nx+i];
				image2[j1*nx+i] = image2_small[j*nx+i];
			}
        }
       
	ny = nx;
	
	CUFFTDATATYPE* image1F = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*ny);
	CUFFTDATATYPE* image2F = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*ny);
	CUFFTDATATYPE* image1ft = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*ny);
	CUFFTDATATYPE* image2ft = (CUFFTDATATYPE*)malloc(sizeof(CUFFTDATATYPE)*nx*ny);
	CUFFTDATATYPE* image1FCU;
	CUFFTDATATYPE* image2FCU;
	CUFFTDATATYPE* image1ftCU;
	CUFFTDATATYPE* image2ftCU;
	cudaMalloc((void**)&(image1FCU),sizeof(CUFFTDATATYPE)*nx*ny);
	cudaMalloc((void**)&(image2FCU),sizeof(CUFFTDATATYPE)*nx*ny);
	cudaMalloc((void**)&(image1ftCU),sizeof(CUFFTDATATYPE)*nx*ny);
	cudaMalloc((void**)&(image2ftCU),sizeof(CUFFTDATATYPE)*nx*ny);
	for (i = 0 ; i < nx*ny ; i++)
	{
		image1F[i].x = image1[i];
		image2F[i].x = image2[i];
		image1F[i].y = image2F[i].y = 0;
	}
	cudaMemcpy(image1FCU,image1F,sizeof(CUFFTDATATYPE)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(image2FCU,image2F,sizeof(CUFFTDATATYPE)*nx*ny, cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan2d(&plan,ny,nx,CUFFTTYPE);
	CUFFTEXEC(plan,image1FCU,image1ftCU,CUFFT_FORWARD);
     	CUFFTEXEC(plan,image2FCU,image2ftCU,CUFFT_FORWARD);
  	cufftDestroy(plan);

	cudaMemcpy(image1ft,image1ftCU,sizeof(CUFFTDATATYPE)*nx*ny, cudaMemcpyDeviceToHost);
	cudaMemcpy(image2ft,image2ftCU,sizeof(CUFFTDATATYPE)*nx*ny, cudaMemcpyDeviceToHost);


	double* upper =  (double*)malloc(sizeof(double)*(nx/2+1));
	double* lower1 = (double*)malloc(sizeof(double)*(nx/2+1));
	double* lower2 = (double*)malloc(sizeof(double)*(nx/2+1));
	int nx_2 = nx/2;
        int ny_2 = ny/2;
	int index;
	double tmp2;
	memset(lower1,0,sizeof(double)*(nx/2+1));
	memset(lower2,0,sizeof(double)*(nx/2+1));
	memset(upper,0,sizeof(double)*(nx/2+1));
	float dampFactor2 = ((float)nx/(float)ny)*((float)nx/(float)ny);
	for (j = 0 ; j < ny ; j++)
		for (i = 0 ; i < nx ;i++)
		{
			     i1 = (i + nx_2)%nx;
                	     j1 = (j + ny_2)%ny;
			     r = floor(sqrt((i1-nx_2)*(i1-nx_2)+(j1-ny_2)*(j1-ny_2)*dampFactor2));
			     if (r < nx/2)
			     {
			     	index = j*nx+i;
				upper[r] += image1ft[index].x*image2ft[index].x+image1ft[index].y*image2ft[index].y;
			     	lower1[r] += image1ft[index].x*image1ft[index].x+image1ft[index].y*image1ft[index].y;
			     	lower2[r] += image2ft[index].x*image2ft[index].x+image2ft[index].y*image2ft[index].y;
			     }
		}

	frc[0] = 1;
	for (i = 1 ; i < nx_2 ; i++)
	{
		tmp2 = sqrt(lower1[i])*sqrt(lower2[i]);
		if (tmp2 != 0)
			frc[i] = upper[i]/tmp2;
		else
			frc[i] = 0;
	}
	free(image1F);
	free(image2F);
	free(image1ft);
	free(image2ft);
	cudaFree(image1FCU);
	cudaFree(image2FCU);
	cudaFree(image1ftCU);
	cudaFree(image2ftCU);
	free(upper);
	free(lower1);
	free(lower2);
	free(image1);
	free(image2);
	return 0;
}




