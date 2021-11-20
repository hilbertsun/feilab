#ifndef TEXT_LINE_MAX
#define TEXT_LINE_MAX         800
#endif


#ifndef     PI
#define     PI                3.14159265358979323846
#define     PI_180            0.01745329252
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "mrcfile3.h"
#include "../log/log.h"

void help()
{
    printf("preProcess parameter\n######\n");
    printf("-input (-i) : the tilt series.\n######\n");
    printf("-tiltfile (-t) : the file containing aligned tilt angle of each projection image. If this option is not used, then only subtract the mode value of projection images.\n######\n");
    printf("-thickness (-th) : the thickness of specimen in pixel. If this option is not used, then only subtract the mode value of projection images\n######\n");
    printf("-output (-o) : preProcessed projection file.\n######\n");
    printf("-help (-h) : for help.\n");
}


double minusmode(float *in , int nx , int ny)
{
    int graylevel = 256;
    double edgeRate = 0.1;
    double min,max,step;
    int xedge = nx*edgeRate;
    int yedge = ny*edgeRate;
    int i,j,index;
    int maxindex;
    min = max = in[yedge*nx+xedge];
    for (j = yedge ; j < ny - yedge ; j++)
        for (i = xedge ; i < nx - xedge ; i++)
        {
            index = j*nx+i;
            if (in[index] < min)
                min = in[index];
            if (in[index] > max)
                max = in[index];
        }
    step = (max-min)/(double)graylevel;
    int *hist = (int *)malloc(graylevel*sizeof(int));
    memset(hist,0,graylevel*sizeof(int));
    for (j = yedge ; j < ny - yedge ; j++)
        for (i = xedge ; i < nx - xedge ; i++)
        {
            index = j*nx+i;
            hist[(int)floor((((double)in[index]-min)/step))]++;
        }
    max = -1000;
    for (i = 0 ; i < graylevel ; i++)
        if (max < hist[i])
        {
            max = hist[i];
            maxindex = i;
        }
    double mode = maxindex*step+min;
    for (i = 0 ; i < nx*ny ; i++)
        in[i] -= mode;
    free(hist);
    return mode;
}


//normalize the variance to be factor ; factor = 0.33*t/cos(angle);
void normalize(float *in,int nx,int ny,float angle,int thickness)
{
	int i,j;
	int size = nx*ny;
	double sum = 0;
	double factor = /*0.33/cos(angle/180.0*3.1415926);//*/0.33*thickness/cos(angle/180.0*3.1415926);
	for (i = 0 ; i < size ; i++)
		sum+=in[i]/size;
	double var = 0;
	for (i = 0 ; i < size ; i++)
		var+=(in[i] - sum)*(in[i] - sum)/size;
	for (i = 0 ; i < size ; i++)
		in[i] = in[i]/sqrt(var)*sqrt(factor);
}


int main(argc,argv)
int argc;
char *argv[];
{
    char infile[1000],anglefile[1000],outfile[1000];
    int radius;
    int thickness;
    int i,j,k;
    int paraNum = 4;
    int *paraMark = (int *)malloc(paraNum*sizeof(int));
    char loginfo[1000];
    time_t rawtime;
    struct tm * timeinfo;
    //log Write
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    logwrite("##############################\n");
    sprintf(loginfo,"time:\n   ");
    logwrite(loginfo);
    sprintf(loginfo,"%s",asctime(timeinfo));
    logwrite(loginfo);
    sprintf(loginfo,"cmd:\n   ");
    logwrite(loginfo);
    for (i = 0 ; i < argc ; i++)
    {
	sprintf(loginfo,"%s ",argv[i]);
	logwrite(loginfo);
    }
    sprintf(loginfo,"\n");
    logwrite(loginfo);
    //end of log Write

    // read Parameter
    i = 1;
    memset(paraMark,0,paraNum*sizeof(int));
    while (i < argc)
    {
        if (argv[i][0] == '-')
        {
            if (strcmp(argv[i]+1,"input") == 0 || strcmp(argv[i]+1,"i") == 0)
            {
                i++;
                sscanf(argv[i],"%s",infile);
                i++;
                paraMark[0] = 1;
            }
            else if (strcmp(argv[i]+1,"tiltfile") == 0 || strcmp(argv[i]+1,"t") == 0)
            {
                i++;
                sscanf(argv[i],"%s",anglefile);
                i++;
                paraMark[1] = 1;
            }
            else if (strcmp(argv[i]+1,"output") == 0 || strcmp(argv[i]+1,"o") == 0)
            {
                i++;
                sscanf(argv[i],"%s",outfile);
                i++;
                paraMark[2] = 1;
            }
	    else if (strcmp(argv[i]+1,"thickness") == 0 || strcmp(argv[i]+1,"th") == 0)
            {
                i++;
                sscanf(argv[i],"%d",&thickness);
                i++;
                paraMark[3] = 1;
            }
            else if (strcmp(argv[i]+1,"help") == 0 || strcmp(argv[i]+1,"h") == 0)
            {
                i++;
                help();
		//log Write
    		sprintf(loginfo,"running state:\n   ");
    		logwrite(loginfo);
    		sprintf(loginfo,"ICONPreProcess help finish!\n");
    		logwrite(loginfo);
    		//end of log Write
                return;
            }
            else
                i++;
        }
        else
            i++;
    }
    if (!(paraMark[0] && paraMark[2]))
    {
	printf("parameter error!\n input or output is missed.\nPlease use -help to see the manual\n");
	//log Write
    	sprintf(loginfo,"running state:\n   ");
    	logwrite(loginfo);
    	sprintf(loginfo,"fail!\n");
    	logwrite(loginfo);
	sprintf(loginfo,"Error message:\n   ");
	logwrite(loginfo);
	sprintf(loginfo,"parameter error! input or output is missed.\n");
	logwrite(loginfo);
    	//end of log Write
	return;
    }
    else if (!(paraMark[1] && paraMark[3]))
    {
	printf("tiltfile or thickness is not specified. Normalization will not be executed\n");
    }
    {
        printf("parameter:\n");
        printf("input : %s\n",infile);
	if (paraMark[1])
        	printf("tiltfile : %s\n",anglefile);
	if (paraMark[3])
        	printf("thickness : %d\n",thickness);
        printf("output : %s\n",outfile);
    }
    // end of read Parameter
    float *indata;

    int i1,j1,k1;
    MrcHeader  *inhead, *outhead;
    FILE *fin, *fout;
    if((fin=fopen(infile,"r"))==NULL)
    {
        printf("\nCannot open file '%s' strike any key exit!",infile);
	//log Write
    	sprintf(loginfo,"running state:\n   ");
    	logwrite(loginfo);
    	sprintf(loginfo,"fail!\n");
    	logwrite(loginfo);
	sprintf(loginfo,"Error message:\n   ");
	logwrite(loginfo);
	sprintf(loginfo,"Cannot open file '%s\n",infile);
	logwrite(loginfo);
    	//end of log Write
        exit(1);
    }

    if((fout=fopen(outfile,"w+"))==NULL)
    {
        printf("\nCannot open file '%s' strike any key exit!",outfile);
	//log Write
    	sprintf(loginfo,"running state:\n   ");
    	logwrite(loginfo);
    	sprintf(loginfo,"fail!\n");
    	logwrite(loginfo);
	sprintf(loginfo,"Error message:\n   ");
	logwrite(loginfo);
	sprintf(loginfo,"Cannot open file '%s\n",outfile);
	logwrite(loginfo);
    	//end of log Write
        exit(1);
    }


    inhead=(MrcHeader *)malloc(sizeof(MrcHeader));
    outhead=(MrcHeader *)malloc(sizeof(MrcHeader));

    mrc_read_head(fin,inhead);
    memcpy(outhead,inhead,sizeof(MrcHeader));
    //mrc_init_head(outhead);
    outhead->nx = inhead->nx;
    outhead->nz = inhead->nz;
    outhead->ny = inhead->ny;
    outhead->mode = MRC_MODE_FLOAT;
    mrc_write_head(fout,outhead);
    int nx = inhead->nx;
    int ny = inhead->ny;
    int nz = inhead->nz;

    // MinusMode slice by slice
    indata = (float*)malloc(nx*ny*nz*sizeof(float));
    for (k = 0 ; k < nz ; k++)
    {
        mrc_read_slice(fin, inhead, k, 'z',indata+k*nx*ny);
	minusmode(indata+k*nx*ny,nx,ny);
        //printf("slice %d mode %lf\n",k,minusmode(indata+k*nx*ny,nx,ny));
    }
    if (!(paraMark[1] && paraMark[3]))
	for (k = 0 ; k < nz ; k++)
	{
		mrc_add_slice(fout,outhead,indata+k*nx*ny);
	}
    //end of MinusMode slice by slice
    //Normalize 
    if (paraMark[1] && paraMark[3])
    {
	FILE *fang;
	if((fang=fopen(anglefile,"r"))==NULL)
   	{
        	printf("\nCannot open file '%s' strike any key exit!",anglefile);
		//log Write
    		sprintf(loginfo,"running state:\n   ");
    		logwrite(loginfo);
    		sprintf(loginfo,"fail!\n");
    		logwrite(loginfo);
		sprintf(loginfo,"Error message:\n   ");
		logwrite(loginfo);
		sprintf(loginfo,"Cannot open file '%s\n",anglefile);
		logwrite(loginfo);
    		//end of log Write
        	exit(1);
    	}
	int Ang_Num = 0;
	float thita[180];
    	while (fscanf(fang,"%f",&(thita[Ang_Num]))!=EOF)
   	{
        	thita[Ang_Num] *=-1;
        	Ang_Num++;
    	}
    	fclose(fang);
	for (k = 0 ; k < nz ; k++)
	{
		normalize(indata+k*nx*ny,nx,ny,thita[k],thickness);
		mrc_add_slice(fout,outhead,indata+k*nx*ny);
	}
    }	
    //end of Normalize

    free(inhead);
    free(outhead);
    free(indata);
    fclose(fout);
    mrc_update_head(outfile);

    //log Write
    sprintf(loginfo,"running state:\n   ");
    logwrite(loginfo);
    sprintf(loginfo,"ICONPreProcess finish!\n");
    logwrite(loginfo);
    //end of log Write

    return 0;
}




