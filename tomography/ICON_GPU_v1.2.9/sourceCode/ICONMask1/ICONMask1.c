#ifndef TEXT_LINE_MAX
#define TEXT_LINE_MAX         800
#endif


#ifndef     PI
#define     PI                3.14159265358979323846
#define     PI_180            0.01745329252
#endif

#define SIZELIMIT 1025

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "fftw3.h"
#include "mrcfile3.h"
#include "../log/log.h"

void help()
{
    printf("ICONMask parameter\n######\n");
    printf("-inputPath (-i) : the folder that contains all 2D reconstructed slices (named midxxxxx.mrc), normally corresponding to the reconstruction folder generated by ICON.\n######\n");
    printf("-tiltfile (-t) : the aligned tilt file.\n######\n");
    printf("-output (-o) : the masked 3D reconstruction.\n######\n");
    printf("-slice (-s) : the reconstructed slices for combination including 2 parts split by ','. For example, 0,511 means that combining 512 slices ranging from slice 0 (mid00000.mrc) to slice 511 (mid00511.mrc).\n######\n");
    printf("-thikness (-th) : the thickness of the final masked 3D reconstruction in pixel.\n######\n");
    printf("-radius (-r) : the mask radius (in pixel) used in the Fourier domain of the combined 3D reconstruction. If this option is used, 'crossVfrc' and 'fullRecfrc' are not used.\n######\n");

    printf("-crossVfrc (-cf) : the FRC curve from the cross validation process. If 'radius' is used, this option is not used.\n######\n");
    printf("-fullRecfrc (-ff) : the FRC file from the full reconstruction process. If 'radius' is used, this option is not used.\n######\n");
    printf("-help (-h) : for help.\n");
}

void fftshift2d(fftwf_complex * a , int nx , int ny)
{
    int i,j,i1,j1;
    fftwf_complex tmp;
    for (j = 0 ; j < ny ; j++)
        for (i = 0 ; i < nx/2 ; i++)
        {
            i1 = (i + nx/2)%nx;
            j1 = (j + ny/2)%ny;
            tmp[0] = a[j*nx+i][0];
            tmp[1] = a[j*nx+i][1];
            a[j*nx+i][0] = a[j1*nx+i1][0];
            a[j*nx+i][1] = a[j1*nx+i1][1];
            a[j1*nx+i1][0] = tmp[0];
            a[j1*nx+i1][1] = tmp[1];
        }
}


int main(argc,argv)
int argc;
char *argv[];
{
    char infilePath[1000],infile[1000],anglefile[1000],outfile[1000],crossVfrcfile[1000],fullRecfrcfile[1000];
    int radius,radius05,radius03,sliceBegin,sliceEnd,thickness;
    int i,j,k;

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
    int paraNum = 8;
    int *paraMark = (int *)malloc(paraNum*sizeof(int));
    i = 1;
    memset(paraMark,0,paraNum*sizeof(int));
    while (i < argc)
    {
        if (argv[i][0] == '-')
        {
            if (strcmp(argv[i]+1,"inputPath") == 0 || strcmp(argv[i]+1,"i") == 0)
            {
                i++;
                sscanf(argv[i],"%s",infilePath);
                i++;
                paraMark[0] = 1;
            }
            else if (strcmp(argv[i]+1,"slice") == 0 || strcmp(argv[i]+1,"s") == 0)
            {
                i++;
                sscanf(argv[i],"%d,%d",&sliceBegin,&sliceEnd);
                i++;
                paraMark[1] = 1;
            }
            else if (strcmp(argv[i]+1,"thickness") == 0 || strcmp(argv[i]+1,"th") == 0)
            {
                i++;
                sscanf(argv[i],"%d",&thickness);
                i++;
                paraMark[2] = 1;
            }
            else if (strcmp(argv[i]+1,"radius") == 0 || strcmp(argv[i]+1,"r") == 0)
            {
                i++;
                sscanf(argv[i],"%d",&radius);
                i++;
                paraMark[3] = 1;
            }
            else if (strcmp(argv[i]+1,"tiltfile") == 0 || strcmp(argv[i]+1,"t") == 0)
            {
                i++;
                sscanf(argv[i],"%s",anglefile);
                i++;
                paraMark[4] = 1;
            }
            else if (strcmp(argv[i]+1,"output") == 0 || strcmp(argv[i]+1,"o") == 0)
            {
                i++;
                sscanf(argv[i],"%s",outfile);
                i++;
                paraMark[5] = 1;
            }
            else if (strcmp(argv[i]+1,"crossVfrc") == 0 || strcmp(argv[i]+1,"cf") == 0)
            {
                i++;
                sscanf(argv[i],"%s",crossVfrcfile);
                i++;
                paraMark[6] = 1;
            }
            else if (strcmp(argv[i]+1,"fullRecfrc") == 0 || strcmp(argv[i]+1,"ff") == 0)
            {
                i++;
                sscanf(argv[i],"%s",fullRecfrcfile);
                i++;
                paraMark[7] = 1;
            }
            else if (strcmp(argv[i]+1,"help") == 0 || strcmp(argv[i]+1,"h") == 0)
            {
                i++;
                help();
		//log Write
    		sprintf(loginfo,"running state:\n   ");
    		logwrite(loginfo);
    		sprintf(loginfo,"ICONMask1 help finish!\n");
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
    if (!(paraMark[0] && paraMark[1] && paraMark[2] && paraMark[4] && paraMark[5]))
    {
        printf("parameter error!\n Please use -help to see the manual\n");
	//log Write
    	sprintf(loginfo,"running state:\n   ");
    	logwrite(loginfo);
    	sprintf(loginfo,"fail!\n");
    	logwrite(loginfo);
	sprintf(loginfo,"Error message:\n   ");
	logwrite(loginfo);
	sprintf(loginfo,"parameter error!\n");
	logwrite(loginfo);
    	//end of log Write
        return;
    }
    else if (!paraMark[3] && !(paraMark[6] && paraMark[7]))
    {
        printf("parameter error!\n radius or (crossVfrc & fullRecfrc) is needed for mask. Please use -help to see the manual\n");
	//log Write
    	sprintf(loginfo,"running state:\n   ");
    	logwrite(loginfo);
    	sprintf(loginfo,"fail!\n");
    	logwrite(loginfo);
	sprintf(loginfo,"Error message:\n   ");
	logwrite(loginfo);
	sprintf(loginfo,"parameter error! radius or (crossVfrc & fullRecfrc) is needed for mask.\n");
	logwrite(loginfo);
    	//end of log Write
        return;
    }
    {
        printf("parameter:\n");
        printf("inputPath : %s\n",infilePath);
        printf("slice : %d,%d\n",sliceBegin,sliceEnd);
        printf("thickness : %d\n",thickness);
        if (paraMark[3])
            printf("radius : %d\n",radius);
        printf("tiltfile : %s\n",anglefile);
        printf("output : %s\n",outfile);
        if (!paraMark[3])
        {
            printf("crossVfrc : %s\n",crossVfrcfile);
            printf("fullRecfrc : %s\n",fullRecfrcfile);
        }
    }
    //free(paraMark);
    //----------------------------------------------------------end of read Parameter
    

    //calculate the radius of mask
    if (!paraMark[3])
    {
        FILE *fcrossVfrc,*ffullRecfrc;
        if((fcrossVfrc=fopen(crossVfrcfile,"r"))==NULL)
        {
            printf("\nCannot open file '%s' strike any key exit!",crossVfrcfile);
	    //log Write
    	    sprintf(loginfo,"running state:\n   ");
    	    logwrite(loginfo);
    	    sprintf(loginfo,"fail!\n");
    	    logwrite(loginfo);
	    sprintf(loginfo,"Error message:\n   ");
	    logwrite(loginfo);
	    sprintf(loginfo,"Cannot open file '%s'\n",crossVfrcfile);
	    logwrite(loginfo);
    	    //end of log Write
            exit(1);
        }
        if((ffullRecfrc=fopen(fullRecfrcfile,"r"))==NULL)
        {
            printf("\nCannot open file '%s' strike any key exit!",fullRecfrcfile);
	    //log Write
    	    sprintf(loginfo,"running state:\n   ");
    	    logwrite(loginfo);
    	    sprintf(loginfo,"fail!\n");
    	    logwrite(loginfo);
	    sprintf(loginfo,"Error message:\n   ");
	    logwrite(loginfo);
	    sprintf(loginfo,"Cannot open file '%s'\n",fullRecfrcfile);
	    logwrite(loginfo);
    	    //end of log Write
            exit(1);
        }
        int frcsize = 0;
        float tmp;
        float crossVfrc[10000],fullRecfrc[10000];
        while (fscanf(fcrossVfrc,"%f %f",&tmp,&(crossVfrc[frcsize]))!=EOF)
        {
            frcsize++;
        }
        fclose(fcrossVfrc);
        frcsize = 0;
        while (fscanf(ffullRecfrc,"%f %f",&tmp,&(fullRecfrc[frcsize]))!=EOF)
        {
            frcsize++;
        }
        fclose(ffullRecfrc);
        for (i = 0 ; i < frcsize ; i++)
            crossVfrc[i] /= fullRecfrc[i];
        //searching from the back
        for (i = frcsize-1 ; i >= 0 ; i--)
            if (crossVfrc[i] >= 0.5)
                break;
        radius = i;
        //searching from the back
        for (i = frcsize-1 ; i >= 0 ; i--)
            if (crossVfrc[i] >= 0.3)
                break;
        radius03 = i;
        //searching from the back
        for (i = frcsize-1 ; i >= 0 ; i--)
            if (crossVfrc[i] >= 0.5)
                break;
        radius05 = i;
        printf("the calculated radius of frc0.5 is %d\nthe calculated radius of frc0.3 is %d\n",radius05,radius03);
    }
    //--------------------------end of calculate the radius of mask

    // MRCCatch
    float *indataD;
    float *indatatmp;
    MrcHeader * inhead=(MrcHeader *)malloc(sizeof(MrcHeader));
    {
        long ii,j,j1;
        FILE * infile;
        char filename[1000];
        for (i = sliceBegin ; i <= sliceEnd; i++)
        {
            sprintf(filename,"%s/mid%05d.mrc",infilePath,i);
            if ((infile = fopen(filename,"r")) == NULL)
            {
                printf("\nCan not open infile %s!\n",filename);
		//log Write
    	    	sprintf(loginfo,"running state:\n   ");
    	    	logwrite(loginfo);
    	    	sprintf(loginfo,"fail!\n");
    	    	logwrite(loginfo);
	    	sprintf(loginfo,"Error message:\n   ");
	    	logwrite(loginfo);
	    	sprintf(loginfo,"Cannot open infile '%s'\n",filename);
	    	logwrite(loginfo);
    	    	//end of log Write
                exit(1);
            }
            mrc_read_head(infile,inhead);
            if (i == sliceBegin)
            {
                indataD = (float *)malloc((long)inhead->nx*(long)thickness*(long)(sliceEnd-sliceBegin+1)*sizeof(float));
                indatatmp = (float *)malloc((long)inhead->nx*(long)inhead->ny*sizeof(float));
            }
            mrc_read_slice(infile,inhead,0,'z',indatatmp);
            long ystart = inhead->ny/2 - thickness/2, yend = ystart + thickness;
            long nxthickness = inhead->nx*thickness;
            long nx = inhead->nx;
            for (j = ystart,j1 = 0 ; j < yend ; j++,j1++)
                for (ii = 0 ; ii < nx ; ii++)
                    indataD[nxthickness*(i-sliceBegin)+j1*nx+ii] = indatatmp[j*nx+ii];
            printf("read slice %d\n",i-sliceBegin);
            fclose(infile);
        }
    }
    printf("read slice finished!\n");
    //--------------------------end of MRCcatch

    //mask
    {
        long nx = inhead->nx;
        long ny = thickness;//inhead->ny;
        long nz = (sliceEnd-sliceBegin+1);
        long i,j,k,i1,j1,k1;
        MrcHeader  *outhead;
        FILE  *fout;
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
	    sprintf(loginfo,"Cannot open outfile '%s'\n",outfile);
	    logwrite(loginfo);
    	    //end of log Write
            exit(1);
        }
        outhead=(MrcHeader *)malloc(sizeof(MrcHeader));

        memcpy(outhead,inhead,sizeof(MrcHeader));
        //mrc_init_head(outhead);
        outhead->nx = inhead->nx;
        outhead->nz = (sliceEnd-sliceBegin+1);
        outhead->ny = thickness;
        outhead->mode = MRC_MODE_FLOAT;
        mrc_write_head(fout,outhead);

        printf("begin mask nx %ld ny %ld nz %ld\n",nx,ny,nz);
        // mask 3d
        fftwf_complex* indataft = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*nx*ny*nz);
        fftwf_plan p = fftwf_plan_dft_r2c_3d(nx,ny,nz,indataD,indataft,FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);

        //read anglefile
        char *lstr, *pch;
        lstr = (char *)malloc(TEXT_LINE_MAX);
        FILE   *fang;
        float *thita;
        thita = (float *)malloc(360*sizeof(float));
        int  ANG_NUM;

        if((fang=fopen(anglefile,"r"))==NULL)
        {
            printf("\nCannot open file %s strike any key exit!\n",anglefile);
	    //log Write
    	    sprintf(loginfo,"running state:\n   ");
    	    logwrite(loginfo);
    	    sprintf(loginfo,"fail!\n");
    	    logwrite(loginfo);
	    sprintf(loginfo,"Error message:\n   ");
	    logwrite(loginfo);
	    sprintf(loginfo,"Cannot open anglefile '%s'\n",anglefile);
	    logwrite(loginfo);
    	    //end of log Write	
            return -1;
        }

        for(i=0; feof(fang)==0;)
        {
            memset(lstr,0,TEXT_LINE_MAX);
            fgets(lstr,TEXT_LINE_MAX,fang);

            pch = strtok(lstr," ;,\n\r\t");

            if(*lstr==EOF||pch == NULL||strncasecmp(pch,"END",3)==0 )break;

            while(pch != NULL)
            {
                thita[i++] = atof(pch);
                pch = strtok(NULL," ;,\n\r\t");
            }

        }
        free(lstr);
        fclose(fang);
        ANG_NUM = i;
        //end of read anglefile


        //generation of bangbang
        long ny_tmp = ny;
        ny = nx;
        float *line = (float *)malloc(nx*ny*sizeof(float));
        float *rotline = (float *)malloc(nx*ny*sizeof(float));
        float *origmask = (float *)malloc(nx*ny*sizeof(float));
        float xf,yf,dx,dy;
        int xi,yi;
        int x,y;
        float costhita,sinthita;
        memset(origmask,0,nx*ny*sizeof(float));
        memset(line,0,nx*ny*sizeof(float));
        for (j = 0 ; j < nx ; j++)
            line[j*nx+nx/2] = 1.0;
        for (k = 0 ; k < ANG_NUM ; k++)
        {
            costhita = cos((-thita[k]-90)*PI_180);
            sinthita = sin((-thita[k]-90)*PI_180);
            memset(rotline,0,nx*ny*sizeof(float));
            for (j = 0 ; j < ny ; j++)
                for (i = 0 ; i < nx ; i++)
                {
                    if (line[j*nx+i] != 0)
                    {
                        xf = (float)(i-nx/2)*costhita-(float)(j-ny/2)*sinthita+nx/2;
                        yf = (float)(i-nx/2)*sinthita+(float)(j-ny/2)*costhita+ny/2;
                        xi = (int)xf;
                        yi = (int)yf;
                        dx = xf - xi;
                        dy = yf - yi;
                        if (xi >= 0 && xi < nx && yi >=0 && yi <ny)
                            rotline[yi*nx+xi] += line[j*nx+i]*(1-dx)*(1-dy);
                        if (xi >= 0 && xi < nx && yi+1 >=0 && yi+1 <ny)
                            rotline[(yi+1)*nx+xi] += line[j*nx+i]*(1-dx)*dy;
                        if (xi+1 >= 0 && xi+1 < nx && yi >=0 && yi <ny)
                            rotline[yi*nx+xi+1] += line[j*nx+i]*dx*(1-dy);
                        if (xi+1 >= 0 && xi+1 < nx && yi+1 >=0 && yi+1 <ny)
                            rotline[(yi+1)*nx+xi+1] += line[j*nx+i]*dx*dy;
                    }
                }
            for (i = 0 ; i < nx*ny ; i++)
            {
                origmask[i] += rotline[i];
            }
            /*{
                FILE *maskf = fopen("mask1","w+");
                for (j = 0 ; j < nx*ny ; j++)
                    if (origmask[j] > 1)
                        fprintf(maskf,"%f\n",1.0);
                    else
                        fprintf(maskf,"%f\n",origmask[j]);
                fclose(maskf);
            }*/
        }

        /*for (i = 0 ; i < nx*ny ; i++)
        	if (origmask[i] > 1)
        		origmask[i] = 1;*/
        //resize mask
        float *origmask2 = (float *)malloc(nx*ny_tmp*sizeof(float));
        memset(origmask2,0,nx*ny_tmp*sizeof(float));
        float resizefactor = (float)ny/(float)ny_tmp;
        float jf,dif;
        int jj1,jj2;
        for (j = 0 ; j < ny_tmp ; j++)
            for (i = 0 ; i < nx ; i++)
            {
                jf = resizefactor*j;
                jj1 = floor(jf);
                if (jj1 != ny-1)
                {
                    jj2 = jj1+1;
                    dif = jf - jj1;
                    origmask2[j*nx+i] += origmask[jj1*nx+i]*(1-dif);
                    origmask2[j*nx+i] += origmask[jj2*nx+i]*dif;
                }
                else
                    origmask2[j*nx+i] += origmask[jj1*nx+i];
            }

        for (i = 0 ; i < nx*ny_tmp ; i++)
            if (origmask2[i] > 1)
                origmask2[i] = 1;
        /*{
            FILE *maskf = fopen("mask2","w+");
            for (j = 0 ; j < nx*ny_tmp ; j++)
                fprintf(maskf,"%f\n",origmask2[j]);
            fclose(maskf);
        }*/

        ny = ny_tmp;
        //end of resize mask

        //end of generation of bangbang
        double w,w2;
        long nynx = ny*(nx/2+1);
        double yfactor = (double)nx/(double)thickness;
        double yfactor2 = yfactor*yfactor;
        long gaussianlenMax;
        if (!paraMark[3])
        {
            gaussianlenMax = radius03-radius05;
            radius = radius05;
        }
        else
        {
            gaussianlenMax = round(((double)nx/150.0)*10.0);
        }
        if (gaussianlenMax < round(((double)nx/150.0)*10.0))
            gaussianlenMax = round(((double)nx/150.0)*10.0);
        long gaussianradius = (radius + gaussianlenMax);
        long gaussianlen = gaussianradius - radius;

        long rr;
        double xxx,c;
        c = (double)gaussianlen/3.0;
        //printf("%ld %ld %f\n",gaussianlen,radius,c);
        for (k = 0 ; k < nz ; k++)
            for (j = 0 ; j < ny ; j++)
                for (i = 0 ; i < nx/2+1 ; i++)
                {
                    i1 = (i + nx/2)%nx;
                    j1 = (j + ny/2)%ny;
                    k1 = (k + nz/2)%nz;
                    rr = sqrt((i1-nx/2)*(i1-nx/2)+yfactor2*yfactor2*(j1-ny/2)*(j1-ny/2)+(k1-nz/2)*(k1-nz/2));
                    if (rr >= radius)
                    {
                        if (rr <= gaussianradius)
                        {
                            w = origmask2[j1*nx+i1];
                            xxx = rr - radius;
                            w2 = (exp(-(xxx*xxx)/(2*c*c))-exp(-(gaussianlen)*(gaussianlen)/(2*c*c)))/(1-exp(-(gaussianlen)*(gaussianlen)/(2*c*c)));
                            w = w > w2 ? w : w2;
                            indataft[k*nynx+j*(nx/2+1)+i][0] *= w;
                            indataft[k*nynx+j*(nx/2+1)+i][1] *= w;
                        }
                        else
                        {
                            w = origmask2[j1*nx+i1];
                            indataft[k*nynx+j*(nx/2+1)+i][0] *= w;
                            indataft[k*nynx+j*(nx/2+1)+i][1] *= w;
                        }
                    }
                }
        /*{
            double yfactor = (double)nx/(double)thickness;
            double yfactor2 = yfactor*yfactor;
            float * pangmask = (float*)malloc(nx*ny*sizeof(float));
            for (j = 0 ; j < ny ; j++)
                for (i = 0 ; i < nx ; i++)
                {
                    i1 = (i + nx/2)%nx;
                    j1 = (j + ny/2)%ny;
                    rr = sqrt((i1-nx/2)*(i1-nx/2)+yfactor2*yfactor2*(j1-ny/2)*(j1-ny/2));
                    if (rr >= radius)
                        pangmask[j*nx+i] = 0;
                    else
                        pangmask[j*nx+i] = 1;
                }
            FILE *maskf = fopen("pangmask","w+");
            for (j = 0 ; j < nx*ny ; j++)
                fprintf(maskf,"%f\n",pangmask[j]);
            fclose(maskf);
            free(pangmask);
        }*/
        p = fftwf_plan_dft_c2r_3d(nx,ny,nz,indataft,indataD,FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
        fftwf_free(indataft);

        double factor = (double)nx*(double)ny*(double)nz;
        long len = (long)nx*(long)ny*(long)nz;
        for (i = 0 ; i < len ; i++)
            indataD[i] /= factor;
        free(line);
        free(rotline);
        free(origmask);
        free(origmask2);
        free(thita);
        //end of mask 3d

        //save data 3d

        float *slice = (float*)malloc(nx*thickness*sizeof(float));
        long nxthickness = nx*thickness;
        for (k = 0 ; k < nz ; k++)
        {
            for (j1 = 0 ; j1 < thickness; j1++)
                for (i = 0 ; i < nx ; i++)
                    slice[j1*nx+i] = indataD[k*nxthickness+j1*nx+i];
            mrc_add_slice(fout,outhead,slice);
        }
        //end of save data 3d

        free(slice);
        free(outhead);
        fclose(fout);
        mrc_update_head(outfile);
    }
    //--------------------------end of mask
    free(indataD);
    free(indatatmp);
    free(inhead);
    free(paraMark);
    //log Write
    sprintf(loginfo,"running state:\n   ");
    logwrite(loginfo);
    sprintf(loginfo,"ICONMask1 finish!\n");
    logwrite(loginfo);
    //end of log Write
    return 0;
}




