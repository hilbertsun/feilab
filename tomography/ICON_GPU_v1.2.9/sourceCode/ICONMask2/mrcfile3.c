#include "mrcfile3.h"


long get_file_size(FILE *fin)
{
	fseek(fin,0,SEEK_END);

	return ftell(fin);

}


/*******************************************************************************************/
int mrc_read_head (FILE *fin,  MrcHeader *head)
{

	if(ftello64(fin)!=0)rewind(fin);
  	fread(head,1024,1,fin);
	return 0;

}


/*******************************************************************************************/
int mrc_write_head(FILE *fout, MrcHeader *head)
{

	if(ftello64(fout)!=0)rewind(fout);
	fwrite(head,1024,1,fout);
	char *empty = (char*)malloc(head->next*sizeof(char));
	fwrite(empty,head->next,1,fout);
	free(empty);
  	return 0;
}




/*******************************************************************************************/
int mrc_init_head(MrcHeader *head)
{

	head->nx=1;
	head->ny=1;
	head->nz=1;

	head->mode = MRC_MODE_FLOAT;

	head->nxstart=0;
	head->nystart=0;
	head->nzstart=0;

	head->mx=1;
	head->my=1;
	head->mz=1;

	head->xlen=1;
	head->ylen=1;
	head->zlen=1;

	head->alpha=90;
	head->beta=90;
	head->gamma=90;

	head->mapc=1;
	head->mapr=2;
	head->maps=3;

	head->amin=0.0;
	head->amax=255.0;
	head->amean=128.0;

	head->ispg=1;
	head->nsymbt=0;

	head->next=0;

	head->creatid=0;
	head->cmap[0]='M';
	head->cmap[1]='A';
	head->cmap[2]='P';
        head->cmap[3]=' ';

	head->stamp[0]='D';
	head->nlabl = 0;

	head->imodStamp = 1146047817;

	return 0;
}

/*******************************************************************************************/
int mrc_replace_head(char *outf,MrcHeader *head)
{

	FILE *fout;
	if((fout=fopen(outf,"r+"))==NULL)
	{
		printf("Cannot open file strike any key exit!\n");
		exit(1);
	}
	mrc_write_head(fout,head);
	fclose(fout);

	return TRUE;

}


/*******************************************************************************************/
//only MRC_MODE_BYTE MRC_MODE_SHORT MRC_MODE_FLOAT will be considered in this function
//calculate the amin amax amean and std(rms)
int mrc_update_head(char *inoutf)
{

	MrcHeader  *head;
	head=(MrcHeader *)malloc(sizeof(MrcHeader));

	FILE *finout;

	if((finout=fopen(inoutf,"r+"))==NULL)
	{
		printf("Cannot open file strike any key exit!\n");
		exit(1);

	}
	
	mrc_read_head(finout,head);

	long filesize = get_file_size(finout);
	long headsize;

	switch(head->mode)
	{
	      	case MRC_MODE_BYTE:
            	headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(char);

		break;

		case MRC_MODE_SHORT:
	    	case MRC_MODE_USHORT:
            	headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(short);
			
           	break;

		case MRC_MODE_FLOAT:
            	headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(float);
			
            	break;
			
		default:
		printf(" File type unknown!");

            	break;

	}

	double sum, sum_xy, amin, amax, amean,std,std_xy;
	long k, pNum, pLen;
	//pNum is the number of pixels in one XoY  slice, 
	//pLen is the 
	unsigned long site;
	unsigned char *p_uchar;
	short *p_short;
	float *p_float;


        // get mean & max & min
	fseek(finout,headsize,SEEK_SET);

	pNum=head->nx*head->ny;

	switch(head->mode)
	{
		//switch start

		/**********case MRC_MODE_BYTE ***********/
		case MRC_MODE_BYTE :
		
			pLen=pNum*sizeof(unsigned char);

			if((p_uchar=(unsigned char *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}

			fread(p_uchar,pLen,1,finout);

			sum=sum_xy=0.0;
			amin=amax=p_uchar[0];

			for(site=0;site<pNum;site++)
			{
				if(p_uchar[site]>amax)amax=p_uchar[site];
				if(p_uchar[site]<amin)amin=p_uchar[site];
				sum_xy+=p_uchar[site];

			}

			sum+=sum_xy;

			for(k=1;k<head->nz;k++)
			{

				sum_xy=0.0;

				fread(p_uchar,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					if(p_uchar[site]>amax)amax=p_uchar[site];
					if(p_uchar[site]<amin)amin=p_uchar[site];
					sum_xy+=p_uchar[site];

				}

				sum+=sum_xy;

			}

			amean=sum/(head->nx*head->ny*head->nz);

			free(p_uchar);

		break;


		/**********case MRC_MODE_SHORT ***********/
		case MRC_MODE_SHORT :
		
			pLen=pNum*sizeof(short);

			if((p_short=( short *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}

			fread(p_short,pLen,1,finout);

			sum=sum_xy=0.0;
			amin=amax=p_short[0];

			for(site=0;site<pNum;site++)
			{
				if(p_short[site]>amax)amax=p_short[site];
				if(p_short[site]<amin)amin=p_short[site];
				sum_xy+=p_short[site];

			}

			sum+=sum_xy;

			for(k=1;k<head->nz;k++)
			{

				sum_xy=0.0;

				fread(p_short,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					if(p_short[site]>amax)amax=p_short[site];
					if(p_short[site]<amin)amin=p_short[site];
					sum_xy+=p_short[site];

				}

				sum+=sum_xy;

			}

			amean=sum/(head->nx*head->ny*head->nz);

			free(p_short);

		break;



		/**********case MRC_MODE_FLOAT ***********/
		case MRC_MODE_FLOAT :
		
			pLen=pNum*sizeof(float);

			if((p_float=( float *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}


			fread(p_float,pLen,1,finout);

			sum=sum_xy=0.0;
			amin=amax=p_float[0];

			for(site=0;site<pNum;site++)
			{
				if(p_float[site]>amax)amax=p_float[site];
				if(p_float[site]<amin)amin=p_float[site];
				sum_xy+=p_float[site];

			}

			sum+=sum_xy;

			for(k=1;k<head->nz;k++)
			{

				sum_xy=0.0;

				fread(p_float,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					if(p_float[site]>amax)amax=p_float[site];
					if(p_float[site]<amin)amin=p_float[site];
					sum_xy+=p_float[site];

				}

				sum+=sum_xy;

			}

			amean=sum/(head->nx*head->ny*head->nz);

			free(p_float);

		break;

	}     //switch end

	head->amin=amin;
	head->amax=amax;
	head->amean=amean;

        //----------------------------end of get mean & max & min

	// get std
	fseek(finout,headsize,SEEK_SET);

	pNum=head->nx*head->ny;

	switch(head->mode)
	{
		//switch start

		/**********case MRC_MODE_BYTE ***********/
		case MRC_MODE_BYTE :
		
			pLen=pNum*sizeof(unsigned char);

			if((p_uchar=(unsigned char *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}

			fread(p_uchar,pLen,1,finout);

			std = 0.0;
			std_xy = 0.0;

			for(site=0;site<pNum;site++)
			{
				std_xy+= (p_uchar[site] - amean)*(p_uchar[site] - amean);
			}

			std+=std_xy;

			for(k=1;k<head->nz;k++)
			{

				std_xy=0.0;

				fread(p_uchar,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					std_xy+=(p_uchar[site] - amean)*(p_uchar[site] - amean);

				}

				std+=std_xy;

			}

			std=sqrt(std/(head->nx*head->ny*head->nz));

			free(p_uchar);

		break;


		/**********case MRC_MODE_SHORT ***********/
		case MRC_MODE_SHORT :
		
			pLen=pNum*sizeof(short);

			if((p_short=( short *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}

			fread(p_short,pLen,1,finout);

			std = 0.0;
			std_xy = 0.0;

			for(site=0;site<pNum;site++)
			{
				std_xy+= (p_short[site] - amean)*(p_short[site] - amean);

			}

			std+=std_xy;

			for(k=1;k<head->nz;k++)
			{

				std_xy=0.0;

				fread(p_short,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					std_xy+= (p_short[site] - amean)*(p_short[site] - amean);
				}

				std+=std_xy;

			}

			std=sqrt(std/(head->nx*head->ny*head->nz));

			free(p_short);

		break;



		/**********case MRC_MODE_FLOAT ***********/
		case MRC_MODE_FLOAT :
		
			pLen=pNum*sizeof(float);

			if((p_float=( float *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}


			fread(p_float,pLen,1,finout);

			std = 0.0;
			std_xy = 0.0;

			for(site=0;site<pNum;site++)
			{
				std_xy+= (p_float[site] - amean)*(p_float[site] - amean);

			}

			std+=std_xy;

			for(k=1;k<head->nz;k++)
			{

				std_xy=0.0;

				fread(p_float,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					std_xy+= (p_float[site] - amean)*(p_float[site] - amean);

				}

				std+=std_xy;

			}

			std=sqrt(std/(head->nx*head->ny*head->nz));

			free(p_float);

		break;

	}     //switch end

        //----------------------------end of get std

	fclose(finout);

	head->rms = std;

	mrc_replace_head(inoutf,head);

        //printf("%f %f %f %f\n",head->amean,head->amax,head->amin,head->rms);
	free(head);

	return 0;
}

/*******************************************************************************************/
//only MRC_MODE_BYTE MRC_MODE_SHORT MRC_MODE_FLOAT will be considered in this function
//calculate the std(rms)
int mrc_update_rms(char *inoutf)
{

	MrcHeader  *head;
	head=(MrcHeader *)malloc(sizeof(MrcHeader));

	FILE *finout;

	if((finout=fopen(inoutf,"r+"))==NULL)
	{
		printf("Cannot open file strike any key exit!\n");
		exit(1);

	}
	
	mrc_read_head(finout,head);

	long filesize = get_file_size(finout);
	long headsize;

	switch(head->mode)
	{
	      	case MRC_MODE_BYTE:
            	headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(char);

		break;

		case MRC_MODE_SHORT:
	    	case MRC_MODE_USHORT:
            	headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(short);
			
           	break;

		case MRC_MODE_FLOAT:
            	headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(float);
			
            	break;
			
		default:
		printf(" File type unknown!");

            	break;

	}

	double sum, sum_xy, amean,std,std_xy;
	long k, pNum, pLen;
	//pNum is the number of pixels in one XoY  slice, 
	//pLen is the 
	unsigned long site;
	unsigned char *p_uchar;
	short *p_short;
	float *p_float;


        amean = head->amean;
        //----------------------------end of get mean & max & min

	// get std
	fseek(finout,headsize,SEEK_SET);

	pNum=head->nx*head->ny;

	switch(head->mode)
	{
		//switch start

		/**********case MRC_MODE_BYTE ***********/
		case MRC_MODE_BYTE :
		
			pLen=pNum*sizeof(unsigned char);

			if((p_uchar=(unsigned char *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}

			fread(p_uchar,pLen,1,finout);

			std = 0.0;
			std_xy = 0.0;

			for(site=0;site<pNum;site++)
			{
				std_xy+= (p_uchar[site] - amean)*(p_uchar[site] - amean);
			}

			std+=std_xy;

			for(k=1;k<head->nz;k++)
			{

				std_xy=0.0;

				fread(p_uchar,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					std_xy+=(p_uchar[site] - amean)*(p_uchar[site] - amean);

				}

				std+=std_xy;

			}

			std=sqrt(std/(head->nx*head->ny*head->nz));

			free(p_uchar);

		break;


		/**********case MRC_MODE_SHORT ***********/
		case MRC_MODE_SHORT :
		
			pLen=pNum*sizeof(short);

			if((p_short=( short *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}

			fread(p_short,pLen,1,finout);

			std = 0.0;
			std_xy = 0.0;

			for(site=0;site<pNum;site++)
			{
				std_xy+= (p_short[site] - amean)*(p_short[site] - amean);

			}

			std+=std_xy;

			for(k=1;k<head->nz;k++)
			{

				std_xy=0.0;

				fread(p_short,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					std_xy+= (p_short[site] - amean)*(p_short[site] - amean);
				}

				std+=std_xy;

			}

			std=sqrt(std/(head->nx*head->ny*head->nz));

			free(p_short);

		break;



		/**********case MRC_MODE_FLOAT ***********/
		case MRC_MODE_FLOAT :
		
			pLen=pNum*sizeof(float);

			if((p_float=( float *)malloc(pLen))==NULL)
			{
				printf("Function 'malloc' erro, while updating head!\n");
				exit(1);

			}


			fread(p_float,pLen,1,finout);

			std = 0.0;
			std_xy = 0.0;

			for(site=0;site<pNum;site++)
			{
				std_xy+= (p_float[site] - amean)*(p_float[site] - amean);

			}

			std+=std_xy;

			for(k=1;k<head->nz;k++)
			{

				std_xy=0.0;

				fread(p_float,pLen,1,finout);

				for(site=0;site<pNum;site++)
				{
					std_xy+= (p_float[site] - amean)*(p_float[site] - amean);

				}

				std+=std_xy;

			}

			std=sqrt(std/(head->nx*head->ny*head->nz));

			free(p_float);

		break;

	}     //switch end

        //----------------------------end of get std

	fclose(finout);

	head->rms = std;

	mrc_replace_head(inoutf,head);

        //printf("%f %f %f %f\n",head->amean,head->amax,head->amin,head->rms);
	free(head);

	return 0;
}



/*******************************************************************************************/
/*******slcN couts from 0 to N-1, so if you want to read the first slice slcN shoud be 0****/

int mrc_read_slice(FILE *fin, MrcHeader  *head, int slcN, char axis, float *slcdata)
{

//check the mrc file to make sure the size is exact in register with the head
		long filesize = get_file_size(fin);
		long headsize;

		switch(head->mode)
		{
	      		case MRC_MODE_BYTE:
            		headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(char);

			break;

		    	case MRC_MODE_SHORT:
	    		case MRC_MODE_USHORT:
            		headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(short);
			
           		break;

			case MRC_MODE_FLOAT:
            		headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(float);
			
            		break;
			
			default:
			printf(" File type unknown!");

            		break;

		}
		//printf("headsize %ld next %d\n",headsize,head->next);
		switch(head->mode)
		{
			case MRC_MODE_BYTE:
			if(get_file_size(fin) - headsize != (long)head->nx*(long)head->ny*(long)head->nz*sizeof(char))
			{
				printf("Error with Function 'mrc_read_slic()'! File size erro!\n");
			}
			break;

			case MRC_MODE_SHORT:
			case MRC_MODE_USHORT:
			if(get_file_size(fin) - headsize != (long)head->nx*(long)head->ny*(long)head->nz*sizeof(short))
			{
				printf("Error with Function 'mrc_read_slice()'! File size erro!\n");
			}
			break;

			case MRC_MODE_FLOAT:
			if(get_file_size(fin) - headsize != (long)head->nx*(long)head->ny*(long)head->nz*sizeof(float))
			{
				printf("Error with Function 'mrc_read_slice()'! File size erro!\n");
			}
			break;
			
			default:
			printf("Error with Function 'mrc_read_slice()'! File type unknown!\n");

			break;
		}


	long psize;
	short buf_short;
	unsigned short buf_ushort;
	unsigned char buf_byte;
	float buf_float;
	long i,k;
	
	switch(head->mode)
	{
		case MRC_MODE_BYTE :
			psize=sizeof(unsigned char);
	
			break;


		case MRC_MODE_SHORT :
		case MRC_MODE_USHORT:
			psize=sizeof(short);
	
			break;
	
		case MRC_MODE_FLOAT :
			psize=sizeof(float);

			break;
	}




	switch(axis)
	{
	
	/***********************************X************************************/
		case 'x':
		case 'X':
	
		fseek(fin,headsize+(long)slcN*(long)psize,SEEK_SET);
	
	
		switch(head->mode)
		{
			case MRC_MODE_BYTE:
			for(i=0;i<head->ny*head->nz;i++)
				{
					fread(&buf_byte,psize,1,fin);
					slcdata[i]=(float)buf_byte;
					fseek(fin,(head->nx-1)*psize,SEEK_CUR);
				}
	
			break;
	
			case MRC_MODE_SHORT:
			for(i=0;i<head->ny*head->nz;i++)
				{
					fread(&buf_short,psize,1,fin);
					slcdata[i]=(float)(buf_short);
					fseek(fin,(head->nx-1)*psize,SEEK_CUR);
				}

			break;

			case MRC_MODE_USHORT:
			for(i=0;i<head->ny*head->nz;i++)
				{
					fread(&buf_ushort,psize,1,fin);
					slcdata[i]=(float)(buf_ushort);
					fseek(fin,(head->nx-1)*psize,SEEK_CUR);
				}

			break;

			case MRC_MODE_FLOAT:
			for(i=0;i<head->ny*head->nz;i++)
				{
				fread(&buf_float,psize,1,fin);
				slcdata[i]=buf_float;
				fseek(fin,(head->nx-1)*psize,SEEK_CUR);
				}
			break;
	
		}
	
		break;
	
	/***********************************Y************************************/
		case 'y':
		case 'Y':
	
		for(k=0;k<head->nz;k++)
		{
			fseek(fin,headsize+(k*(long)head->nx*(long)head->ny+(long)head->nx*(long)slcN)*psize,SEEK_SET);
	
	
		switch(head->mode)
		{
			case MRC_MODE_BYTE:
			for(i=0;i<head->nx;i++)
				{
					fread(&buf_byte,psize,1,fin);
					slcdata[k*head->nx+i]=(float)buf_byte;
				}
	
			break;
	
			case MRC_MODE_SHORT:
			for(i=0;i<head->nx;i++)
				{
					fread(&buf_short,psize,1,fin);
					slcdata[k*head->nx+i]=(float)(buf_short);
				}
	
			break;

			case MRC_MODE_USHORT:
			for(i=0;i<head->nx;i++)
				{
					fread(&buf_ushort,psize,1,fin);
					slcdata[k*head->nx+i]=(float)(buf_ushort);
				}
	
			break;
	
			case MRC_MODE_FLOAT:
			for(i=0;i<head->nx;i++)
			{
				fread(&buf_float,psize,1,fin);
				slcdata[k*head->nx+i]=buf_float;
			}
	
			break;
	
		}

		}
		break;
	
	/***********************************Z************************************/
		case 'z':
		case 'Z':

		fseek(fin,headsize+(long)slcN*(long)head->nx*(long)head->ny*(long)psize,SEEK_SET);
	
		if(head->mode==MRC_MODE_FLOAT)fread(slcdata,psize*head->nx*head->ny,1,fin); 
	
		else if(head->mode==MRC_MODE_BYTE)
		{
			for(i=0;i<head->nx*head->ny;i++)
			{
			fread(&buf_byte,psize,1,fin);
			slcdata[i]=(float)buf_byte;
			}
		}
	
		else if(head->mode==MRC_MODE_SHORT)
		{
			for(i=0;i<head->nx*head->ny;i++)
			{
				fread(&buf_short,psize,1,fin);
				slcdata[i]=(float)buf_short;
			}
		}

		else if(head->mode==MRC_MODE_USHORT)
		{
			for(i=0;i<head->nx*head->ny;i++)
			{
				fread(&buf_ushort,psize,1,fin);
				slcdata[i]=(float)buf_ushort;
			}
		}


		break;

	}
	/*FILE *f = fopen("slice","w+");
	for (i = 0 ; i < head->nx*head->ny ; i++)
		fprintf(f,"%f\n",slcdata[i]);
	fclose(f);*/
	return 0;
}

/*******************************************************************************************/
/*******slcN couts from 0 to N-1, so if you want to read the first slice slcN shoud be 0****/
int mrc_read_slice_in_Z_with_thickness(FILE *fin, MrcHeader  *head, int slcN, char axis, float *slcdata,int ystart,int yend)
{

//check the mrc file to make sure the size is exact in register with the head
		long filesize = get_file_size(fin);
		long headsize;

		switch(head->mode)
		{
	      		case MRC_MODE_BYTE:
            		headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(char);

			break;

		    	case MRC_MODE_SHORT:
	    		case MRC_MODE_USHORT:
            		headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(short);
			
           		break;

			case MRC_MODE_FLOAT:
            		headsize=filesize-(long long)head->nx*head->ny*head->nz*sizeof(float);
			
            		break;
			
			default:
			printf(" File type unknown!");

            		break;

		}
		//printf("headsize %ld next %d\n",headsize,head->next);
		switch(head->mode)
		{
			case MRC_MODE_BYTE:
			if(get_file_size(fin) - headsize != (long)head->nx*(long)head->ny*(long)head->nz*sizeof(char))
			{
				printf("Error with Function 'mrc_read_slic()'! File size erro!\n");
			}
			break;

			case MRC_MODE_SHORT:
			case MRC_MODE_USHORT:
			if(get_file_size(fin) - headsize != (long)head->nx*(long)head->ny*(long)head->nz*sizeof(short))
			{
				printf("Error with Function 'mrc_read_slice()'! File size erro!\n");
			}
			break;

			case MRC_MODE_FLOAT:
			if(get_file_size(fin) - headsize != (long)head->nx*(long)head->ny*(long)head->nz*sizeof(float))
			{
				printf("Error with Function 'mrc_read_slice()'! File size erro!\n");
			}
			break;
			
			default:
			printf("Error with Function 'mrc_read_slice()'! File type unknown!\n");

			break;
		}


	long psize;
	short buf_short;
	unsigned short buf_ushort;
	unsigned char buf_byte;
	float buf_float;
	long i,k;
	
	switch(head->mode)
	{
		case MRC_MODE_BYTE :
			psize=sizeof(unsigned char);
	
			break;


		case MRC_MODE_SHORT :
		case MRC_MODE_USHORT:
			psize=sizeof(short);
	
			break;
	
		case MRC_MODE_FLOAT :
			psize=sizeof(float);

			break;
	}




	switch(axis)
	{
	
	/***********************************X************************************/
		
	/***********************************Y************************************/
	
	/***********************************Z************************************/
		case 'z':
		case 'Z':

		fseek(fin,headsize+(long)slcN*(long)head->nx*(long)head->ny*(long)psize+(long)ystart*(long)head->nx*(long)psize,SEEK_SET);

		int ysize = yend - ystart;

		if(head->mode==MRC_MODE_FLOAT)fread(slcdata,psize*head->nx*ysize,1,fin); 
	
		else if(head->mode==MRC_MODE_BYTE)
		{
			for(i=0;i<head->nx*ysize;i++)
			{
			fread(&buf_byte,psize,1,fin);
			slcdata[i]=(float)buf_byte;
			}
		}
	
		else if(head->mode==MRC_MODE_SHORT)
		{
			for(i=0;i<head->nx*ysize;i++)
			{
				fread(&buf_short,psize,1,fin);
				slcdata[i]=(float)buf_short;
			}
		}

		else if(head->mode==MRC_MODE_USHORT)
		{
			for(i=0;i<head->nx*ysize;i++)
			{
				fread(&buf_ushort,psize,1,fin);
				slcdata[i]=(float)buf_ushort;
			}
		}


		break;

	}
	/*FILE *f = fopen("slice","w+");
	for (i = 0 ; i < head->nx*head->ny ; i++)
		fprintf(f,"%f\n",slcdata[i]);
	fclose(f);*/
	return 0;
}

/*******************************************************************************************/
int mrc_add_slice(FILE *fout , MrcHeader  *headout, float *slcdata)
{
	fseeko(fout,0,SEEK_END);
	fwrite(slcdata,headout->nx*headout->ny*sizeof(float),1,fout);
	/*FILE *f = fopen("slice","w+");
	int i;
	for (i = 0 ; i < headout->nx*headout->ny ; i++)
		fprintf(f,"%f\n",slcdata[i]);
	fclose(f);*/
	//fflush(fout);
	return 0;
}





