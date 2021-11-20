#include "log.h"
#include "../config/config.h"

/*int checklogfile()
{
	if (access(LOGFILE,0) != -1)
		return 1;
	else
		return -1;
}*/

int logwrite(char *loginfo)
{
	FILE *fl = fopen(LOGLOCATIONCONF,"r");
	if (fl != NULL)
	{
		char loglocation[1000];
		fscanf(fl,"%s",loglocation);
		//printf("%s\n",loglocation);
		FILE * f = fopen(loglocation,"a");
		if (f != NULL)
		{
			fprintf(f,"%s",loginfo);
			fclose(f);
		}	
		fclose(fl);
	}
	return 0;
}
