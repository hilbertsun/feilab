#include <stdio.h>
#include <stdlib.h>

void main()
{
	char cwd[1000];
	getcwd(cwd,1000);
	FILE *f = fopen("config/logLocation.conf","w+");
	fprintf(f,"%s/ICONlog.txt\n",cwd);
	fclose(f);
    FILE *fl = fopen("config/config.h","w+");
	fprintf(fl,"#define LOGLOCATIONCONF \"%s/config/logLocation.conf\"",cwd);
	fclose(fl);
}
