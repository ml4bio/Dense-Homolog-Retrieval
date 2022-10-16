#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>
#include <set>
using namespace std;


//-------- utility ------//
void getBaseName(string &in,string &out,char slash,char dot)
{
	int i,j;
	int len=(int)in.length();
	for(i=len-1;i>=0;i--)
	{
		if(in[i]==slash)break;
	}
	i++;
	for(j=len-1;j>=0;j--)
	{
		if(in[j]==dot)break;
	}
	if(j==-1)j=len;
	out=in.substr(i,j-i);
}
void getRootName(string &in,string &out,char slash)
{
	int i;
	int len=(int)in.length();
	for(i=len-1;i>=0;i--)
	{
		if(in[i]==slash)break;
	}
	if(i<=0)out=".";
	else out=in.substr(0,i);
}

//---- load set list -----//
void Load_Set_List(string &fn,set <string> &str_set)
{
	ifstream fin;
	string buf,temp;
	//read
	fin.open(fn.c_str(), ios::in);
	if(fin.fail()!=0)
	{
		fprintf(stderr,"file %s not found!\n",fn.c_str());
		exit(-1);
	}
	//proc
	str_set.clear();
	for(;;)
	{
		if(!getline(fin,buf,'\n'))break;
		str_set.insert(buf);
	}
}

//----- retrieve FASTA sequence according to header ----------//
void Retrieve_FASTA_With_Header(string &fn,set <string> &str_set)
{
	ifstream fin;
	string buf,temp;
	//read
	fin.open(fn.c_str(), ios::in);
	if(fin.fail()!=0)
	{
		fprintf(stderr,"file %s not found!\n",fn.c_str());
		exit(-1);
	}
	//proc
	for(;;)
	{
		if(!getline(fin,buf,'\n'))break;
		if(!getline(fin,temp,'\n'))break;
		string name=buf.substr(1,buf.length()-1);
		set <string>::iterator iter;
		iter = str_set.find(name);
		if(iter == str_set.end())continue;
		//printf
		printf(">%s\n%s\n",name.c_str(),temp.c_str());
	}
}

//------------ main -------------//
int main(int argc, char** argv)
{
	//------- Retrieve_FASTA -----//
	{
		if(argc<3)
		{
			fprintf(stderr,"Retrieve_FASTA <head_file> <db_file> \n");
			exit(-1);
		}
		string head_file=argv[1];
		string db_file=argv[2];
		//proc
		set <string> str_set;
		Load_Set_List(head_file,str_set);
		Retrieve_FASTA_With_Header(db_file,str_set);
		//exit
		exit(0);
	}
}
