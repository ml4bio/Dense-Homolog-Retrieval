

# make the subdir first
CXX = g++
#FLAG = -O3 -static
FLAG = -O3
SHELL :=/bin/bash

all:
	$(CXX) Retrieve_FASTA.cpp $(FLAG) -o Retrieve_FASTA
	mv Retrieve_FASTA ../bin

# clean all the temp files
clean:
