##########################################################################################
#
#   Adaptive Auto Tuning of Computations on Heterogeneous Environments
#
#   University of New Mexico
#   Department of Electrical and Computer Engineering
#   Melissa Castillo and Christian Curley
#
#   Sponsor and Technical Mentor: Carlos Reyes - Stellar Science
#
#   Last Updated:
#      - May 1st, 2018
#
#   Project:
#   Exploring auto tuning techniques using the Random Forest machine learning algorithm.
#   Performed on SGEMM (Single Precision Matrix Multiplication).
#
#   Test Hardware and OS:
#      - Intel HD Graphics 4000 | macOS High Sierra 10.13 (Tested)
#      - Intel Iris 1536 MB Graphics | macOS High Sierra 10.13 (Tested)
#      - Intel i5-3317U | macOS High Sierra 10.13 (Tested)
#      - Intel i7-4578U | macOS High Sierra 10.13 (Tested)
#
##########################################################################################

# Compilers
CXX = g++

# Compiler Flags
CXXFLAGS += -std=c++11 -O1 -Wall

# OpenCL Library Flags
LDFLAGS += $(libcl_$(shell uname -s))

libcl_Darwin = -framework OpenCL
libcl_Linux  = -lCL

##########################################################################################

# Build Binary from the Objects
# C++ Sources
oclsgemm: main.o devInfo.o host.o arg_parse.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o oclsgemm main.o devInfo.o host.o arg_parse.o

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

devInfo.o : devInfo.cpp devInfo.hpp
	$(CXX) $(CXXFLAGS) -c devInfo.cpp

host.o: host.cpp host.hpp
	$(CXX) $(CXXFLAGS) -c host.cpp

arg_parse.o: arg_parse.cpp arg_parse.hpp
	$(CXX) $(CXXFLAGS) -c arg_parse.cpp

# Execute Binaries
run:
	./oclsgemm -h

# Clean-Up
clean:
	rm -f *.o *~

