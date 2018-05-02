/****************************************************************************************/
//
//	Adaptive Auto Tuning of Computations on Heterogeneous Environments
//
//	University of New Mexico
//	Department of Electrical and Computer Engineering
//	Melissa Castillo and Christian Curley
//
//	Sponsor and Technical Mentor: Carlos Reyes - Stellar Science
//
// ---------------------------------------------------------------------------------------
//
//	Last Update: May 1st, 2018
//	
//	File Name: host.hpp
//	Purpose of File: Header File for host.cpp 
//
/****************************************************************************************/

#ifndef HOST
#define HOST

#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <fstream>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void seedMatrix(float* data, int size);
long LoadOpenCLKernel(char const* path, char **buf);
void printMatrix(float* buffer, int dimension);
double host(const int matrix_dim,  const int local_mem, 
			const int block_size,  const int display);

#endif