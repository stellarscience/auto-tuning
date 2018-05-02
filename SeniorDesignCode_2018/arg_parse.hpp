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
//	File Name: arg_parse.hpp
//	
//	Purpose: 	The header file for the arg_parse.cpp
//
/****************************************************************************************/


#ifndef ARG_PARSE
#define ARG_PARSE

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
#include <unistd.h>
#include <cstdlib>
#include <new>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void print_help(int argc, char** argv);
double basic_matrix();
void call_python(int argc, char** argv);
void generate_samples(int argc, char** argv);
void parse_args(int argc, char** argv);
        
#endif
