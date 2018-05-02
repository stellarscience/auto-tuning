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
//	File Name: devInfo.hpp
//
//	Purpose:	The header file for devInfo.hpp
//
/****************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sstream>
#include <fstream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>	// Compiler Flag: -framework OpenCL
#else
#include <CL/cl.h>			// Compiler Flag: -lcl
#endif

#ifndef devInfo_H
#define devInfo_H

void clPrintDevInfo(cl_device_id device);
int devicequery(void);

#endif