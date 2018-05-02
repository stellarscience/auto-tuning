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
//	File Name: host.cpp
//	Function(s): host(), seedMatrix(), LoadOpenCLKernel()
//	
//	Purpose: 	This file contains the necessary operations to properly execute the OpenCL
//				kernel.
//
/****************************************************************************************/

#include "host.hpp"

// Allocates a matrix with random float entries.
void seedMatrix(float* data, int size)
{

   for (int i = 0; i < size; ++i)
   	data[i] = rand() / (float)RAND_MAX;
   
}

long LoadOpenCLKernel(char const* path, char **buffer)
{
    FILE  *fptr;
    size_t filesize;
    long   off_end;
    int    rc;
    
    // Open the file 
    fptr = fopen(path, "r");
    if( NULL == fptr ) {
        return -1L;
    }

    // Seek to the end of the file 
    rc = fseek(fptr, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }

    // Byte offset to the end of the file (size) 
    if( 0 > (off_end = ftell(fptr)) ) {
        return -1L;
    }
    filesize = (size_t)off_end;

    // Allocate a buffer to hold the whole file 
    *buffer = (char *) malloc( filesize+1 );
    if( NULL == *buffer ) {
        return -1L;
    }

    // Rewind file pointer to start of file 
    rewind(fptr);

    // Slurp file into buffer 
    if( filesize != fread(*buffer, 1, filesize, fptr) ) {
        free(*buffer);
        return -1L;
    }

    // Close the file 
    if( EOF == fclose(fptr) ) {
        free(*buffer);
        return -1L;
    }


    // Make sure the buffer is NULL-terminated
    (*buffer)[filesize] = '\0';

    // Return the file size 
    return (long)filesize;
}

//Function to display the matrices
void printMatrix(float* buffer, int dimension){
	
	for(int i = 0; i < dimension; i++){
		for(int j = 0; j < dimension; j++){			
			printf("%03.2f\t ", buffer[i*dimension+j]);
		}
		printf("\n");
	}
	printf("\n");
	
}


double host(const int matrix_dim,  const int local_mem, 
			const int block_size,  const int display){
			
	if(block_size > matrix_dim){
		std::cout << "	Block size exceeds matrix dimension size!" << std::endl;
		return -1;
	}
			
	
	std::string kernel_name;
	const char * nameKernel = kernel_name.c_str();
	std::string program_name;
	const char * nameProgram = program_name.c_str();
	
	kernel_name = "sgemm.cl";
	program_name = "sgemm";	

	int dim = matrix_dim;
	
	std::cout << " Size of dim: " 				<< size_t(matrix_dim) 	<< std::endl;
	std::cout << " Size of local mem: " 		<< size_t(local_mem) 	<< std::endl;
	std::cout << " Size of block sub matrix: " 	<< size_t(block_size) 	<< std::endl;
	

	//Set OpenCL Variables
	cl_int				err;                            
   	cl_device_id 		device_id;            
   	cl_context 			context;                 
   	cl_command_queue 	queue;          
   	cl_program 			program;                 
   	cl_kernel 			kernel;                   
   	
   	// OpenCL device memory for matrices
   	cl_mem d_A;
   	cl_mem d_B;
   	cl_mem d_C;
   	
   	//Allocate host memory for matrices A and B
   	unsigned int size_A = dim * dim;
   	unsigned int mem_size_A = sizeof(float) * size_A;
   	float* h_A = (float*) malloc(mem_size_A);
   	float* hostA_copy = h_A;
 
   	unsigned int size_B = dim * dim;
   	unsigned int mem_size_B = sizeof(float) * size_B;
   	float* h_B = (float*) malloc(mem_size_B);
   	float* hostB_copy = h_B;

   	//Initialize host memory
   	seedMatrix(hostA_copy, size_A);
   	seedMatrix(hostB_copy, size_B);
 
   	//Allocate host memory for the result C
   	unsigned int size_C = dim * dim;
   	unsigned int mem_size_C = sizeof(float) * size_C;
   	float* h_C = (float*) malloc(mem_size_C);
   	float* hostC_copy = h_C;
   	
   	//Allocate host memory for test function
   	unsigned int size_test = dim * dim;
   	unsigned int mem_size_test = sizeof(float) * size_test;
   	float* h_test = (float*) malloc(mem_size_test);
   	float* h_copy = h_test;
   	
   	//OpenCL Platforms
   	cl_uint dev_cnt = 0;
   	clGetPlatformIDs(0, 0, &dev_cnt);
	
   	cl_platform_id platform_ids[100];
   	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
   	
   	//Select OpenCL Device
   	err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
   	if (err != CL_SUCCESS)
   	{
    	std::cerr << "	Error: Failed to create a device group!\n";
    	return -1;
   	}
   	
   	// Create a compute context 
   	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
   	if (!context)
   	{
       	std::cerr << "	Error. Failed to create a compute context!\n";
       	return -1;
   	}
   	
   	// Create a command queue
   	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
   	if (!queue)
   	{
       	std::cerr << "	Error. Failed to create a command queue!\n";
       	return -1;
   	}
   	
   	// Create the compute program from the source file
   	char *KernelSource;
   	long lFileSize;

   	lFileSize = LoadOpenCLKernel(nameKernel, &KernelSource);
   	if( lFileSize < 0L ) {
       	perror("	File read failed");
       	return -1;
   	}
   	
   	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
   	if (!program)
   	{
       	std::cerr << "	Error. Failed to create compute program!\n";
       	return -1;
   	}
   	
   	// Build the program executable with options
   	
   	char options_buffer[300];
   	sprintf(options_buffer, "-D LOCAL_MEM=%d -D BLOCK_SIZE=%d", local_mem, block_size);
   	
   	err = clBuildProgram(program, 0, NULL, options_buffer, NULL, NULL);
   	if (err != CL_SUCCESS)
   	{
       	size_t len;
       	char buffer[2048];
       	std::cerr << "	Error. Failed to build program executable!\n";
       	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       	std::cout << buffer;
       	free(h_A);
   		free(h_B);
   		free(h_C);
       	return -1;
   	}
   	
   	// Create the compute kernel in the program we wish to run
   	//
   	kernel = clCreateKernel(program, nameProgram, &err);
   	if (!kernel || err != CL_SUCCESS)
   	{
       	std::cerr << "	Error. Failed to create compute kernel!\n";
       	return -1;
   	}
   	
   	// Create the input and output arrays in device memory for our calculation
   	d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
   	d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, hostA_copy, &err);
   	d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, hostB_copy, &err);

   	if (!d_A || !d_B || !d_C)
   	{
       	std::cerr << "	Error. Failed to allocate device memory!\n";
       	return -1;
   	}
   	
   	std::cout << "	Running matrix multiplication for matrices A (" << dim 
   			  << "x" << dim << ")  and B (" << dim << "x" << dim << ") ...\n";
   			  
   	//Launch OpenCL kernel
   	size_t localWorkSize[2];	
   	size_t globalWorkSize[2]; 	//The global_work_size is essentially the size of your problem
   								//If the global_work_size is not the size of your problem it causes weird computations

	//Set Kernel Arguments
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
   	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
   	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
   	err |= clSetKernelArg(kernel, 3, sizeof(int)   , (void *)&dim);

   	if (err != CL_SUCCESS)
   	{
       std::cerr << "	Error. Failed to set kernel arguments!" << err << std::endl;
       return -1;
   	}
   	
   	cl_event event;
   	double time;
   	
   	//Local and Global Work Size
   	localWorkSize[0] 	= block_size;
   	localWorkSize[1] 	= block_size;
   	globalWorkSize[0]	= dim;
   	globalWorkSize[1] 	= dim;
   	
   	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);

    if (err != CL_SUCCESS)
    {
    	std::cerr << "	Error. Failed to execute kernel!" << err << std::endl;
       	free(h_A);
   		free(h_B);
   		free(h_C);
    	return -1;
    }
    
    err = clFinish(queue);
   
    if (err != CL_SUCCESS)
    {
   	 	std::cerr << "	Error. Waiting for kernel!" << err << std::endl;
   	   	free(h_A);
   		free(h_B);
   		free(h_C);
   	 	return -1;
    }
    
    // The unsigned 64-bit values returned can be used to measure the time in nano-seconds consumed by OpenCL commands.
    cl_ulong start_time, end_time;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
      
    // time in milliseconds
    std::cout << "	Execution Time (msec): " << (double)(end_time - start_time)/1000000.0 << std::endl;
    time = (double)(end_time - start_time)/1000000.0;
    
    if (err != CL_SUCCESS)
    {
   	   if 		(err == CL_PROFILING_INFO_NOT_AVAILABLE) {
   	   			std::cerr << "	Error. Cl profiling info not available! " << std::endl; }
   	   else if 	(err == CL_INVALID_VALUE) {
   	   			std::cerr << "	Error. Cl invalid value! " << std::endl; }
   	   else if 	(err == CL_INVALID_EVENT) {
   	   			std::cerr << "	Error. Cl invalid event! " << std::endl; }
   	   else {
   	   			std::cerr << "	Error. Timing Error!" << err <<std::endl; }
   	   return -1;
    }
    
    //Retrieve result from device
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mem_size_C, hostC_copy, 0, NULL, NULL);

    if (err != CL_SUCCESS)
    {
       std::cerr << "	Error. Failed to read output array!" << err << std::endl;
       	free(h_A);
   		free(h_B);
   		free(h_C);
       return -1;
    }
   
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    
    if(display){
    	printf("\n	Matrix A \n==========================\n");
    	printMatrix(hostA_copy, dim);
    	
    	printf("\n	Matrix B \n==========================\n");
    	printMatrix(hostB_copy, dim);
    	
    	printf("\n	Matrix C \n==========================\n");
    	printMatrix(hostC_copy, dim);
    }
    
    // Display execution time
    //std::cout << "	Kernel Execution Time (msec): " << (double)(end_time - start_time)/1000000.0 << std::endl;
    
    // Test for equality
    clock_t clk_start, clk_end;
    clk_start = clock();
    for (int i = 0; i < dim; i++){
		for (int j = 0; j < dim; j++){
			h_copy[i*dim + j] = 0;
			for (int k = 0; k < dim; k++){
				h_copy[i*dim + j] += hostA_copy[i*dim + k] * hostB_copy[k*dim + j];
				//h_copy[i*dim + j] += 2;
			}
		}
	}
	clk_end = clock();
	//mtxO3 not mtx03
	double mtxO3 = double(clk_end - clk_start)/(CLOCKS_PER_SEC);
	//printf("	MtxO3 running time is: %f milliseconds\n", mtxO3*1000);
	
	int flag = 0;
	clock_t compare_start, compare_end;
	compare_start = clock();
	for (int i = 0; i < dim; i++){
		for (int j = 0; j < dim; j++){
			
			if(hostC_copy[i*dim + j] != h_copy[i*dim + j]){
				flag = 1;
				break;
			}
		}
    }
    compare_end = clock();
    double mtx_compare = double(compare_end - compare_start)/(CLOCKS_PER_SEC);
    
    if (flag){
    	printf("	The kernel matrix is not equal\n");
    	return -1;
    }
    else {
    	printf("	The matrices are equal!\n");
    	printf("	Kernel Execution Time is %f milliseconds\n", (double)(end_time - start_time)/1000000.0);
    	printf("	MtxO3 running time is: %f milliseconds\n", mtxO3*1000);
    	printf("	Comparison execution time is %f milliseconds\n", mtx_compare*1000);
    	
    }
    
    //Shutdown and cleanup
    free(h_A);
   	free(h_B);
   	free(h_C);
   	free(h_test);
 
   	clReleaseMemObject(d_A);
   	clReleaseMemObject(d_B);
   	clReleaseMemObject(d_C);

   	clReleaseProgram(program);
   	clReleaseKernel(kernel);
   	clReleaseCommandQueue(queue);
   	clReleaseContext(context);
   
   	return time;
   	
}