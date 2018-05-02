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
//	File Name: sgemm.cl
//	Function(s): sgemm
//		Parameter(s):	__global float* C, const __global float*A, const __global float*B,
//						const int dim
//
//	Purpose:  	OpenCL Kernel Used to Execute Matrix Multiplication
//				Given the choice between OpenCL global and local memory
//				you can subdivide matrices to perform parallel execution	
//
/****************************************************************************************/


// OpenCL Matrix Multiplication Kernel
__kernel void sgemm(__global float* C,
					const __global float* A,
					const __global float* B,
					const int dim) {
					  
					  
#if LOCAL_MEM

		// Block index
    	int bx = get_group_id(0);
    	int by = get_group_id(1);
    	
    	// Thread index
    	int tx = get_local_id(0);
    	int ty = get_local_id(1);
    	
    	// Index of the first sub-matrix of A processed by the block
    	int aBegin = dim * BLOCK_SIZE * by;
    	
    	// Index of the last sub-matrix of A processed by the block
    	int aEnd   = aBegin + dim - 1;
    	
    	// Step size used to iterate through the sub-matrices of A
    	int aStep  = BLOCK_SIZE;
 
    	// Index of the first sub-matrix of B processed by the block
    	int bBegin = BLOCK_SIZE * bx;
 
    	// Step size used to iterate through the sub-matrices of B
    	int bStep  = BLOCK_SIZE * dim;
 
   	 	// Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    	float Csub = 0.0;
    	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){

        	// Declaration of the local memory array As used to store the sub-matrix of A
        	__local float As[BLOCK_SIZE][BLOCK_SIZE];
 
        	// Declaration of the local memory array Bs used to store the sub-matrix of B
        	__local float Bs[BLOCK_SIZE][BLOCK_SIZE];
 
        	// Load the matrices from global memory to local memory; each thread loads
        	// one element of each matrix
        	As[ty][tx] = A[a + dim * ty + tx];
        	Bs[ty][tx] = B[b + dim * ty + tx];
 
        	// Synchronize to make sure the matrices 
        	// are loaded
        	barrier(CLK_LOCAL_MEM_FENCE);
 
        	// Multiply the two matrices together;
        	// each thread computes one element
        	// of the block sub-matrix
        	for (int k = 0; k < BLOCK_SIZE; ++k)
            	Csub += As[ty][k] * Bs[k][tx];
 
        	// Synchronize to make sure that the preceding
        	// computation is done before loading two new
        	// sub-matrices of A and B in the next iteration
        	barrier(CLK_LOCAL_MEM_FENCE);
 
    	}
 
    	// Write the block sub-matrix to device memory;
    	// each thread writes one element
    	int c = dim * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    	C[c + dim * ty + tx] = Csub;
    	
    	
#else 
    
    	const int globalRow = get_global_id(0); 
   		const int globalCol = get_global_id(1);
    	 
    	// value stores the element that is 
		// computed by the thread
   		float acc = 0.0;
   		for (int k = 0; k < dim; k++){
      		float elementA = A[globalRow * dim + k];
      		float elementB = B[k * dim + globalCol];
      		acc += elementA * elementB;
   		}
   		
   		// Store the final result in C
    	C[globalRow*dim + globalCol] = acc;
    
#endif
    
    // Store the final result in C
    //C[globalRow*dim + globalCol] = acc;
}