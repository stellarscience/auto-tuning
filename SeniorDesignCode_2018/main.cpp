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
//	File Name:	main.cpp
//	Function: 	main()
//	
//	Purpose:	Execute the program. The programs default functionality is to ask the user
//				what size input matrices A and B to use and what OpenCL host/kernel 
//				parameters to use. The program then passes control to host.cpp to perform
//				matrix multiplication on A and B. The result is displayed matrix C, and 
//				the execution time in milliseconds of the kernel matrix multiplication.
//
//				The program also has additional functionality that can be called through
//				the terminal by ./program [-options] where [-options] are execution flags.
//				For example ./program -h will provide the user helpful information about
//				other features of the program like: 
//					- OpenCL Device Query
//					- Basic CPU Matrix Multiplication for comparison
//					- Generating Kernel Parameter/Execution Datasets for Random Forest
//					- Executing Python Random Forest scripts
//
/****************************************************************************************/

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
#include <OpenCL/opencl.h>

#include "devInfo.hpp"
#include "host.hpp"
#include "arg_parse.hpp"

using namespace std;

//Function to display the matrices
void printMatrix(float* buffer, int row, int column){
	
	for(int i = 0; i < row; i++){
		for(int j = 0; j < column; j++){			
			printf("%03.2f\t ", buffer[i*column+j]);
		}
		printf("\n");
	}
	printf("\n");
	
}

int main(int argc, char** argv){

	// Parse Program Arguments [options]
	parse_args(argc, argv);

	// Set CSV File
	string filename;
	filename = "actual.csv";
	ofstream csv;
	csv.open(filename);
	
	// Inputs (Parameter Space)
	int mtx_dim;
	int local_mem;
	int block_size;
	
	// Set Seed
	srand(2018);
	
	//Generate the Headers for the CSV Table
	csv << "Time"  << ",";
	csv << "Matrix_Dim" << ",";
	csv << "Local_Mem" 	<< ",";
	csv << "Block_Size" << "\n";
	
	//Ask for size of dataset
	int sample_size;
	cout << "How many samples do you want for the dataset?: ";
	cin  >> sample_size;
		
	// Ask for size of matrix
	cout << "What is the X and Y dimensions that you want for the Matrices?: ";
	cin  >> mtx_dim;
	cout << " Matrix A and B are both (" << mtx_dim << ") x (" << mtx_dim << ") " << endl;
	
	// Ask for local or global memory
	cout << "Execute on local memory? " << endl;
	cout << "	>>> Enter 1 for yes, Enter 0 for no: ";
	cin  >> local_mem;
	
	if(local_mem){
		// Ask for block size
		cout << "Enter block size (must be less than matrix dimensions): " << endl;
		cin  >> block_size;
		if (block_size > mtx_dim){
			cerr << "Error: Block_Size must be less than Matrix Dimensions! " << endl;
			block_size = floor(block_size/mtx_dim);
		}
	}
	else{
		block_size = 1;
	}
	
	// Set display to True
	int display = 0;
	cout << "Do you want to display the matrices on the terminal?" << endl;
	cout << "	>>> Enter 1 for yes, Enter 0 for no: ";
	cin  >> display;
	
	//Main Loop
	for(int i = 0; i < sample_size; i++){
		
		//Test for Inputs Sets
		int x1 = mtx_dim;
		int x2 = local_mem;
		int x3 = block_size;
		
		// Display the input for following iteration	
		printf("Iteration %d\n", i);
		printf("	Inputs:  [%d, %d, %d]\n", x1, x2, x3);

		// Execute the Kernel from Host Code and Obtain the Execution Time
		double kernel_time = host(x1,x2,x3,display);
		printf("	Outputs (ms): [%.3f]\n", kernel_time);
		
		// Output the results to CSV file
		csv << kernel_time << ",";
		csv << x1 << ",";
		csv << x2 << ",";
		csv << x3 << "\n";
		
	}
	
	//Close CSV File
	csv.close();
	
	// Display Successful Execution
	cout << "Success! Results are saved in " << filename << "." << endl;

	return 0;
}



