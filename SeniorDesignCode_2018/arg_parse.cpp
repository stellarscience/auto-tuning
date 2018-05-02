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
//	File Name:   arg_parse.cpp
//	
//	Purpose: 	This file contains the parser args functions that gives the executable
//				program different options aside from the base functionality.
//
//				The base functionality will prompt the user for matrix dimensions, 
//				which OpenCL memory to use, and the block size to partition the work.
//
//				Flag -h will display the help message
//
//				Flag -l will execute the devInfo - OpenCL device query
//
//				Flag -g will obtain samples to be used in the random forest
//
//				Flag -m will perform basic matrix multiplication on CPU on OpenCL
//
//				Flag -r will execute the random forest python script
//
/****************************************************************************************/


#include "devInfo.hpp"
#include "host.hpp"
#include "arg_parse.hpp"


static const char* help =
"Options: \n \
-h			Display this messages and exit \n \
-g 			Obtain samples for Random Forest predictions  \n \
-l			List all available OpenCL Devices in detail and exit \n \
-m 			Perform basic matrix multiplication on CPU no OpenCL, \n \
				report execution time, and exit \n \
-r			Execute Random Forest Python Script and exit \n \
\n";

void print_help(int argc, char** argv){
    printf("\nUseage: %s [options]\n\n%s", argv[0], help);
    exit(0);
}


double basic_matrix(){

	int size = 0;
	srand(2018);

	std::cout << "Enter Data Size (1,2,4,...2048): ";
	std::cin  >> size;
	
	float** matxA = new float* [size];
	float** matxB = new float* [size];
	float** matxC = new float* [size];
	for (int i = 0; i < size; i++) {
		matxA[i] = new float[size];
		matxB[i] = new float[size];
		matxC[i] = new float[size];
	}
	
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			matxA[i][j] = rand()/ (float)RAND_MAX;
		}
	}
	
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			matxB[i][j] = rand()/ (float)RAND_MAX;
		}
	}
	
	clock_t clk_start, clk_end;
	
	printf("Running matrix multiplication for matrices A (%i x %i) and B (%i x %i) ...\n",
		size, size, size, size);
	clk_start = clock();
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			matxC[i][j] = 0;
			for (int k = 0; k < size; k++){
				matxC[i][j] += matxA[i][k] * matxB[k][j];
			}
		}
	}
	clk_end = clock();
	
	double time = (clk_end - clk_start)/double(CLOCKS_PER_SEC);
	
	printf("Elapsed Time is (sec): %f\n", time);
	printf("Running Time is: %.3f milliseconds\n", time*1000);
	
	// Display Matrices
	printf("\n	Matrix A \n==========================\n");
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			printf("%03.2f\t ", matxA[i][j]);
		}
		printf("\n");
	}
	
	printf("\n	Matrix B \n==========================\n");
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			printf("%03.2f\t ", matxB[i][j]);
		}
		printf("\n");
	}
	
	printf("\n	Matrix C \n==========================\n");
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			printf("%03.2f\t ", matxC[i][j]);
		}
		printf("\n");
	}
	
	//Free
	for (int i = 0; i < size; ++i) {
		delete [] matxA[i];
		delete [] matxB[i];
		delete [] matxC[i];
	}
	delete [] matxA;
	delete [] matxB;
	delete [] matxC;
	
	return time;
	
}

void call_python(int argc, char** argv){
	
	//Use System Call to Open Python3 Script
	system("python3 plotspace.py");
	exit(0);
}


void generate_samples(int argc, char** argv){
	
	// Set CSV File
	std::string filename;
	filename = "kernel_dataset.csv";
	std::ofstream csv;
	csv.open(filename);
	
	//Ask for size of dataset
	int sample_size;
	std::cout << "How many samples do you want for the dataset?: ";
	std::cin  >> sample_size;
	
	// Ask for size of matrix
	int mtx_dim;
	std::cout << "What is the X and Y dimensions that you want for the Matrices?: ";
	std::cin  >> mtx_dim;
	
	// Inputs (Parameter Space)
	int x_set1 	 =  mtx_dim;	//Matrix Dimension
	int x_set2[] = 	{0,1};		//Local Memory 
	int x_set3[] = 	{1,2,4,8,16}; // Block Size Depends of # of Compute Units
	
	// Size of Input Sets
	int set2_size = sizeof(x_set2)/sizeof(int);
	int set3_size = sizeof(x_set3)/sizeof(int);
	
	// Display Values False - Only Want Execution Samples
	int display = 0;
	
	// Set the seed of the pseudorandom number generator
	srand(2018);
	
	//Generate the Headers for the CSV Table
	csv << "Time"  << ",";
	csv << "Matrix_Dim" << ",";
	csv << "Local_Mem" << ",";
	csv << "Block_Size" << "\n";
	
	//Main Loop
	for(int i = 0; i < sample_size; i++){
		
		//Test for Inputs Sets
		int x1 = x_set1;
		int x2 = x_set2[rand()%set2_size];
		int x3;
		if (x2 == 0){
			x3 = x_set3[0];
		}
		else {
			x3 = x_set3[rand()%set3_size];
		}
		
		// Display the input for following iteration	
		printf("Iteration %d\n", i);
		printf("	Inputs:  [%d, %d, %d]\n", x1, x2, x3);

		// Execute the Kernel from Host Code and Obtain the Execution Time
		double kernel_time = host(x1,x2,x3, display);
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
	std::cout << "Success! Results are saved in " << filename << "." << std::endl;
}


void parse_args(int argc, char** argv){

	int c;
	while ( (c = getopt(argc, argv, "abcdefghijklmnopqrstuvwxyz")) != -1){
		switch (c) {
			case 'h':
				print_help(argc, argv);
				break;
			case 'g':
				// Samples Function to Random Forest Usage
				generate_samples(argc, argv);
				exit(1);
				break;
			case 'l':
				//Function taken from devInfo.hpp
				devicequery();
				exit(1);
				break;
			case 'm':
				//Function for basic matrix multiplication
				basic_matrix();
				exit(1);
				break;
			case 'r':
				//Execute Random Forest Python Script
				call_python(argc, argv);
				exit(1);
				break;
			default:
				std::cout << "Please use : '" << argv[0] << " -h' for help" << std::endl;
				abort();
				break;
		}
	}
}




