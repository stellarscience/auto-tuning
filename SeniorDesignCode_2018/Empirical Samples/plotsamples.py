##########################################################################################
#
#	Adaptive Auto Tuning of Computations on Heterogeneous Environments
#
#	University of New Mexico
#	Department of Electrical and Computer Engineering
#	Melissa Castillo and Christian Curley
#
#	Sponsor and Technical Mentor: Carlos Reyes - Stellar Science
#
# ----------------------------------------------------------------------------------------
#
#	Last Update: May 1st, 2018
#
#	File Name: 	plotsamples.py
#	Function: 	main() - Python3
#
#	Purpose:	This python script was created to demonstrate the need for kernel auto 
#				tuning. Our Random Forest - Machine Learning component of the project 
#				was unreliable because of dataset was small, and we need more tests on 
#				different data to confidently say the Random Forest was successful.
#
#				Because of our small dataset, we were able to perform the alternative
#				empirical tuning, to demonstrate the best and worst kernels in our
#				tested devices.
#
#				The plotted comparisons show that kernel auto tuning is necessary and the
#				best kernel can generate excellent performance.
#
##########################################################################################


import numpy  as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def show_matrices():
	melgpu_best = pd.read_csv("MelissaGPU_best_time.csv")
	melgpu_worst = pd.read_csv("MelissaGPU_worst_time.csv")
	chrisgpu_best = pd.read_csv("chris_gpu_best.csv");
	chrisgpu_worst = pd.read_csv("chris_gpu_worst.csv");
	naive = pd.read_csv("naive.csv");
	
	best_x = melgpu_best.iloc[:,1]
	best_y = melgpu_best.iloc[:,0]
	
	worst_x = melgpu_worst.iloc[:,1]
	worst_y = melgpu_worst.iloc[:,0]
	
	chris_bestx = chrisgpu_best.iloc[:,1]
	chris_besty = chrisgpu_best.iloc[:,0]
	
	chris_worstx = chrisgpu_worst.iloc[:,1]
	chris_worsty = chrisgpu_worst.iloc[:,0]
	
	naive_x = naive.iloc[:,1]
	naive_y = naive.iloc[:,0]
	
	plt.plot(best_x, best_y, 'r--', label='Intel Iris (Best)')
	plt.plot(worst_x, worst_y, 'b--', label='Intel Iris (Worst)')
	plt.plot(chris_bestx, chris_besty, 'g-', label='Intel HD 4000 (Best)') 
	plt.plot(chris_worstx, chris_worsty, 'm-', label='Intel HD 4000 (Worst)')
	plt.plot(naive_x, naive_y, 'k-', label='Intel i5 (Naive Matrix Mult)')
	plt.title('Kernel Performance For Worst and Best Parameters')
	plt.xlabel('Matrix Dimension')
	plt.ylabel('Execution Time (msec)')
	plt.legend(loc='upper left')
	plt.show() 
	
	plt.plot(best_x, best_y, 'r--', label='Intel Iris GPU')
	plt.plot(chris_bestx, chris_besty, 'g-', label='Intel HD 4000 GPU')
	plt.title('Best Kernels Parameters (GPUs)')
	plt.xlabel('Matrix Dimension')
	plt.ylabel('Execution Time (msec)')
	plt.legend(loc='upper left')
	plt.show()
	
	
def main():
	show_matrices()


if __name__ == "__main__":
	main()
