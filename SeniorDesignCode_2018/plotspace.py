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
#	File Name: 	plotspace.py
#	Function: 	main() - Python3
#
#	Purpose:	This python script is designed to retrieve a CSV dataset about OpenCL 
#				kernel execution times and their kernel parameters. The dataset is then
#				split into a train set for random forest machine learning, and a test set
#				for evaluating the accuracy of the random forest prediction model.
#
##########################################################################################

import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D as mlines
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def main():
	print("foo")
	
	data = pd.read_csv('kernel_dataset.csv')

	data_x = data.iloc[:,1:]	# x - input values are the all N rows and the cols from 1 to N
	data_y = data.iloc[:,0]		# y - output values are the all N rows and the column 0
	
	blk_x = data.iloc[:,3] # x - block size values 
	blk_y = data.iloc[:,0] # y - matrix execution time


	# Plot the Relationship Between Matrix Size and Execution Time
	plt.plot(blk_x, blk_y, 'g^')
	plt.title('Relationship Between Block Size and Matrix Execution Times')
	plt.xlabel('Block Size')
	plt.ylabel('Execution Time (msec)')
	plt.show()

	# Splitting the Original Data Into Training and Testing (Half Training/Half Testing 
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.5)

	# Fit a Random Forest Model
	rf = RandomForestRegressor(n_estimators = 100)
	model = rf.fit(x_train, y_train)

	# Generate Predictions from the Test Set Applied to the Model
	predictions = rf.predict(x_test)

	# The Line/Model
	plt.scatter(y_test, predictions, s=10)
	plt.plot([0,10], [0,10], 'r-')	# Find a way to dynamically change the line
	plt.title("Comparison Between Acutal and Predicted Values")
	plt.xlabel("True Values")
	plt.ylabel("Predictions")
	plt.grid()
	plt.show()

	# Print Accuracy
	print ("Score:", model.score(x_test, y_test))


if __name__ == "__main__":
	main()
