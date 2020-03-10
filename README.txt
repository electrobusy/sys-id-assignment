* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* * * READ THIS CAREFULLY IF YOU WOULD LIKE TO UNDERSTAND THE THE STRUCTURE OF THE CODE GIVEN IN THIS PROJECT * * *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

This document contains the information and organisation of all the MATLAB scripts developped for this assignment. 

Most of the functions were not used directly to obtain the results and most of the work was done as EXTRA. Therefore, due to page limit in the actual report, an overview is given here so that anyone can use these functions for its own purpose and try for different data sets.

Cheers,
Rohan 

-----> PART 1: '2_Kalman' folder

Main script: kalman_ex_3_4.m -> It allows to show the results for the EKF and IEKF in order to remove the noise and bias of the data obtained from the sensors and yields a .mat file in '2_Par_Est' and '3_4_NN' folders. 

Other scripts: They are in 'Extra Functions' folder -> These are functions adapted from scripts of the Kalman Filter algorithm given in the course page.

-----> PART 2: '2_Par_Est' folder

Main script: least_squares_ex_5_6_7.m -> Using the filtered data from the Kalman Filter, it uses a polynomial model to create a mapping between the I/O data -> C_m = C_m(alpha,beta). Running this script, can choose the data set (EKF or IEKF) and then see the results of the reconstruction using the OLS, WLS and RLS algorithms. He/She can also observe validation metrics of this reconstruction. 

Other scripts: 
	ord_least_squares.m -> Computes the OLS algorithm
	wei_least_squares.m -> Computes the WLS algorithm (not shown in the report)
	rec_least_squares.m -> Computes the RLS algorithm (not shown in the report)
	perm.m 		    -> Gives the factorial of a number
	reg_matrix.m        -> Constructs a regression matrix
	plot_Cm.m 	    -> Plots the reconstructed data

-----> PARTS 3 and 4: '3_4_NN' folder

--- Given by the Professor: SimNet.m

--- Main scripts: 
	RBF_ex_3_1.m 	     		-> Solves OLS algorithm applied to a RBF-NN (+ sensistivity analysis)
	RBF_ex_3_2.m			-> Applies LM training algorithm for RBF-NN
	RBF_ex_3_3_cross_val.m		-> Number of neurons optimization - RBF-NN + OLS - using the cross-validation algorithm
	RBF_ex_3_3_OLS_cross_val.m 	-> Number of neurons optimization - RBF-NN + LM - using the cross-validation algorithm
	FF_ex_4_1.m			-> Applied Backpropagation algorithm for FF-NN, using stepest gradient descent
	FF_ex_4_2.m			-> Applies LM training algorithm for FF-NN
	FF_ex_4_3_cross_val.m		-> Number of neurons optimization - FF-NN + LM - using the cross-validation algorithm
	FF_ex_4_4.m			-> Approximation power comparison of different models obtained from RBF-NN, FF-NN and OLS polynomial.

(...The following scripts are functions...)
	createNet.m 			-> Function that creates the neural network
	plot_RBF_inputs.m 		-> Plots the input dataset whose datapoints are divided by training/validation/testing + location of the centroids
	plot_FF_inputs.m 		-> Plots the input dataset whose datapoints are divided by training/validation/testing
	BackPropDerMat.m 		-> Function that contains the gradients of FF and RBF neural networks for the backpropagation algorithm (DONE IN VECTORIZED MANNER FOR ALL DATAPOINTS)
	BP_update.m 			-> Performs the update after received the backpropagation update law
	LMDerMat.m 			-> Function that contains the gradients of FF and RBF neural networks for the LM algorithm (DONE IN VECTORIZED MANNER FOR ALL DATAPOINTS)
	LM_update.m 			-> Performs the update after received the LM update law

NOTE: In the above scripts, for the backpropagation and LM algorithm the "Adaptive Learning" procedure is used. When the script contains the tag "_Fixed" then it uses "Fixed Learning". 
	
--- Other scripts:
	RBF_3_2_Fixed.m			-> Applies LM training algorithm for RBF-NN using Fixed Learning
	RBF_3_2_BP.m 			-> Applies backpropagation training algorithm for RBF-NN (using Adaptive Learning)
	RBF_3_2_BP_Fixed.m		-> Applies backpropagation training algorithm for RBF-NN using Fixed Learning	
	RBF_ex_3_3.m			-> Number of neurons optimization - RBF-NN + LM - using common division of the data-set by percentage
	FF_ex_4_1_Fixed.m		-> Applies backpropagation training algorithm for RBF-NN using Fixed Learning
	FF_ex_4_2_Fixed.m		-> Applies LM training algorithm for FF-NN using Fixed Learning 
	FF_ex_4_3_BP.m			-> Number of neurons optimization - FF-NN + backprogragation - using the cross-validation algorithm
	FF_ex_4_4.m 			-> Number of neurons optimization - FF-NN + LM - using common division of the data-set by percentage

	BackPropDer.m 			-> Function that contains the gradients of FF and RBF neural networks for the LM algorithm (CAN ONLY RECEIVE ON DATAPOINT) -> Usefull for Stochastic Gradient Descent
	LMDer.m 			-> Function that contains the gradients of FF and RBF neural networks for the LM algorithm (CAN ONLY RECEIVE ON DATAPOINT) -> Usefull for Stochastic Gradient Descent
	Experiment_extra_stuff 		-> You can see a preliminary code of the Stochastic Gradient Descent and Mini-Batch Gradient Descent (but the code needs to be adapted)

--- Code used to make tests with simulated data obtained from simpler functions:

	NN_BP_Fixed.m			-> Simulates backpropagation algorithm using Fixed Learning
	NN_BP_Adaptive.m		-> Simulates backpropagation algorithm using Adaptive Learning
	NN_LM_Fixed.m 			-> Simulates LM algorithm using Fixed Learning
	NN_LM_Adaptive.m		-> Simulates LM algorithm using Adaptive Learning
	NN_LM_Fixed_OPT.m 		-> Simulates the optimization of number of neurons using the LM algorithm + Fixed Learning
	NN_LM_Adaptive_OPT.m		-> Simulates the optimization of number of neurons using the LM algorithm + Adaptive Learning
	