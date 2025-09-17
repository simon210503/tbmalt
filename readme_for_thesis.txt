The repository is based on tbmalt and uses a version after #86 where the repulsive feed has
been refactored. It was changed in the MSE and L1 error slightly to allow for not normalized 
weights.


The following files/directories are important for the thesis:

traintest.py --> main script for full training and testing cycles 
(some executions of the functions including the seeds used might be shown 
only in previous commits)

plots.py --> all plotting functions 
(executions of the functions and seeds might be visible only in previous commits)

training.py --> training process and early stopping condition

testing.py --> testing process 

new_feeds.py --> definition of repulsive potentials

plots/ --> includes all plots used in the thesis

logs/ --> includes the logs of the runs used in the thesis

The following information are found in the log files:
result_0.txt --> datapoints used in training
result_1.txt --> datapoints used in testing
result_2.txt --> MSE loss during training, at each epoch
result_3.txt --> the params dictionary used to calculate DFTB formation energies
result_4.txt --> target formation energies in testing
result_5.txt --> model formation energies in testing
result_6.txt --> target formation energies in training
result_7.txt --> model formation energies in training
result_8.txt --> final parameter alpha
result_9.txt --> final parameter Z
result_10.txt --> MSE in testing
result_11.txt --> MAE in testing