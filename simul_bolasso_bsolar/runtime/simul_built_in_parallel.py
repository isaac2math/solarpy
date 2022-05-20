import numpy as np

from simulator import simul
from bolasso   import bolasso
from tqdm      import tqdm

##########################################
# define the class 'simulation_function' #
##########################################
'''
this class is used to compute average runtime of bolasso (lars) under the sklearn built-in parallel scheme:

Check this before you run the code:
Plz check if you have 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install scikit-learn numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install scikit-learn numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. from scikit-learn, we call 'LassoLarsCV' for bolasso;
    2. we use 'numpy' for matrix computation and random variable generation;
    3. for 'simulator_ic', 'solar' and 'costcom', plz see 'simulator_ic.py', 'solar.py' and 'costcom.py' for detail;
    4. 'tqdm' is used to construct the progress bar;
    5. we use 'matplotlib' to plot all figures;

Inputs:
    1. X and Y     : the inputs and output of regression
    2. sample_size : the total sample size we generate
    3. n_dim       : the number of total variables in X
    4. n_info      : the number of informative variables in data-generating process
    5. num_rep     : the number of repeatitions in our simulation 
    6. step_size   : step size for tuning the value of c for solar;
    7. one_thre    : to control whether bolasso is bolasso-H or bolasso-S
    8. rnd_seed    : the Numpy random seed for replication;

Outputs:
    There is no output for this class. The average runtime will be computed and display using 'tqdm' in Jupyter Lab/Notebook.
'''

class simul_func:

    def __init__(self, sample_size, n_dim, n_info, num_rep, step_size, rnd_seed):
    ##for convinience, we define the common variable (variables we need to use multiple times) in the class as follows (xxxx as self.xxxx)

        #sample size
        self.sample_size = sample_size
        #the number of total variables in X
        self.n_dim = n_dim
        #number of non-zero regression coefficients in data-generating process
        self.n_info = n_info
        #the number of repeatitions in simulation
        self.num_rep = num_rep
        #step size for tuning the value of c for solar
        self.step_size = step_size 
        #the random seed for reproduction
        self.rnd_seed = rnd_seed
    
    def simul_bolasso(self):
    #compute 200 repeats of bolasso for average runtime

        #the bolasso-S variable stack of 200 repeats
        bolasso_Q_opt_c_stack = list() 
        
        #to make parallel computing replicable, set random seeds
        np.random.seed(self.rnd_seed)
        # Spawn off 10 child SeedSequences to pass to child processes.
        seeds = np.random.randint(1e8, size=self.num_rep)

        ##use for-loop to compute 200 repeats
        #use 'tqdm' in for loop to construct the progress bar
        for i in tqdm(range(self.num_rep)):

            np.random.seed(seeds[i])

            #1.  call the class 'simul' from 'simul.py'
            trial1 = simul(self.sample_size, self.n_dim, self.n_info)
            #2.  generate X and Y in each repeat
            X, Y = trial1.data_gen()

            #6. compute bolassoS on the sample 
            #6a. call the class 'bolasso' from 'bolasso.py'
            trial6 = bolasso(X, Y, 256)
            #4b. compute bolasso
            bolassoS_coef, bolassoH_coef, Qc_list, Q_opt_c_S, Q_opt_c_H = trial6.fit()
            #4c. save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by bolasso' in 200 repeats)
            bolasso_Q_opt_c_stack.append(Q_opt_c_H)
        
##################################
# test if this module works fine #
##################################

'''
this part is set up to test the functionability of the class above;
you can run all the codes in this file to test if the class works;
when you call the class from this file, the codes (even functions or classes) after " if __name__ == '__main__': " will be ingored
'''

if __name__ == '__main__':

    sample_size      = 50
    n_dim            = 5
    n_info           = 2
    step_size        = -0.02
    num_rep          = 2
    rnd_seed         = 1
    plot_on          = False

    trial = simul_func(sample_size, n_dim, n_info, num_rep, step_size, rnd_seed)

    trial.simul_bolasso()