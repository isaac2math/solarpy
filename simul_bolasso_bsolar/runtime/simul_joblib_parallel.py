import numpy as np

from simulator        import simul
from bsolar_parallel  import bsolar
from tqdm             import tqdm

##########################################
# define the class 'simulation_function' #
##########################################
'''
this class is used to compute average runtime of bsolar (lars) and bsolar (coordinate descent) under customized Joblib parallel scheme:

Check this before you run the code:
Plz check if you have 'joblib', 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install joblib scikit-learn numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install joblib scikit-learn numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. we use 'numpy' for matrix computation and random variable generation;
    2. for 'simulator', 'solar' and 'costcom', plz see 'simulator.py', 'solar.py' and 'costcom.py' for detail;
    3. 'tqdm' is used to construct the progress bar;
    4. we use 'matplotlib' to plot all figures;

Inputs:
     1. X and Y          : the inputs and output of regression
     2. sample_size      : the total sample size we generate for cv-lars-lasso, cv-cd and solar
     3. n_dim            : the number of total variables in X
     4. n_info           : the number of informative variables in data-generating process
     5. n_repeat_solar   : the number of subsamples in solar
     6. n_repeat_bsolar  : the number of subsamples in bsolar
     7. n_repeat_dbsolar : the number of subsamples in dbsolar
     8. num_rep          : the number of repeatitions in our simulation (200 in paper)
     9. step_size        : step size for tuning the value of c for solar;
    10. one_thre         : to control whether bolasso is bolasso-H or bolasso-S
    11. rnd_seed         : the Numpy random seed for replication;

Outputs:
    There is no output for this class. The average runtime will be computed and displayed using 'tqdm' in Jupyter Lab/Notebook.
'''

class simul_func:

    def __init__(self, sample_size, n_dim, n_info, n_repeat_solar, n_repeat_bsolar, num_rep, step_size, rnd_seed):
    ##for convinience, we define the common variable (variables we need to use multiple times) in the class as follows (xxxx as self.xxxx)

        #sample size
        self.sample_size = sample_size
        #the number of total variables in X
        self.n_dim = n_dim
        #number of non-zero regression coefficients in data-generating process
        self.n_info = n_info 
        #the number of subsamples in solar         
        self.n_repeat_solar  = n_repeat_solar 
        #the number of subsamples in bsolar
        self.n_repeat_bsolar = n_repeat_bsolar
        #the number of repeatitions in simulation  
        self.num_rep = num_rep
        #step size for tuning the value of c for solar    
        self.step_size = step_size
        #the random seed for reproduction
        self.rnd_seed = rnd_seed        

    #compute 200 repeats of bsolar (lars) for average runtime
    def simul_bsolar(self):
        
        #the bsolar-3 variable stack of 200 repeats
        bsolar_Q_opt_c_stack  = list() 
        
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

            #2. compute bsolar3 on the sample
            #2a. call the class 'bsolar' from 'bsolar.py'
            trial2 = bsolar( X, Y, self.n_repeat_solar, self.n_repeat_bsolar, self.step_size)
            #2b. compute bsolar
            bsolar_coef_H, bsolar_coef_S, Qc_list, Q_opt_c_H, Q_opt_c_S= trial2.fit()
            #2c. save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by bsolar-3' in 200 repeats)
            bsolar_Q_opt_c_stack.append(Q_opt_c_H)

    #compute 200 repeats of bsolar (coordinate descent) for average runtime
    def simul_bsolar_cd(self):
        
        #the bsolar-3 variable stack of 200 repeats
        bsolar_Q_opt_c_stack  = list() 
        
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

            #2. compute bsolar3 on the sample
            #2a. call the class 'bsolar' from 'bsolar.py'
            trial2 = bsolar( X, Y, self.n_repeat_solar, self.n_repeat_bsolar, self.step_size)
            #2b. compute bsolar
            bsolar_coef_H, bsolar_coef_S, Qc_list, Q_opt_c_H, Q_opt_c_S= trial2.fit_cd()
            #2c. save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by bsolar-3' in 200 repeats)
            bsolar_Q_opt_c_stack.append(Q_opt_c_H)
        

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
    n_repeat_solar   = 10
    n_repeat_bsolar  = 3
    step_size        = -0.02
    num_rep          = 2
    rnd_seed         = 1
    plot_on          = False

    trial = simul_func(sample_size, n_dim, n_info, n_repeat_solar, n_repeat_bsolar, num_rep, step_size, rnd_seed)

    trial.simul_bsolar_cd()
