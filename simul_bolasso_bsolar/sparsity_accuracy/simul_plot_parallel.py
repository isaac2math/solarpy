import numpy as np

from simulator        import simul
from bsolar_parallel  import bsolar
from bolasso_parallel import bolasso
from tqdm             import tqdm

#################################################################################
# define the class 'the simulation_plot' using Joblib parallel computing scheme #
#################################################################################
'''
this class is used for plotting the result of sparsity-accuracy comparison

Check this before you run the code:
Plz check if you have 'joblib', 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install scikit-learn joblib numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install scikit-learn joblib numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. from scikit-learn, we call 'LassoLarsCV' for lasso and bolasso;
    2. we use 'numpy' for matrix computation and random variable generation;
    3. for 'simulator', 'solar' and 'costcom', plz see 'simulator.py', 'solar.py' and 'costcom.py' for detail;
    4. 'tqdm' is used to construct the progress bar;
    5. we use 'matplotlib' to plot all figures;

Inputs:
    1. sample_size : the total sample size we generate
    2. n_dim       : the number of total variables in X
    3. n_info      : the number of informative variables in data-generating process
    4. num_rep     : the number of repeatitions in Simulation 3
    5. step_size   : step size for tuning the value of c for solar;
    6. rnd_seed    : the random seed

Outputs:
    1. bsolar3S_Q_opt_c_stack  : the stack of bsolar-3S  active sets in 200 repeats;
    2. bsolar3H_Q_opt_c_stack  : the stack of bsolar-3H  active sets in 200 repeats;
    3. bsolar5S_Q_opt_c_stack  : the stack of bsolar-5S  active sets in 200 repeats;
    3. bsolar5H_Q_opt_c_stack  : the stack of bsolar-5H  active sets in 200 repeats;
    3. bsolar10S_Q_opt_c_stack : the stack of bsolar-10S active sets in 200 repeats;
    3. bsolar10H_Q_opt_c_stack : the stack of bsolar-10H active sets in 200 repeats;
    4. bolassoS_Q_opt_c_stack  : the stack of bolasso-S  active sets in 200 repeats;
    5. bolassoH_Q_opt_c_stack  : the stack of bolasso-H  active sets in 200 repeats;

Remarks:
    1. simul_func() : compute the simulation.
'''

class simul_plot:

    def __init__(self, sample_size, n_dim, n_info, num_rep, step_size, rnd_seed):
    ##for convinience, we define the common variable (variables we need to use multiple times) in the class as follows (xxxx as self.xxxx)

        #define the paras
        self.sample_size     = sample_size     #sample size
        self.n_dim           = n_dim           #the number of total variables in X
        self.n_info          = n_info          #number of non-zero regression coefficients in data-generating process
        self.num_rep         = num_rep         #the number of repeatitions in Simulation 1, 2 and 3
        self.step_size       = step_size       #step size for tuning the value of c for solar
        self.rnd_seed        = rnd_seed        #the random seed for reproduction


    def simul_func(self):
    #compute 200 repeats

        bsolar3S_Q_opt_c_stack  = list() #the bsolar-3S variable stack of 200 repeats
        bsolar3H_Q_opt_c_stack  = list() #the bsolar-3H variable stack of 200 repeats
        bsolar10S_Q_opt_c_stack = list() #the bsolar-10S variable stack of 200 repeats
        bsolar10H_Q_opt_c_stack = list() #the bsolar-10H variable stack of 200 repeats
        bsolar5S_Q_opt_c_stack  = list() #the bsolar-5S variable stack of 200 repeats
        bsolar5H_Q_opt_c_stack  = list() #the bsolar-5H variable stack of 200 repeats
        bolassoS_Q_opt_c_stack  = list() #the bolasso-S variable stack of 200 repeats
        bolassoH_Q_opt_c_stack  = list() #the bolasso-H variable stack of 200 repeats
        
        #to make parallel computing replicable, set random seeds
        np.random.seed(self.rnd_seed)
        # Spawn off 200 child SeedSequences to pass to child processes.
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
            trial2 = bsolar( X, Y, 10, 3, self.step_size)
            #2b. compute bsolar
            _, _, _, Q_opt_c_H, Q_opt_c_S = trial2.fit()
            #2c. save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by solar' in 200 repeats)
            bsolar3S_Q_opt_c_stack.append(Q_opt_c_S)
            bsolar3H_Q_opt_c_stack.append(Q_opt_c_H)

            #3. compute bsolar10 on the sample
            #3a. call the class 'bsolar' from 'bsolar.py'
            trial3 = bsolar( X, Y, 10, 10, self.step_size)
            #3b. compute bsolar
            _, _, _, Q_opt_c_H, Q_opt_c_S= trial3.fit()
            #3c. save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by bsolar' in 200 repeats)
            bsolar10S_Q_opt_c_stack.append(Q_opt_c_S)
            bsolar10H_Q_opt_c_stack.append(Q_opt_c_H)

            #4. compute dbsolar on the sample 
            #4a. call the class 'dbsolar' from 'dbsolar.py'
            trial4 = bsolar( X, Y, 10, 5, self.step_size)
            #4b. compute bolasso
            _, _, _, Q_opt_c_H, Q_opt_c_S = trial4.fit()
            #4c. save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by bolasso' in 200 repeats)
            bsolar5S_Q_opt_c_stack.append(Q_opt_c_S)
            bsolar5H_Q_opt_c_stack.append(Q_opt_c_H)

            #5. compute bolasso on the sample 
            #5a. call the class 'bolasso' from 'bolasso.py'
            trial5 = bolasso(X, Y, 256)
            #5b. compute bolasso
            _, _, _, Q_opt_c_S, Q_opt_c_H = trial5.fit()
            #4c. save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by bolasso' in 200 repeats)
            bolassoH_Q_opt_c_stack.append(Q_opt_c_H)
            bolassoS_Q_opt_c_stack.append(Q_opt_c_S)            

        return bsolar3S_Q_opt_c_stack, bsolar3H_Q_opt_c_stack, bsolar10S_Q_opt_c_stack, bsolar10H_Q_opt_c_stack, bsolar5S_Q_opt_c_stack, bsolar5H_Q_opt_c_stack, bolassoS_Q_opt_c_stack, bolassoH_Q_opt_c_stack


##################################
# test if this module works fine #
##################################

'''
this part is set up to test the functionability of the class above;
you can run all the codes in this file to test if the class works;
when you call the class from this file, the codes (even functions or classes) after " if __name__ == '__main__': " will be ingored
'''

if __name__ == '__main__':

    sample_size     = 100
    n_dim           = 12
    n_info          = 5
    step_size       = -0.02
    num_rep         = 3
    rnd_seed        = 1

    trial = simul_plot(sample_size, n_dim, n_info,  num_rep, step_size, rnd_seed)

    bsolar3S_Q_opt_c_stack, bsolar3H_Q_opt_c_stack, bsolar10S_Q_opt_c_stack, bsolar10H_Q_opt_c_stack, bsolar5S_Q_opt_c_stack, bsolar5H_Q_opt_c_stack, bolassoS_Q_opt_c_stack, bolassoH_Q_opt_c_stack = trial.simul_func()