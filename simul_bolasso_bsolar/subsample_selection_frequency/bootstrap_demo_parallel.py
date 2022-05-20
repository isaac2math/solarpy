import numpy             as np
import matplotlib.pyplot as plt
import warnings

#warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
#warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from bsolar           import bsolar
from bolasso_parallel import bolasso
from simulator        import simul

##############################################
# define the class 'bsolar and bolasso demo' #
##############################################

'''
this class is used to report the comparison of the subsample selection frequency between bsolar and bolasso

Check this before you run the code:
Plz check if you have 'joblib', 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install joblib scikit-learn numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install joblib scikit-learn numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. from scikit-learn, we call 'LassoLarsCV' for boostrap lasso;
    2. we use 'numpy' for matrix computation and random variable generation;
    3. for 'simulator',  plz see 'simulator.py' for detail;
    4. 'tqdm' is used to construct the progress bar;

Inputs:
    1. X and Y          : the inputs and output of regression
    2. sample_size      : the total sample size we generate
    3. n_dim            : the number of total variables in X
    4. n_info           : the number of informative variables in data-generating process
    5. n_repeat_solar   : the number of subsamples that solar generates
    6. n_repeat_bsolar  : the number of subsamples that bsolar generates
    7. n_repeat_bolasso : the number of subsamples that bolaaso generates
    8. rnd_seed         : the random seed
    9. plot_on          : binary, whether the plot will be saved as pdf

Outputs:
    1. Qc_list_bsolar   : the full list of subsample selection frequency for bsolar
    2. Qc_list_bolasso  : the full list of subsample selection frequency for bolasso
    
'''

class one_shot_simul:

    def __init__(self, sample_size, n_dim, n_info, n_repeat_solar, n_repeat_bsolar, step_size, rnd_seed, plot_on):
    ##for convinience, we define the common variable (variables we need to use for each of the following functions) in the class as follows (the common variable is defined as self.xxxx)

        #sample size
        self.sample_size = sample_size
        #the number of total variables in X
        self.n_dim = n_dim
        #the number of informative variables in data-generating process
        self.n_info = n_info
        #the number of subsamples in solar
        self.n_repeat_solar = n_repeat_solar
        #the number of subsamples in bsolar
        self.n_repeat_bsolar = n_repeat_bsolar
        #step size for tuning the value of c for solar;
        self.step_size = -0.02
        #the random seed
        self.rnd_seed = rnd_seed
        #the maximum value of c in its grid search (for plotting only)
        self.q_start = 1
        #the minimum value of c in its grid search (for plotting only)
        self.q_end = 0.1
        #step size of c in its grid search (for plotting only)
        self.q_step = -0.02

    ##compute bsolar and bolasso for subsample selection frequency demonstration
    def simul_func(self):

        #1. control the random seed for reproduction
        np.random.seed(self.rnd_seed)

        #2. call class 'simul' from 'simulator.py' to simulate data
        trial0 = simul(self.sample_size, self.n_dim, self.n_info)
        #3. generate X and Y
        X, Y = trial0.data_gen()

        #3. call class 'bsolar' from 'bsolar.py'
        trial2 = bsolar( X, Y, self.n_repeat_solar, self.n_repeat_bsolar, self.step_size)
        #4. compute bsolar
        _, _, Qc_list_bsolar, _, _ = trial2.fit()

        #5. call class 'bolasso' from 'bolasso.py'
        trial3 = bolasso( X, Y, 256)
        #6. compute bolasso
        _, _, Qc_list_bolasso, _, _ = trial3.fit()

        return Qc_list_bsolar, Qc_list_bolasso

    ##plot the full list of subsample selection frequency for each variable in bsolar
    def q_list_bsolar(self, Qc_list_bsolar):

        #1. concatenate Qc_list into a matrix
        var_mark_plot = np.concatenate(Qc_list_bsolar)
        #2. compute the value of c for each Q(c) and the corresponding variables in each Q(c)
        var_index, counts = np.unique(var_mark_plot, return_counts=True)

        var_index_ordered = [x for _,x in sorted(zip(counts,var_index))]

        var_plot = var_index_ordered[::-1]

        cou_plot = np.sort(counts)[::-1] / ((self.q_end - self.q_start)/self.q_step)

        var_plot = [ 'X' + str(i) for i in var_plot]


        #3. print the list of variables with different value of c

        var_loc_list = list()
        var_q_list   = list()

        q_value_list = np.unique(cou_plot)[::-1]

        i = 1

        for j in q_value_list:

            ans_ind = np.where([cou_plot == j])[1]
            ans_var = [var_plot[i] for i in ans_ind]

            var_loc_list.append(ans_ind)
            var_q_list.append(ans_var)

            print('selection frequency >= ',j)
            print(var_q_list[:i])

            i += 1

    ##plot the full list of subsample selection frequency for each variable in bolasso
    def q_list_bolasso(self, Qc_list):

        # 1. concatenate Qc_list into a matrix
        var_mark_plot = np.concatenate(Qc_list)
        # 2. compute the value of c for each Q(c) and the corresponding variables in each Q(c)
        var_index, counts = np.unique(var_mark_plot, return_counts=True)

        var_index_ordered = [x for _, x in sorted(zip(counts, var_index))]

        var_plot = var_index_ordered[::-1]

        cou_plot = np.sort(counts)[::-1] * 0.01 + 0.1

        var_plot = ['X' + str(i) for i in var_plot]

        # 3. print the list of variables with different value of c

        var_loc_list = list()
        var_q_list = list()

        q_value_list = np.unique(cou_plot)[::-1]

        i = 1

        for j in q_value_list:

            ans_ind = np.where([cou_plot == j])[1]
            ans_var = [var_plot[i] for i in ans_ind]

            var_loc_list.append(ans_ind)
            var_q_list.append(ans_var)

            print('selection frequency >= ', j)
            print(var_q_list[:i])

            i += 1


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
    n_repeat_solar  = 10
    n_repeat_bsolar = 3
    step_size       = -0.02
    rnd_seed        = 0
    plot_on         = False

    np.random.seed(0)

    #generate X and Y
    trial = one_shot_simul(sample_size, n_dim, n_info, n_repeat_solar, n_repeat_bsolar, step_size, rnd_seed, plot_on)

    #train solar
    Qc_list_bsolar, Qc_list_bolasso = trial.simul_func()

    print('############# the nested variable list of bsolar #############')
    trial.q_list_bsolar(Qc_list_bsolar)

    print('############# the nested variable list of bolasso #############')
    trial.q_list_bolasso(Qc_list_bolasso)
