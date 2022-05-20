import numpy             as np
import matplotlib.pyplot as plt

from solar_parallel import solar
from simulator      import simul

##########################################
# define the class 'the simulation_plot' #
##########################################

'''
this class is used for plotting the result of the demonstration simulation in this folder:

Check this before you run the code:
Plz check if you have 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install scikit-learn numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install scikit-learn numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. from scikit-learn, we call 'LassoLarsCV' and 'LassoCV' for cv-lars-lasso and cv-cd respectively;
    2. we use 'numpy' for matrix computation and random variable generation;
    3. for 'simulator_ic', 'solar' and 'costcom', plz see 'simulator_ic.py', 'solar.py' and 'costcom.py' for detail;
    4. 'tqdm' is used to construct the progress bar;
    5. we use 'matplotlib' to plot all figures;

Inputs:
    1. X and Y     : the inputs and output of regression
    2. sample_size : the total sample size we generate for cv-lars-lasso, cv-cd and solar
    3. n_dim       : the number of total variables in X
    4. n_info      : the number of informative variables in data-generating process
    5. n_repeat    : the number of subsamples in solar
    6. num_rep     : the number of repeatitions in Simulation
    7. step_size   : (grid search)step size for tuning the value of c for solar;
    8. rnd_seed    : the random seed
    9. plot_on     : binary, whether the plot will be saved as pdf

Outputs:
    1. solar_coef   : the solar regression coefficients (defined at the end of Algorithm 2 and 3);
    2. opt_c        : value of c* in solar;
    3. test_error   : the list of test errors for tuning the value of c;
    4. Qc_list      : the nest sets of Q(c), for all value of c from 1 to 0;
    5. la_list      : number of variables selected by CV-lars-lasso;
    6. la_vari_list : the indices of variables selected by CV-lars-lasso;
    7. cd_list      : number of variables selected by CV-cd;
    8. cd_vari_list : the indices of variables selected by CV-cd;

In each round of subsampling, we randomly take out 1/K points out of the sample and make the rest as the subsample in this round

As competitors, we use X and Y for LassoLarsCV (called CV-lars-lasso in paper) and LassoCV (called CV-cd in paper) estimation, which relies on 10-fold CV.
'''

class one_shot_simul:

    def __init__(self, sample_size, n_dim, n_info, n_repeat, step_size, rnd_seed, plot_on):
    ##for convinience, we define the common variable (variables we need to use for each of the following functions) in the class as follows (the common variable is defined as self.xxxx)

        self.sample_size = sample_size #sample size
        self.n_dim       = n_dim       #the number of total variables in X
        self.n_info      = n_info      #the number of informative variables in data-generating process
        self.n_repeat    = n_repeat    #the number of subsamples in solar
        self.step_size   = step_size   #step size for tuning the value of c for solar;
        self.rnd_seed    = rnd_seed    #the random seed
        self.q_start     = 1           #the maximum value of c in its grid search (for plotting)
        self.q_end       = 0.1         #the minimum value of c in its grid search (for plotting)
        self.q_step      = -0.02       #step size of c in its grid search (for plotting)
        self.plot_on     = plot_on     #whether the plot will be saved as pdf

    ##compute solar, cv-lar-lasso and cv-cd for Demonstration Simulation in Section 3
    def simul_func(self):

        #1. control the random seed for reproduction
        np.random.seed(self.rnd_seed)

        #2. call class 'simul' from 'simulator.py' to simulate data
        trial1 = simul(self.sample_size, self.n_dim, self.n_info)
        #3. generate X and Y
        X, Y = trial1.data_gen()

        #4. call class 'solar' from 'solar.py'
        trial2 = solar( X, Y, self.n_repeat, self.step_size)
        #5. compute solar, cv-lar-lasso and cv-cd on X and Y
        solar_coef, opt_c, test_error, Qc_list, Q_opt_c, la_list, la_vari_list, cd_list, cd_vari_list = trial2.fit()

        return solar_coef, opt_c, test_error, Qc_list, la_list, la_vari_list, cd_list, cd_vari_list

    ##for solar, plot the corresponding test error of each value of c in its tuning (grid search)
    def q_plot(self, test_error, opt_c):

        #1. control which value of c we want to plot (start from q_start and end at q_end)
        q_value = np.arange(self.q_start, self.q_end, self.q_step)

        f1 = plt.figure()
        #2. scatter plot the value of c and its corresponding test error
        plt.scatter(q_value, test_error, color = 'b', label = 'the c values and their validation errors')
        #3. plot a vertical line at the value of c*: max(opt_c) is because there may be multiple values assigned with the same test error
        plt.axvline(max(opt_c), linewidth = 2.5, color = 'g', ls = '-.', label = 'the optimal c value')

        plt.xlabel('the value of c', fontsize=16)
        plt.ylabel('validation error', fontsize=16)
        plt.ylim(0, 5)
        plt.xlim(0.2, 1.01)
        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), borderaxespad=0., ncol=2, shadow=True)

        if self.plot_on == True:
            f1.savefig("q_plot_one_shot.pdf", bbox_inches='tight')
        plt.show()


    ##return Q(c), for all c from (start from q_start and end at q_end)
    def q_list(self, Qc_list):

        #1. concatenate Qc_list into a matrix
        var_mark_plot = np.concatenate(Qc_list)
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

            print('q_hat value >= ',j)
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

    sample_size = 200
    n_dim       = 100
    n_info      = 5
    n_repeat    = 20
    step_size   = -0.02
    rnd_seed    = 0
    plot_on     = False

    np.random.seed(0)

    #generate X and Y
    trial = one_shot_simul(sample_size, n_dim, n_info, n_repeat,  step_size, rnd_seed, plot_on)

    #train solar
    solar_coef, opt_c, test_error, Qc_list, la_list, la_vari_list, cd_list, cd_vari_list = trial.simul_func()

    #plot test error of each value of c
    trial.q_plot(test_error, opt_c)

    #return Q(c)
    trial.q_list(Qc_list)

    #return variables selected by cv-lars-lasso
    print('variables selected by cv-lars-lasso: ', [ 'X' + str(i) for i in la_vari_list])
    #return variables selected by cv-cd
    print('variables selected by cv-cd: ', [ 'X' + str(i) for i in cd_vari_list])
    #return solar regression coefficients
    print(solar_coef)
