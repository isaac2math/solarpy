import numpy as np
import time
import warnings

from joblib               import Parallel, delayed
from sklearn.linear_model import LassoLarsCV, LinearRegression
from sklearn.exceptions   import ConvergenceWarning
from sklearn              import preprocessing
from importlib.metadata   import version

assert version('scikit-learn') <= '1.2.0', "Please make sure the scikit-learn version <= 1.2.0"

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
warnings.simplefilter(action='ignore', category=FutureWarning)

###############################################################
# define the class of bolasso using Joblib parallel computing #
###############################################################

'''
this class is used to demonstrate the performance of bootstrap lasso using Joblib parallel computing.

Check this before you run the code:
Plz check if you have 'sci-kit learn', 'numpy', 'joblib', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install scikit-learn joblib numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install scikit-learn joblib numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. from scikit-learn, we call 'LassoLarsCV' to compute bootstrap lasso;
    2. we use 'numpy' for matrix computation and random variable generation;
    3. for simulator and costcom, plz see 'simulator.py' and 'costcom.py' for detail;
    4. we use class 'joblib' to control the parallel computation;

Inputs:
    1. X and y          : the inputs and output of regression;
    2. n_repeat_bolasso : the number of subsamples that bolasso generates;
    3. rnd              : the random seed for replication;

Outputs:
    1. bolassoS_coef : the bolasso-S regression coefficients;
    2. bolassoH_coef : the bolasso-H regression coefficients;
    3. Qc_list       : the detailed subsample selection frequency of bolasso;
    4. Q_opt_c_S     : the bolasso-S active set;
    5. Q_opt_c_H     : the bolasso-H active set;

Remarks:
    1. fit    : the function that trains bolasso;
'''

class bolasso:

    def __init__(self, X, Y, n_repeat_bolasso, rnd=0):
        # for convinience, we define the common variable (variables we need to use for each of the following functions) in the class as follows (the common variable is defined as self.xxxx)

        # sample size
        self.sample_size = X.shape[0]
        # the number of subsamples generated in bolasso
        self.n_repeat_bolasso = n_repeat_bolasso
        # the number of total variables in X
        self.n_dim = X.shape[1]
        # randome seeds for parallel computations
        self.rnd = rnd
        # the maximum value of subsample selection frequency for plotting
        self.q_start = 1
        # the minimum value of subsample selection frequency for plotting
        self.q_end = 0.1
        # the sample we generate via data-generating process
        self.X = X; self.y = Y

    def fit(self):

        #1. construct a placeholder called 'qhat_k_container', which is the list of all qhat^k (a binary string representing whether each variable is selected by lasso on subsample k) of each subsample
        qhat_k_container = list()

        #2. train a lasso on each subsample, find out which variable is selected on a given sample and save the corresponding selection result on subsample k as qhat^k

        #parallel computing starts
        # 2a. to make parallel computing replicable, set random seeds
        np.random.seed(self.rnd)
        # 2b. spawn off child seed sequences to pass to child processes.
        seeds = np.random.randint(1e8, size=self.n_repeat_bolasso)

        # 2c. first we define what we do in each stage of the loop
        def loop_fun(self, i, seeds, qhat_k_container):

            # 2c(1). fix random seed for replication
            np.random.seed(seeds[i])

            # 2c(2). randomly choose a bootstrap set of sample points (whose index is 'index_subsample'); 
            # e.g. choosing n points from n points with replacement
            index_subsample = np.random.choice(self.sample_size, self.sample_size, replace=True)
            # 2c(3). based on the index "index_subsample", take the corresponding observations of X as "X_subample"
            X_subsample = self.X[index_subsample]
            # 2c(4). based on the index "index_subsample", take the corresponding observations of Y as "y_subample"
            y_subsample = self.y[index_subsample]

            # 2c(5). change dimension for the Sklearn lasso package.
            y_subsample.shape = (y_subsample.shape[0],)

            # 2c(6). given a subsample, compute lasso and output the active set;
            #standardize training data
            scaler = preprocessing.StandardScaler().fit(X_subsample)
            X_subsample = scaler.transform(X_subsample)
            # call the lasso class and set the number of fold as 10
            trial_1 = LassoLarsCV(cv=10, normalize=False)
            # fit lasso on the subsample
            trial_1.fit(X_subsample, y_subsample)
            # save the lasso active set (indices of variables select by lassso) as 'active'.
            active = trial_1.active_

            # 2c(7). based on the active set of lasso, we compute qhat^k as the binary string of whether each variable is selected by lasso on subsample K
            # we generate 'qhat_k' as a row of zeros;
            qhat_k = np.zeros((1, self.n_dim))
            # if a variable (the ith column in matrix X) is selected by lasso, we change the ith value of qhat_k (the ith column) as 1
            for i in active:

                qhat_k[0, i] = 1

            # we append the result into 'qhat_k_container' as one element of the list
            qhat_k_container.append(qhat_k)

            return qhat_k_container

        # 2d. parallel the whole for-loop using the function we define previously and save the result
        # prefer="processes"  means that we prefer use processes
        # n_jobs=-1           means we use all possible processes
        qhat_k_container = Parallel(n_jobs=-1, prefer="processes")(delayed(loop_fun)(self, i, seeds, qhat_k_container) for i in range(self.n_repeat_bolasso))

        # 3. compute subsample selection frequency for all variables
        # 3a. we transform the list of all q_hat^k ('qhat_k_container') into a matrix ('qhat_k_container_matrix')
        # axis = 0 means we treat each item as a row in matrix;
        # row of the matrix   : the q_hat^k on a given subsample for all variables;
        # column of the matrix: the corresponding value of qhat^k for variable "X_i" on all subsamples;
        qhat_k_container_matrix = np.concatenate(qhat_k_container, axis=0)

        # 3b. compute the the value of qhat for each variable (the subsample selection frequency of each variable)
        # e.g., compute the mean of each column 
        qhat_value = np.mean(qhat_k_container_matrix, axis=0)

        # 3c. set 'Qc_list' as the container for the subsample selection frequencies of all variables, ranking in decreasing order.
        Qc_list = list()
        # 3d. set 'c_seq' as the sequence of subsample selection frequency in bolasso
        q_step = -0.01  #when I need a detailed subsample frequency table, I take it as -0.01; otherwise, I use -0.02 to speed up the computation. This option virtually have no effect on runtime and accuracy on my PC.
        c_seq = np.arange(1, 0.1, q_step)

        # 3e. for each value of c, generate Q(c) --- the set of variables with subsample frequency larger or equal to c;
        for j in c_seq:
            # 3e(1). define 'container' as the placeholder of Q(c) when c == j;
            container = list()

            for i in range(self.X.shape[1]):
                # 3e(2). include all variables into 'container' if their corresponding values in q-hat 
                # (the subsample selection frequency of X_i) are larger or equal to j;
                if (qhat_value[0][i] >= j):

                    container.append(i)
            # 3e(3). append 'container' (Q(c) when c == j) into 'Qc_list' (the container of Q(c) for all value of c);
            Qc_list.append(container)

        # 4. pick the variable that are selected most of the time;
        # 4a. find the active set and save it as 'Q_opt_c';
        # 4a(1). if it is bolasso-H, choose c == 1
        Q_opt_c_H = Qc_list[0]
        # 4a(2). if it is bolasso-S, choose c == 0.9
        Q_opt_c_S = Qc_list[10]

        # 5. output the bolasso-S result (Q_opt_c_S is the active set of bolasso-S)
        # 5a. if Q_opt_c_S is empty, return a zero array and empty active set
        if Q_opt_c_S == []:

            bolassoS_coef = np.zeros([self.n_dim, 1])
        # 5b. otherwise, regress Y onto the selected variables in X (variables in Q_opt_c_S)
        else:
            # 5b(1). call the LinearRegression class;
            OLS_S = LinearRegression()
            # 5b(2). fit OLS of Y to the variables in Q_opt_c_S on X;
            OLS_S.fit(self.X[:, Q_opt_c_S], self.y)
            # 5b(3). set 'bolassoS_coef' (an array of zeros) as the placeholder of bolasso-S regression coefficents
            bolassoS_coef = np.zeros([self.n_dim, 1])
            # 5b(4). put the estimated regression coefficents into their corresponding place of 'bolassoS_coef'
            bolassoS_coef[Q_opt_c_S, 0] = OLS_S.coef_

        # 5c. output the bolasso-H result (Q_opt_c_H is the active set of bolasso-H)
        # if Q_opt_c_H is empty, return a zero array and empty active set
        if Q_opt_c_H == []:

            bolassoH_coef = np.zeros([self.n_dim, 1])
        # 5d. otherwise, regress Y onto the selected variables in X (variables in Q_opt_c_H)
        else:
            # 5d(1). call the LinearRegression class;
            OLS_H = LinearRegression()
            # 5d(2). fit OLS of Y on the variables of Q_opt_c_H in X;
            OLS_H.fit(self.X[:, Q_opt_c_H], self.y)
            # 5d(3). set 'bolassoH_coef' (an array of zeros) as the placeholder of bolasso-H regression coefficents
            bolassoH_coef = np.zeros([self.n_dim, 1])
            # 5d(4). put the estimated regression coefficents into their corresponding place of 'bolassoH_coef'
            bolassoH_coef[Q_opt_c_H, 0] = OLS_H.coef_

        return bolassoS_coef, bolassoH_coef, Qc_list, Q_opt_c_S, Q_opt_c_H

##################################
# test if this module works fine #
##################################

'''
this part is set up to test the functionability of the class above;
you can run all the codes in this file to test if the class works;
when you call the class from this file, the codes (even functions or classes) after " if __name__ == '__main__': " will be ingored
'''

if __name__ == '__main__':

    from simulator import simul

    sample_size = 100
    n_dim = 12
    n_info = 5
    n_repeat_bolasso = 256

    np.random.seed(2)

    # generate X and Y
    trial1 = simul(sample_size, n_dim, n_info)
    X, Y = trial1.data_gen()

    # start timing
    start = time.time()

    # train solar
    trial2 = bolasso(X, Y, n_repeat_bolasso)

    bolassoS_coef, bolassoH_coef, Qc_list, Q_opt_c_S, Q_opt_c_H = trial2.fit()

    # end timing
    end = time.time()

    # print the result
    print('variables that bolassoS selects: ', Q_opt_c_S)

    print('variables that bolassoH selects: ', Q_opt_c_H)
