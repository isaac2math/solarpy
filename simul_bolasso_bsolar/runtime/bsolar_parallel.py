import numpy as np
import time
import warnings

from joblib               import Parallel, delayed
from sklearn.linear_model import LinearRegression
from solar                import solar
from sklearn.exceptions   import ConvergenceWarning
from sklearn              import preprocessing

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

##########################################################
# define the class of bsolar (joblib parallel computing) #
##########################################################

'''
this class is used to demonstrate the performance of bootstrap solar (bsolar) using lars via Joblib parallel computation

Check this before you run the code:
Plz check if you have 'joblib', 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install joblib scikit-learn numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install joblib scikit-learn numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. from scikit-learn, we call 'Lars' to compute solar.
    2. we use 'numpy' for matrix computation and random variable generation;
    3. for simulator, plz see 'simulator.py' for detail;
    4. we use class 'time' to time the computation of solar

Inputs:
    1. X and y         : the inputs and output of regression;
    2. n_repeat_solar  : the number of subsamples that solar generates;
    3. n_repeat_bsolar : the number of subsamples that bsolar generates;
    4. step_size       : the step size of grid search for threshold optimization of subsample selection frequency;

Outputs:
    1. bsolar_coef_H : the bsolar-H regression coefficients;
    2. bsolar_coef_S : the bsolar-S regression coefficients;
    4. Qc_list       : the detailed subsample selection frequency of bsolar;
    5. Q_opt_c_H     : the variable that bsolar-H selects;
    5. Q_opt_c_S     : the variable that bsolar-S selects;

Remarks:
    1. fit    : the function that trains bsolar;
    2. q_list : the plot function that returns the full list of subsample selection frequency for each variable in bsolar;
'''

class bsolar:

    def __init__(self, X, Y, n_repeat_solar, n_repeat_bsolar, step_size, rnd=0):
        # for convinience, we define the common variable (variables we need to use for each of the following functions) in the class as follows (the common variable is defined as self.xxxx)

        # sample size
        self.sample_size = X.shape[0]  
        # the number of subsamples generated in solar
        self.n_repeat_solar = n_repeat_solar 
        # the number of subsamples generated in bsolar
        self.n_repeat_bsolar = n_repeat_bsolar 
        # (grid search) step size for tuning the threshold of subsample selection frequency for bsolar
        self.step_size = -0.02
        # the Numpy random seed for replication
        self.rnd = rnd 
        # the size of each subsample
        self.subsample_size = int(self.sample_size * 0.9)
        # the number of total variables in X
        self.n_dim = X.shape[1]  
        # the maximum value of c in its grid search (for plotting only)
        self.q_start = 1
        # the minimum value of c in its grid search (for plotting only)
        self.q_end = 0.1
        # step size of c in its grid search (for plotting only)
        self.q_step = -0.02
        # the sample we generate via data-generating process
        self.X = X; self.y = Y

    def fit(self):

        #1. construct a placeholder called 'qhat_k_container', which is the list of all qhat^k (a binary string representing whether each variable is selected by solar on subsample k) of each subsample
        qhat_k_container = list()

        #2. train a solar on each subsample, find out which variable is selected on a given sample and save the corresponding selection result on subsample k as qhat^k

        # parallel computing starts
        # 2a. to make parallel computing replicable, set random seeds
        np.random.seed(self.rnd)

        # 2b. Spawn off child seed sequences to pass to child processes.
        seeds = np.random.randint(2e8, size=self.n_repeat_bsolar)

        # 2c. first we define what we do in each stage of the loop
        def loop_fun(self, j, seeds, qhat_k_container):

            # 2c(1). fix random seed for replication
            np.random.seed(seeds[j])

            # 2c(2). randomly choose a subset of sample points (whose index is 'index_subsample') and use them to generate a subsample in the given repeat of for-loop
            index_subsample = np.random.choice(self.sample_size, self.subsample_size, replace=False)
            # 2c(3). based on 'index_subsample', take the corresponding observations of X as the "X_subample"
            X_subsample = self.X[index_subsample]
            # 2c(4).based on 'index_subsample', take the corresponding observations of Y out and save them as the subample
            y_subsample = self.y[index_subsample]
            # 2c(5). change dimension for solar training
            y_subsample.shape = (y_subsample.shape[0],1)
            # 2c(6). given a subsample, compute solar on it

            #standardize training data
            scaler = preprocessing.StandardScaler().fit(X_subsample)
            X_subsample = scaler.transform(X_subsample)
            
            # call the class 'solar'
            trial2 = solar( X_subsample, y_subsample, self.n_repeat_solar, self.step_size, lasso=False)
            # compute solar on the subsample
            solar_coef, _, _, _, _, _, _, _,_ = trial2.fit()
            # save the active set of solar on this subsample (indices of variables select by solar) as 'active'.
            active = np.nonzero(solar_coef)[0]
            # 2c(7). based on the active set of solar, we compute qhat^k as the binary string of whether each variable is selected by solar on subsample K
            # we generate 'qhat_k' as a row of zeros;
            qhat_k = np.zeros((1, self.n_dim))
            # if a variable (the ith column in matrix X) is selected by solar, we change the ith value of qhat_k as 1
            for i in active:

                qhat_k[0, i] = 1

            # we append the result into 'qhat_k_container' and save it as one element of the list
            qhat_k_container.append(qhat_k)

            return qhat_k_container

        # 2d. parallel the whole for-loop using the function we define previously and save the result
        qhat_k_container = Parallel(n_jobs=-1, prefer="processes")(delayed(loop_fun)(self, j, seeds, qhat_k_container) for j in range(self.n_repeat_bsolar))

        # 3. compute the subsample selection frequency for all variables
        # 3a. we transform the list of all qhat^k ('qhat_k_container') into a matrix ('qhat_k_container_matrix')
        # row of the matrix    : the qhat^k on a given subsample for all variables
        # column of the matrix : the corresponding values of qhat^k for variable "X_i" on all subsamples
        # axis =0  means that we treat each item as a row;
        qhat_k_container_matrix = np.concatenate(qhat_k_container, axis=0)
        # 3b. compute the the value of qhat for each variable (the subsample selection frequency of each variable)
        # e.g., compute the mean of each column
        qhat_value = np.mean(qhat_k_container_matrix, axis=0)
        # 3c. set 'Qc_list' as the container for the subsample selection frequencies for all variables, ranking in decreasing order.
        Qc_list = list()
        # 3d. set 'c_seq' as the sequence of c (the threshold of subsample selection frequency in bsolar)
        c_seq = np.arange(1, 0.1, -0.02)
        # 3e. for each value of c, generate Q(c) --- the set of variables with subsample selection frequency larger or equal to c
        for j in c_seq:
            # 3e(1). define 'container' as the placeholder of Q(c) when c == j;
            container = list()

            for i in range(self.X.shape[1]):
                # 3e(2). include all variables into 'container' if their corresponding values in q-hat are larger or equal to j;
                if (qhat_value[0][i] >= j):

                    container.append(i)
            # 3e(3). append 'container' (Q(c) when c == j) into 'Qc_list' (the container of Q(c) for all value of c);
            Qc_list.append(container)

        # 4. pick the variable that are selected most of the time;
        # 4a. if it is bsolar-S, choose c = 0.9
        Q_opt_c_S = Qc_list[5]
        # if it is bsolar-H, choose c = 1
        Q_opt_c_H = Qc_list[0]

        # 5. output the bsolar-S result (Q_opt_c_S is the active set of bolasso-S)
        # 5a. if Q_opt_c_S is empty, return a zero array and empty active set
        if Q_opt_c_S == []:

            bsolar_coef_S = np.zeros([self.n_dim, 1])
        # 5b. otherwise, regress Y onto the selected variables in X (variables in Q_opt_c_S)
        else :
            # 5b(1). call the LinearRegression class;
            OLS_S = LinearRegression()
            # 5b(2). fit OLS of Y to the variables of Q_opt_c_S on X;
            OLS_S.fit(self.X[:, Q_opt_c_S], self.y)
            # 5b(3). set 'bsolar_coef_S' (an array of zeros) as the placeholder of bsolar-S regression coefficents
            bsolar_coef_S = np.zeros([self.n_dim, 1])
            # 5b(4). put the estimated regression coefficents into their corresponding place of 'bsolarS_coef'
            bsolar_coef_S[Q_opt_c_S, 0] = OLS_S.coef_

        # 5c. output the bsolar-H result (Q_opt_c_H is the active set of bolasso-H)
        # if Q_opt_c_H is empty, return a zero array and empty active set
        if Q_opt_c_H == []:

            bsolar_coef_H = np.zeros([self.n_dim, 1])
        # 5d. otherwise, regress Y onto the selected variables in X (variables in Q_opt_c_H)
        else :
            # 5d(1). call the LinearRegression class;
            OLS_H = LinearRegression()
            # 5d(2). fit OLS of Y on the variables of Q(c*) in X;
            OLS_H.fit(self.X[:, Q_opt_c_H], self.y)
            # 5d(3). set 'bsolar_coef_H' (an array of zeros) as the placeholder of bsolar regression coefficents
            bsolar_coef_H = np.zeros([self.n_dim, 1])
            # 5d(4). put the estimated regression coefficents into their corresponding place of 'bsolarH_coef'
            bsolar_coef_H[Q_opt_c_H, 0] = OLS_H.coef_

        return bsolar_coef_H, bsolar_coef_S, Qc_list, Q_opt_c_H, Q_opt_c_S

    def fit_cd(self):

        #1. construct a placeholder called 'qhat_k_container', which is the list of all qhat^k (a binary string representing whether each variable is selected by solar on subsample k) of each subsample
        qhat_k_container = list()

        #2. train a solar on each subsample, find out which variable is selected on a given sample and save the corresponding selection result on subsample k as qhat^k

        # parallel computing starts
        # 2a. to make parallel computing replicable, set random seeds
        np.random.seed(self.rnd)

        # 2b. Spawn off child seed sequences to pass to child processes.
        seeds = np.random.randint(2e8, size=self.n_repeat_bsolar)

        # 2c. first we define what we do in each stage of the loop
        def loop_fun(self, j, seeds, qhat_k_container):

            # 2c(1). fix random seed for replication
            np.random.seed(seeds[j])

            # 2c(2). randomly choose a subset of sample points (whose index is 'index_subsample') and use them to generate a subsample in the given repeat of for-loop
            index_subsample = np.random.choice(self.sample_size, self.subsample_size, replace=False)
            # 2c(3). based on 'index_subsample', take the corresponding observations of X as the "X_subample"
            X_subsample = self.X[index_subsample]
            # 2c(4).based on 'index_subsample', take the corresponding observations of Y out and save them as the subample
            y_subsample = self.y[index_subsample]
            # 2c(5). change dimension for solar training
            y_subsample.shape = (y_subsample.shape[0],1)
            # 2c(6). given a subsample, compute solar on it

            #standardize training data
            scaler = preprocessing.StandardScaler().fit(X_subsample)
            X_subsample = scaler.transform(X_subsample)
            
            # call the class 'solar'
            trial2 = solar( X_subsample, y_subsample, self.n_repeat_solar, self.step_size, lasso=False)
            # compute solar on the subsample
            solar_coef, _, _, _, _, _, _, _,_ = trial2.fit_cd()
            # save the active set of solar on this subsample (indices of variables select by solar) as 'active'.
            active = np.nonzero(solar_coef)[0]
            # 2c(7). based on the active set of solar, we compute qhat^k as the binary string of whether each variable is selected by solar on subsample K
            # we generate 'qhat_k' as a row of zeros;
            qhat_k = np.zeros((1, self.n_dim))
            # if a variable (the ith column in matrix X) is selected by solar, we change the ith value of qhat_k as 1
            for i in active:

                qhat_k[0, i] = 1

            # we append the result into 'qhat_k_container' and save it as one element of the list
            qhat_k_container.append(qhat_k)

            return qhat_k_container

        # 2d. parallel the whole for-loop using the function we define previously and save the result
        qhat_k_container = Parallel(n_jobs=-1, prefer="processes")(delayed(loop_fun)(self, j, seeds, qhat_k_container) for j in range(self.n_repeat_bsolar))

        # 3. compute the subsample selection frequency for all variables
        # 3a. we transform the list of all qhat^k ('qhat_k_container') into a matrix ('qhat_k_container_matrix')
        # row of the matrix    : the qhat^k on a given subsample for all variables
        # column of the matrix : the corresponding values of qhat^k for variable "X_i" on all subsamples
        # axis =0  means that we treat each item as a row;
        qhat_k_container_matrix = np.concatenate(qhat_k_container, axis=0)
        # 3b. compute the the value of qhat for each variable (the subsample selection frequency of each variable)
        # e.g., compute the mean of each column
        qhat_value = np.mean(qhat_k_container_matrix, axis=0)
        # 3c. set 'Qc_list' as the container for the subsample selection frequencies for all variables, ranking in decreasing order.
        Qc_list = list()
        # 3d. set 'c_seq' as the sequence of c (the threshold of subsample selection frequency in bsolar)
        c_seq = np.arange(1, 0.1, -0.02)
        # 3e. for each value of c, generate Q(c) --- the set of variables with subsample selection frequency larger or equal to c
        for j in c_seq:
            # 3e(1). define 'container' as the placeholder of Q(c) when c == j;
            container = list()

            for i in range(self.X.shape[1]):
                # 3e(2). include all variables into 'container' if their corresponding values in q-hat are larger or equal to j;
                if (qhat_value[0][i] >= j):

                    container.append(i)
            # 3e(3). append 'container' (Q(c) when c == j) into 'Qc_list' (the container of Q(c) for all value of c);
            Qc_list.append(container)

        # 4. pick the variable that are selected most of the time;
        # 4a. if it is bsolar-S, choose c = 0.9
        Q_opt_c_S = Qc_list[5]
        # if it is bsolar-H, choose c = 1
        Q_opt_c_H = Qc_list[0]

        # 5. output the bsolar-S result (Q_opt_c_S is the active set of bolasso-S)
        # 5a. if Q_opt_c_S is empty, return a zero array and empty active set
        if Q_opt_c_S == []:

            bsolar_coef_S = np.zeros([self.n_dim, 1])
        # 5b. otherwise, regress Y onto the selected variables in X (variables in Q_opt_c_S)
        else :
            # 5b(1). call the LinearRegression class;
            OLS_S = LinearRegression()
            # 5b(2). fit OLS of Y to the variables of Q_opt_c_S on X;
            OLS_S.fit(self.X[:, Q_opt_c_S], self.y)
            # 5b(3). set 'bsolar_coef_S' (an array of zeros) as the placeholder of bsolar-S regression coefficents
            bsolar_coef_S = np.zeros([self.n_dim, 1])
            # 5b(4). put the estimated regression coefficents into their corresponding place of 'bsolarS_coef'
            bsolar_coef_S[Q_opt_c_S, 0] = OLS_S.coef_

        # 5c. output the bsolar-H result (Q_opt_c_H is the active set of bolasso-H)
        # if Q_opt_c_H is empty, return a zero array and empty active set
        if Q_opt_c_H == []:

            bsolar_coef_H = np.zeros([self.n_dim, 1])
        # 5d. otherwise, regress Y onto the selected variables in X (variables in Q_opt_c_H)
        else :
            # 5d(1). call the LinearRegression class;
            OLS_H = LinearRegression()
            # 5d(2). fit OLS of Y on the variables of Q(c*) in X;
            OLS_H.fit(self.X[:, Q_opt_c_H], self.y)
            # 5d(3). set 'bsolar_coef_H' (an array of zeros) as the placeholder of bsolar regression coefficents
            bsolar_coef_H = np.zeros([self.n_dim, 1])
            # 5d(4). put the estimated regression coefficents into their corresponding place of 'bsolarH_coef'
            bsolar_coef_H[Q_opt_c_H, 0] = OLS_H.coef_

        return bsolar_coef_H, bsolar_coef_S, Qc_list, Q_opt_c_H, Q_opt_c_S

    # return the full list of subsample selection frequency for each variable in bsolar
    def q_list(self, Qc_list):

        # 1. concatenate Qc_list into a matrix
        var_mark_plot = np.concatenate(Qc_list)
        # 2. compute the value of c for each Q(c) and the corresponding variables in each Q(c)
        var_index, counts = np.unique(var_mark_plot, return_counts=True)

        var_index_ordered = [x for _, x in sorted(zip(counts, var_index))]

        var_plot = var_index_ordered[::-1]

        cou_plot = np.sort(counts)[::-1] / \
            ((self.q_end - self.q_start) / self.q_step)

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

    from simulator import simul

    sample_size     = 100
    n_dim           = 12
    n_info          = 5
    n_repeat_solar  = 10
    n_repeat_bsolar = 3
    step_size       = -0.02

    np.random.seed(0)

    # generate X and Y
    trial1 = simul(sample_size, n_dim, n_info)
    X, Y = trial1.data_gen()

    # start timing
    start = time.time()

    # train solar
    trial2 = bsolar(X, Y, n_repeat_solar, n_repeat_bsolar, step_size)

    bsolar_coef_H, bsolar_coef_S, Qc_list, Q_opt_c_H, Q_opt_c_S = trial2.fit_cd()

    # end timing
    end = time.time()

    # print the result
    print('variables that bsolar-H selects: ', Q_opt_c_H)

    print('variables that bsolar-S selects: ', Q_opt_c_S)
    
    trial2.q_list(Qc_list)