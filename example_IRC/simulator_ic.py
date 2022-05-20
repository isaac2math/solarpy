import numpy as np

################################
# define the class 'simulator' #
################################

'''
this class is used to define the data-generating process (Y = Xb + e) for IRC demostration

Check this before you run the code:
Plz check if you have 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install scikit-learn numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install scikit-learn numpy matplotlib tqdm' if you use Anaconda3

Inputs:
    1. sample_size       : the total sample size we generate for cv-lars-lasso, cv-cd and solar
    2. n_dim             : the number of total variables in X
    3. n_info            : the number of informative variables
    4. coef_1 and coef_2 : they are all equal to the value of omega in data-generating process

Outputs:
    1. X, Y   : the inputs and output of regression
    2. abs_ic : the empirical mu (defined in irrepresentable condition) in sample
'''

class simul:

    def __init__(self, sample_size, n_dim, n_info, coef_1, coef_2):
    ##for convinience, we define the common variable (variables we need to use for each of the following functions) in the class as follows (the common variable is defined as self.xxxx)

        self.sample_size = sample_size
        self.n_dim       = n_dim
        self.n_info      = n_info
        self.coef_1      = coef_1
        self.coef_2      = coef_2
        self.coef_res    = np.sqrt(1 - np.power(coef_1,2) - np.power(coef_2,2))


    #data-generating process
    def data_gen(self):

        ##1. generating the covariance matrix for X (except x_5),
        # we add a matrix full of 1/2 to an identity matrix multiplied by 1/2
        a = np.ones((self.n_dim-1, self.n_dim-1)) * 0.5; A = np.eye(self.n_dim-1)*0.5

        cov_x = a + A

        ##2. generating the mean of each column in X (which is 0), except x_5
        mean_x = np.zeros(self.n_dim-1)

        ##3. enerating X as a multivariate Gaussian (except x_5)
        X = np.random.multivariate_normal(mean_x, cov_x, self.sample_size)

        ##4. generating x_5 as the sum of a Gaussian random noise ('noise_ic') and a linear combination of x_0 and x_1
        noise_ic = np.random.normal(0, 1, self.sample_size)
        X_ic = self.coef_1 * X[:,0] + self.coef_2 * X[:,1] + self.coef_res * noise_ic

        ##5. insert x_5 into the 5-th column of X
        X = np.insert(X, 5, X_ic, axis = 1)

        ##6. generate regression coefficients in DGP as an increasing sequence (2,3,4,5,6 in our paper)
        beta_info = np.arange(2, self.n_info + 2)

        ##7. in DGP, generate regression coefficients of redundant variables as 0
        #concatenate the regression coefficients of informative variables and redundant variables
        beta = np.concatenate((beta_info, np.zeros(self.n_dim - self. n_info)), axis = 0)

        ##8. generate the Gaussian random noise
        noise = np.random.normal(0, 1, self.sample_size)

        #transform Gaussian random noise into a column
        #transform regression coefficients in DGP into a row (based on the requirement of np.inner )
        noise.shape = (self.sample_size, 1); beta.shape = (1, self.n_dim)

        ##9. generate Y by adding random noise with the inner product of X and beta
        Y = np.inner(X,beta) + noise

        ##10. compute the empirical mu (defined in irrepresentable condition) in sample
        M = X[:,0:5]
        A = np.linalg.inv(np.dot(M.T,M))
        B = np.dot(M.T,X_ic)
        abs_ic = np.linalg.norm(np.dot(A,B),1)

        return X, Y, abs_ic

##################################
# test if this module works fine #
##################################

'''
this part is set up to test the functionability of the class above;
you can run all the codes in this file to test if the class works;
when you call the class from this file, the codes (function or class) after " if __name__ == '__main__': " will be ingored
'''

if __name__ == '__main__':

    from sklearn.linear_model import LinearRegression
    
    sample_size = 200
    n_dim       = 100
    n_info      = 5

    trial = simul(sample_size, n_dim, n_info, 1/2, 1/2)
    X, Y, abs_ic = trial.data_gen()

    print(X)
    print(Y)
    print(abs_ic)

    # to verfiy if we generate X and Y correctly, we compute and output the OLS coefficients of Y onto all informative variables
    reg = LinearRegression(); reg.fit(X[:,0:5],Y)
    print(reg.coef_)
