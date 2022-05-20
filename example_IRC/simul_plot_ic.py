import numpy             as np
import matplotlib.pyplot as plt

from simulator_ic   import simul
from solar_parallel import solar
from tqdm           import tqdm

########################################################################
# define the class 'the simulation_plot' for irrepresentable condition #
########################################################################
'''
this class is used for plotting the result of IRC example

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
    4. n_info      : number of non-zero regression coefficients in data-generating process
    5. n_repeat    : the number of subsamples in solar
    6. num_rep     : the number of repeatitions
    7. step_size   : step size for tuning the value of c for solar;
    8. rnd_seed    : the random seed
    9. plot_on     : binary, whether the plot will be saved as pdf

Outputs:
    1. solar_coef_stack: the stack of solar regression coefficients in 200 repeats
    2. opt_c_stack     : the stack of values of c* in 200 repeats
    3. Q_opt_c_stack   : the stack of Q(c*) in 200 repeats
    4. la_array_stack  : the stack of number of variables selected by CV-lars-lasso in 200 repeats
    5. la_var_stack    : the stack of the variables selected by CV-lars-lasso in 200 repeats
    6. cd_array_stack  : the stack of number of variables selected by CV-cd in 200 repeats
    7. cd_var_stack    : the stack of the variables selected by CV-cd in 200 repeats

Remarks:
    1. In each round of subsampling, we randomly take out 1/K points out of the sample and make the rest as the subsample in this round
    2. As competitors, we use X and Y for LassoLarsCV (called CV-lars-lasso in paper) and LassoCV (called CV-cd in paper) estimation, which relies on 10-fold CV.
'''
class simul_plot:

    def __init__(self, sample_size, n_dim, n_info, coef_1, coef_2, n_repeat, num_rep, step_size, rnd_seed, plot_on):
    ##for convinience, we define the common variable (variables we need to use multiple times) in the class as follows (xxxx as self.xxxx)

        #define the paras
        self.sample_size = sample_size #sample size
        self.n_dim       = n_dim       #the number of total variables in X
        self.n_info      = n_info      #number of non-zero regression coefficients in data-generating process
        self.n_repeat    = n_repeat    #the number of subsamples generated in solar
        self.num_rep     = num_rep     #the number of repeatitions in Simulation 1, 2 and 3
        self.step_size   = step_size   #step size for tuning the value of c for solar
        self.rnd_seed    = rnd_seed    #the random seed for reproduction
        self.coef_1      = coef_1      #the value of omega in data-generating process
        self.coef_2      = coef_2      #the value of omega in data-generating process
        self.coef_t      = coef_1 + coef_2  #if this value is less than 1 (omega is positive in our simulation), irrepresentable condition is not violated
        self.plot_on     = plot_on     #whether the plot will be saved as pdf


    def simul_func(self):
    #compute 200 repeats of solar vs cv-lars-lasso and cv-cd

        opt_c_stack      = list() #the stack of values of c* in 200 repeats
        Q_opt_c_stack    = list() #the stack of Q(c*) in 200 repeats
        la_array_stack   = list() #the stack of number of variables selected by CV-lars-lasso in 200 repeats
        la_var_stack     = list() #the stack of the variables selected by CV-lars-lasso in 200 repeats
        cd_array_stack   = list() #the stack of number of variables selected by CV-cd in 200 repeats
        cd_var_stack     = list() #the stack of the variables selected by CV-cd in 200 repeats
        solar_coef_stack = list() #the stack of solar regression coefficients in 200 repeats
        abs_ic_stack     = list() #the stack of empirical value of mu (defined alongwide with irrepresentable condition (Section 2.1)) in each sample (across 200 repeats)

        #to make parallel computing replicable, set random seeds
        np.random.seed(self.rnd_seed)
        # Spawn off 200 child SeedSequences to pass to child processes.
        seeds = np.random.randint(1e8, size=self.num_rep)

        ##use for-loop to compute 200 repeats
        #use 'tqdm' in for loop to construct the progress bar
        for i in tqdm(range(self.num_rep)):

            np.random.seed(seeds[i])

            #if irrepresentable condition is violated in population, break
            if self.coef_t > 1:

                print("Warning: this simulation and plotting classes is composed only for the cases where irrepresentable condition is satistied. When irrepresentable condition is violated, the ranking of variables in plots and their probaility will be in error and misordered.")

                break

            #1.  call the class 'simul' from 'simul.py'
            trial1 = simul(self.sample_size, self.n_dim, self.n_info, self.coef_1, self.coef_2)
            #2. generate X and Y in each repeat
            X, Y, abs_ic = trial1.data_gen()

            #3.  call the class 'solar' from 'solar.py'
            trial2 = solar( X, Y, self.n_repeat, self.step_size, lasso = True)
            #4.  compute solar, cv-lars-lasso and cv-cd on X and Y
            solar_coef, opt_c, test_error, Qc_list, Q_opt_c, la_list, la_vari_list, cd_list, cd_vari_list = trial2.fit()

            #5.  find Q(c*) for solar
            min_loc_val = np.where(test_error == min(test_error))[0]
            Q_opt_c     = Qc_list[max(min_loc_val)]

            #6.  save the value of c* (opt_c) into 'opt_c_stack' (the stack of 'the value of c* of solar' in 200 repeats)
            opt_c_stack.append(max(opt_c))
            #7.  save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by solar' in 200 repeats)
            Q_opt_c_stack.append(Q_opt_c)
            #8.  save the number of variables selected by cv-lars-lasso (la_list) into 'la_array_stack' (the stack of 'number of variables selected by cv-lars-lasso' in 200 repeats)
            la_array_stack.append(la_list)
            #9.  save the variables selected by cv-lars-lasso (la_vari_list) into 'la_var_stack' (the stack of 'variables selected by cv-lars-lasso' in 200 repeats)
            la_var_stack.append(la_vari_list)
            #10. save the number of variables selected by cv-cd (cd_list) into 'cd_array_stack' (the stack of 'number of variables selected by CV-cd' in 200 repeats)
            cd_array_stack.append(cd_list)
            #11. save the variables selected by cv-cd (cd_vari_list) into 'cd_var_stack' (the stack of 'variables selected by CV-cd' in 200 repeats)
            cd_var_stack.append(cd_vari_list)
            #12. save solar regression coefficients (solar_coef) into 'solar_coef_stack' (the stack of 'solar regression coefficents' in 200 repeats)
            solar_coef_stack.append(solar_coef)

            abs_ic_stack.append(abs_ic)

        return opt_c_stack, Q_opt_c_stack, la_array_stack, la_var_stack, solar_coef_stack, abs_ic_stack, cd_array_stack, cd_var_stack


    def vari_hist(self, Q_opt_c_stack, la_array_stack, cd_array_stack):
    #the histogram of number of variables selected : solar vs competitors

        if self.coef_t <= 1:

            #use 'solar_len_array' to record the number of variables selected by solar in each repeat
            solar_len_array = np.empty([len(Q_opt_c_stack)])

            #count the number of variables selected by solar in each repeat by computing the length (number of elements) of Q(c) in each repeat
            for i in range(len(Q_opt_c_stack)):

                solar_len_array[i] = len(Q_opt_c_stack[i])


            ##overlaid histogram plot of num_var_selected: solar vs CV-cd
            f11 = plt.figure()

            #histogram plot of solar and CV-cd
            plt.hist(solar_len_array, 20, density = True, alpha=0.8, color = "dodgerblue", label='Number of variables selected by solar')
            plt.hist(cd_array_stack, 20, density = True, alpha=0.65, color = "lawngreen", label='Number of variables selected by CV-cd ')

            #plot vertical lines for the mean of solar and cv-cd (and report them as legend)
            plt.axvline(x=np.mean(solar_len_array), linewidth=2.5, color='b',label='solar mean')
            plt.gcf().text(1, 1, 'mean for solar : ' + str(np.mean(solar_len_array)))

            plt.axvline(x=np.mean(cd_array_stack), linewidth=2.5, color='g',label='CV-cd mean')
            plt.gcf().text(1, 0.9, 'mean for CV-cd  : ' + str(np.mean(cd_array_stack)))

            #plot vertical lines for the median of solar and cv-cd (and report them as legend)
            plt.axvline(x=np.median(solar_len_array), linewidth=2.5, color='b', ls =':', label='solar median')
            plt.gcf().text(1, 0.85, 'median for solar : ' + str(np.median(solar_len_array)))

            plt.axvline(x=np.median(cd_array_stack), linewidth=2.5, color='g', ls =':', label=' CV-cd median')
            plt.gcf().text(1, 0.75, 'median for CV-cd  : ' + str(np.median(la_array_stack)))

            #legend of histogram plot
            plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), borderaxespad=0., ncol=3, shadow=True)
            plt.xlabel('number of variables selected', fontsize=16)
            plt.xlim(3,30)
            plt.ylim(0,0.6)
            plt.ylabel('Density', fontsize=16)
            plt.title('solar vs CV-cd : sparsity', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.grid(True)
            plt.show()

            #output it into pdf
            if self.plot_on == True:
                f11.savefig("./figure/sparsity_plot_ic_"+str(self.coef_1)+"_vs_CVcd.pdf", bbox_inches='tight')


            ##overlaid hist plot of num_var_selected: solar vs CV-lars-lasso
            f1 = plt.figure()

            #histogram plot of solar and CV-lars-lasso
            plt.hist(solar_len_array, 20, density = True, alpha=0.8, color = "dodgerblue", label='Number of variables selected by solar')
            plt.hist(la_array_stack, 20, density = True, alpha=0.65,  color = "orange", label='Number of variables selected by CV-lars-lasso')

            #plot vertical lines for the mean of solar and CV-lars-lasso (and report them as legend)
            plt.axvline(x=np.mean(solar_len_array), linewidth=2.5, color='b',label='solar mean')
            plt.gcf().text(1, 1, 'mean for solar : ' + str(np.mean(solar_len_array)))

            plt.axvline(x=np.mean(la_array_stack), linewidth=2.5, color='r',label='CV-lars-lasso mean')
            plt.gcf().text(1, 0.95, 'mean for CV-lars-lasso  : ' + str(np.mean(la_array_stack)))

            #plot vertical lines for the median of solar and CV-lars-lasso (and report them as legend)
            plt.axvline(x=np.median(solar_len_array), linewidth=2.5, color='b', ls =':', label='solar median')
            plt.gcf().text(1, 0.85, 'median for solar : ' + str(np.median(solar_len_array)))

            plt.axvline(x=np.median(la_array_stack), linewidth=2.5, color='r', ls =':', label=' CV-lars-lasso median')
            plt.gcf().text(1, 0.8, 'median for CV-lars-lasso  : ' + str(np.median(la_array_stack)))

            #legend
            plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), borderaxespad=0., ncol=3, shadow=True)
            plt.xlabel('number of variables selected', fontsize=16)
            plt.xlim(3,30)
            plt.ylim(0,0.6)
            plt.ylabel('Density', fontsize=16)
            plt.title('solar vs CV-lars-lasso : sparsity', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.grid(True)
            plt.show()

            #output it into pdf
            if self.plot_on == True:
                f1.savefig("./figure/sparsity_plot_ic_"+str(self.coef_1)+"_vs_CVlars.pdf", bbox_inches='tight')

            #count how many repeats solar select 5 variables
            print("in " + str(np.count_nonzero(solar_len_array- 5)) + " out of " + str(self.num_rep) + " solar pick more/less than 5 variables")
            #count how many repeats CV-lars-lasso select 5 variables
            print("in " + str(np.count_nonzero(np.array(la_array_stack)  - 5)) + " out of " + str(self.num_rep) + " CV-lars-lasso  pick more/less than 5 variables")
            #count how many repeats CV-cd select 5 variables
            print("in " + str(np.count_nonzero(np.array(cd_array_stack)  - 5)) + " out of " + str(self.num_rep) + " CV-cd pick more/less than 5 variables")

        #if the irrepresentable condition is violated
        else:
            print("Warning: this simulation and plotting classes is composed only for the cases where irrepresentable condition is satistied. When irrepresentable condition is violated, the ranking of variables in plots and their probaility will be in error and misordered.")



    def q_hist(self, opt_c_stack):
    #histogram plot of c* of solar in 200 repeatss

        if self.coef_t <= 1:

            f2 = plt.figure()

            plt.hist(opt_c_stack, facecolor='r', alpha=0.75)

            plt.xlabel('c*', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.title('histogram of c* selected by solar among 200 repeats', fontsize=16)
            plt.grid(True)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tight_layout()
            plt.show()

            if self.plot_on == True:
                f2.savefig("./figure/q-values_plot_ic_"+str(self.coef_1)+".pdf", bbox_inches='tight')

        #if the irrepresentable condition is violated
        else:
            print("Warning: this simulation and plotting classes is composed only for the cases where irrepresentable condition is satistied. When irrepresentable condition is violated, the ranking of variables in plots and their probaility will be in error and misordered.")



    def acc_plot(self, Q_opt_c_stack, la_var_stack, cd_var_stack, num_var_to_plot, print_true):
    #bar plot: probability of solar/cv-lars-lasso/cv-cd selecting each variable (all 5 informative variables; redundant variables (of top 35 probability))

        if self.coef_t <= 1:

            #prepare the bar plot of solar
            #1. first we concatenate 'Q_opt_c_stack' (the stack of indices of variables selected by solar among 200 repeats) into one whole array ('solar_vari_appe_stack')
            solar_vari_appe_stack = np.concatenate(Q_opt_c_stack,0)
            #2. set 'solar_plot_stack' as the placeholder of the probailities of selecting each variable by solar (ranking in decreasing order)
            solar_plot_stack = list()
            #3. set 'solar_x_axi' as the label of variable names (from x_0 to x_p, where p is the number of variables in X)
            solar_x_axi = np.arange(0, self.n_dim)

            #prepare the bar plot of CV-lars-lasso
            #1. first we concatenate 'la_var_stack' (the stack of indices of variables selected by cv-lars-lasso among 200 repeats) into one whole array ('l_vari_appe_stack')
            l_vari_appe_stack = np.concatenate(la_var_stack,0)
            #2. set 'l_plot_stack' as the container of the probailities of selecting each variable by cv-lars-lasso (ranking in decreasing order)
            l_plot_stack = list()
            #3. set 'l_x_axi' as the array of variable names (from x_0 to x_p, where p is the number of variables in X)
            l_x_axi = np.arange(0,self.n_dim)

            #prepare the bar plot of CV-cd
            #1. first we concatenate 'cd_var_stack' (the stack of indices of variables selected by cv-cd) into one whole array ('cd_vari_appe_stack')
            cd_vari_appe_stack = np.concatenate(cd_var_stack,0)
            #2. set 'cd_plot_stack' as the container of the probailities of selecting each variable by cv-cd (ranking in decreasing order)
            cd_plot_stack = list()
            #3. set 'cd_x_axi' as the array of variable names (from x_0 to x_p, where p is the number of variables in X)
            cd_x_axi = np.arange(0,self.n_dim)

            #compute the probailities of solar/cv-lars-lasso/cv-cd selecting each variable.
            for i in range(self.n_dim):

                #1. Given the variable i in X, compute the probability of selecting it by solar via (1) counting how many i in 'solar_vari_appe_stack' (the stack of indices of variables selected by solar among 200 repeats) (2) divide that count by number of repeats (200 in paper) (3) save it as the i-th element of solar_plot_stack
                solar_plot_stack.append((solar_vari_appe_stack == i).sum()/self.num_rep)

                #2. Given the variable i in X, compute the probability of selecting it by cv-lars-lasso via (1) counting how many i in 'l_vari_appe_stack' (the stack of indices of variables selected by cv-lars-lasso among 200 repeats) (2) divide that count by number of repeats (200 in paper) (3) save it as the i-th element of l_plot_stack
                l_plot_stack.append((l_vari_appe_stack == i).sum()/self.num_rep)

                #3. Given the variable i in X, compute the probability of selecting it by cv-cd via (1) counting how many i in 'cd_vari_appe_stack' (the stack of indices of variables selected by cv-cd among 200 repeats) (2) divide that count by number of repeats (200 in paper) (3) save it as the i-th element of cd_plot_stack
                cd_plot_stack.append((cd_vari_appe_stack == i).sum()/self.num_rep)


            ##Ranking (decreasingly) the probability of solar selecting each variable
            ####Since we only plot the redundant variable with top 35 probabilities, in this part we use the method called 'simultaneous sorting': sort two array (the array of 'probability of selecting each variable' and the array of 'labels for the horizontal axis in the  figure') simultaneously based on the order of the first array

            #1a. rank elements of 'solar_plot_stack' in increasing order; reverse it into decreasing order and save it as 'solar_plot_stack_ranked'
            #1b. ranking 'solar_x_axi' based on the increasing order of 'solar_plot_stack' and save it as 'solar_x_axi_ranked'
            solar_x_axi_ranked       = [x for _,x in sorted(zip(solar_plot_stack,solar_x_axi))]
            solar_plot_stack_ranked  = np.sort(solar_plot_stack)[::-1]
            #2.  generate 'solar_label' as the label (for horizontal axis in figure) of each variable in the increasing order.
            solar_labels             = [ 'X' + str(x) for x in solar_x_axi_ranked]
            #3.  the index of variable plotted at each bar
            solar_final_x_axi_ranked = solar_x_axi_ranked[::-1]
            #4.  reverse the order of 'solar_label' to decreasing order
            solar_final_label        = solar_labels[::-1]


            ##Ranking (decreasingly) the probability of cv-lars-lasso selecting each variable
            ##Since we only plot the redundant variable with top 35 probabilities, in this part we use the method called 'simultaneous sorting': sort two array (the array of 'probability of selecting each variable' and the array of 'labels for the horizontal axis in the  figure') simultaneously based on the order of the first array

            #1a. rank elements of 'l_plot_stack' in increasing order; reverse it into decreasing order and save it as 'l_plot_stack_ranked'
            #1b. ranking 'l_x_axi' based on the increasing order of 'l_plot_stack' and save it as 'l_x_axi_ranked'
            l_x_axi_ranked       = [x for _,x in sorted(zip(l_plot_stack,l_x_axi))]
            l_plot_stack_ranked  = np.sort(l_plot_stack)[::-1]
            #2.  generate 'l_label' as the label (for horizontal axis in figure) of each variable in the increasing order.
            l_labels             = [ 'X' + str(x) for x in l_x_axi_ranked]
            #3.  the index of variable plotted at each bar
            l_final_x_axi_ranked = l_x_axi_ranked[::-1]
            #4.  reverse the order of 'l_label' to decreasing order
            l_final_label        = l_labels[::-1]


            ##Ranking (decreasingly) the probability of cv-cd selecting each variable
            ##Since we only plot the redundant variable with top 35 probabilities, in this part we use the method called 'simultaneous sorting': sort two array (the array of 'probability of selecting each variable' and the array of 'labels for the horizontal axis in the  figure') simultaneously based on the order of the first array

            #1a. rank elements of 'cd_plot_stack' in increasing order; reverse it into decreasing order and save it as 'cd_plot_stack_ranked'
            #1b. ranking 'cd_x_axi' based on the increasing order of 'cd_plot_stack' and save it as 'cd_x_axi_ranked'
            cd_x_axi_ranked       = [x for _,x in sorted(zip(cd_plot_stack,cd_x_axi))]
            cd_plot_stack_ranked  = np.sort(cd_plot_stack)[::-1]
            #2.  generate 'cd_label' as the label (for horizontal axis in figure) of each variable in the increasing order.
            cd_labels             = [ 'X' + str(x) for x in cd_x_axi_ranked]
            #3.  the index of variable plotted at each bar
            cd_final_x_axi_ranked = cd_x_axi_ranked[::-1]
            #4.  reverse the order of 'cd_label' to decreasing order
            cd_final_label        = cd_labels[::-1]

            #bar plot for solar
            f4 = plt.figure()

            # The big plot
            ax = f4.add_subplot(111)

            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax.yaxis.set_major_locator(plt.NullLocator())

            #the bar plot of the probability of solar selecting each variable
            ax1 = f4.add_subplot(212)

            for i in range(min(num_var_to_plot, self.n_dim)):

                #turn the bar plot of informative variables into red
                if solar_final_x_axi_ranked[i] < 5:

                    ax1.bar(x = solar_x_axi[i], height= solar_plot_stack_ranked[i], width=0.3, color = ['red'])

                #turn the bar plot of x_5 into orangered
                elif solar_final_label[i] == 'X5':

                    ax1.bar(x = solar_x_axi[i], height= solar_plot_stack_ranked[i], width=0.3, color = ['orangered'])

                #turn the bar plot of redundant variables into blue
                else:

                    ax1.bar(x = solar_x_axi[i], height= solar_plot_stack_ranked[i], width=0.3, color = ['blue'])

            # add the variable name as the labels for the horizontal axis of the bar plot
            plt.xticks(solar_x_axi[0 : num_var_to_plot], solar_final_label[:num_var_to_plot], rotation='vertical')
            plt.tick_params(axis='both', which='major', labelsize=16)

            #if we want to see the probability for redundant variables, change the plot range of the horizontal axis
            if print_true == False :

                ax1.set_xlim(4.5, num_var_to_plot + 1)
                ax1.set_ylim(0, 0.55)
                ax1.set_ylabel('Probability', fontsize=16)
                plt.title('solar', fontsize=16)

            #if we want to see the probability for informative variables, change the plot range of the horizontal axis
            else:

                ax1.set_xlim(-0.5, 4.5)
                ax1.set_ylim(0, 1.05)
                ax1.set_ylabel('Probability', fontsize=16)
                plt.title('solar', fontsize=16)

            plt.tight_layout()
            plt.show()

            #output it as pdf file
            if self.plot_on == True:
                f4.savefig("./figure/acc_plot_top" + str(num_var_to_plot) + "_ic_" + str(self.coef_1) + "_" + str(print_true) + "_solar.pdf", bbox_inches='tight')


            ##bar plot of cv-lars-lasso
            f44 = plt.figure()

            # The big plot
            ax00 = f44.add_subplot(111)

            # Turn off axis lines and ticks of the big subplot
            ax00.spines['top'].set_color('none')
            ax00.spines['bottom'].set_color('none')
            ax00.spines['left'].set_color('none')
            ax00.spines['right'].set_color('none')
            ax00.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax00.yaxis.set_major_locator(plt.NullLocator())

            #the bar plot of the probability of cv-lars-lasso selecting each variable
            ax2 = f44.add_subplot(212)

            for i in range(min(num_var_to_plot, self.n_dim)):

                #turn the bar plot of informative variables into red
                if l_final_x_axi_ranked[i] < 5:

                    ax2.bar(x = l_x_axi[i], height= l_plot_stack_ranked[i], width=0.3, color = ['red'])

                #turn the bar plot of x_5 into orangered
                elif l_final_label[i] == 'X5':

                    ax2.bar(x = l_x_axi[i], height= l_plot_stack_ranked[i], width=0.3, color = ['orangered'])

                #turn the bar plot of redundant variables into blue
                else:

                    ax2.bar(x = l_x_axi[i], height= l_plot_stack_ranked[i], width=0.3, color = ['blue'])

            # add the variable name as the labels for the horizontal axis of the bar plot
            plt.xticks(l_x_axi[0 : num_var_to_plot], l_final_label[:num_var_to_plot], rotation='vertical')
            plt.tick_params(axis='both', which='major', labelsize=16)

            #if we want to see the probability for redundant variables, change the plot range of the horizontal axis
            if print_true == False :

                ax2.set_xlim(4.5, num_var_to_plot + 1)
                ax2.set_ylim(0, 0.55)
                ax2.set_ylabel('Probability', fontsize=16)
                plt.title('CV-lars-lasso ', fontsize=16)

            #if we want to see the probability for informative variables, change the plot range of the horizontal axis
            else:

                ax2.set_xlim(-0.5, 4.5)
                ax2.set_ylim(0, 1.05)
                ax2.set_ylabel('Probability', fontsize=16)
                plt.title('CV-lars-lasso ', fontsize=16)

            plt.tight_layout()
            plt.show()

            #output it as pdf file
            if self.plot_on == True:
                f44.savefig("./figure/acc_plot_top" + str(num_var_to_plot) + "_ic_" + str(self.coef_1) + "_" + str(print_true) + "_lars.pdf", bbox_inches='tight')


            #bar plot for cv-cd
            f444 = plt.figure()

            # The big plot
            ax000 = f444.add_subplot(111)

            # Turn off axis lines and ticks of the big subplot
            ax000.spines['top'].set_color('none')
            ax000.spines['bottom'].set_color('none')
            ax000.spines['left'].set_color('none')
            ax000.spines['right'].set_color('none')
            ax000.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax000.yaxis.set_major_locator(plt.NullLocator())

            #bar plot of the probability of cv-cd selecting each variable
            ax3 = f444.add_subplot(212)

            for i in range(min(num_var_to_plot, self.n_dim)):

                #turn the bar plot of informative variables into red
                if cd_final_x_axi_ranked[i] < 5:

                    ax3.bar(x = cd_x_axi[i], height= cd_plot_stack_ranked[i], width=0.3, color = ['red'])

                #turn the bar plot of x_5 into orangered
                elif cd_final_label[i] == 'X5':

                    ax3.bar(x = cd_x_axi[i], height= cd_plot_stack_ranked[i], width=0.3, color = ['orangered'])

                #turn the bar plot of redundant variables into blue
                else:

                    ax3.bar(x = cd_x_axi[i], height= cd_plot_stack_ranked[i], width=0.3, color = ['blue'])

            # add the variable name as the labels for the horizontal axis of the bar plot
            plt.xticks(cd_x_axi[0 : num_var_to_plot], cd_final_label[:num_var_to_plot], rotation='vertical')
            plt.tick_params(axis='both', which='major', labelsize=16)

            #if we want to see the probability for redundant variables, change the plot range of the horizontal axis
            if print_true == False :

                ax3.set_xlim(4.5, num_var_to_plot + 1)
                ax3.set_ylim(0, 0.55)
                ax3.set_ylabel('Probability', fontsize=16)
                plt.title('CV-cd', fontsize=16)

            #if we want to see the probability for informative variables, change the plot range of the horizontal axis
            else:

                ax3.set_xlim(-0.5, 4.5)
                ax3.set_ylim(0, 1.05)
                ax3.set_ylabel('Probability', fontsize=16)
                plt.title('CV-cd', fontsize=16)

            plt.tight_layout()
            plt.show()

            #output it as pdf file
            if self.plot_on == True:
                f444.savefig("./figure/acc_plot_top" + str(num_var_to_plot) + "_ic_" + str(self.coef_1) + "_" + str(print_true) + "_cd.pdf", bbox_inches='tight')

        else:
            print("Warning: this simulation and plotting classes is composed only for the cases where irrepresentable condition is satistied. When irrepresentable condition is violated, the ranking of variables in plots and their probaility will be in error and misordered.")



    def bl_vari_plot(self, solar_coef_stack, num_var_to_plot):
    #boxplot of each solar regression coefficient
    #since we only boxplot olar regression coefficients (of top 15 means), we do the ranking again

        if self.coef_t <= 1:

            #boxplot of each regression coefficient for solar
            bl_coef_stack = np.concatenate(solar_coef_stack,1)
            solar_x_axi      = np.arange(0, self.n_dim)
            solar_plot_stack = np.mean(bl_coef_stack, axis=1)

            solar_final_x_axi_ranked = [x for _,x in sorted(zip(solar_plot_stack,solar_x_axi))][::-1]
            #solar_plot_stack_ranked  = np.sort(solar_plot_stack)[::-1]
            solar_final_label        = [ 'beta' + str(x) for x in solar_final_x_axi_ranked]

            f3 = plt.figure()

            plt.boxplot(list(bl_coef_stack[solar_final_x_axi_ranked[:num_var_to_plot],:]),
                        positions=solar_x_axi[:num_var_to_plot])


            plt.xticks(solar_x_axi[0 : num_var_to_plot], solar_final_label[:num_var_to_plot], rotation='vertical')

            loc_5 = solar_final_x_axi_ranked.index(5)

            if loc_5 <= num_var_to_plot:

                plt.axvspan(loc_5 - 0.5, loc_5 + 0.5, color='red', alpha=0.25)

            plt.title('The boxplot of solar regression coefficient', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tight_layout()
            plt.show()

            #count in how many repeats solar estimate the regression coefficient of x_0 as 0
            print('the number of non-zeros in the boxplot of beta_0: ', self.num_rep - len(np.where(bl_coef_stack[0,:] == 0)[0]))
            #count in how many repeats solar estimate the regression coefficient of x_1 as 1
            print('the number of non-zeros in the boxplot of beta_1: ', self.num_rep - len(np.where(bl_coef_stack[1,:] == 0)[0]))
            #count in how many repeats solar estimate the regression coefficient of x_2 as 2
            print('the number of non-zeros in the boxplot of beta_2: ', self.num_rep - len(np.where(bl_coef_stack[2,:] == 0)[0]))
            #count in how many repeats solar estimate the regression coefficient of x_3 as 3
            print('the number of non-zeros in the boxplot of beta_3: ', self.num_rep - len(np.where(bl_coef_stack[3,:] == 0)[0]))
            #count in how many repeats solar estimate the regression coefficient of x_4 as 4
            print('the number of non-zeros in the boxplot of beta_4: ', self.num_rep - len(np.where(bl_coef_stack[4,:] == 0)[0]))

            #output it as pdf file
            if self.plot_on == True:
                f3.savefig("./figure/solar_vari_plot_top"+str(num_var_to_plot)+"_ic_"+str(self.coef_1)+".pdf", bbox_inches='tight')

        else:
            print("Warning: this simulation and plotting classes is composed only for the cases where irrepresentable condition is satistied. When irrepresentable condition is violated, the ranking of variables in plots and their probaility will be in error and misordered.")


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
    n_repeat    = 3
    step_size   = -0.02
    num_rep     = 3
    rnd_seed    = 1
    coef_1      = 1/3
    coef_2      = 1/3
    plot_on     = False

    trial = simul_plot(sample_size, n_dim, n_info, coef_1, coef_2, n_repeat, num_rep, step_size, rnd_seed, plot_on)

    opt_c_stack, Q_opt_c_stack, la_array_stack, la_var_stack, solar_coef_stack, abs_ic_stack, cd_array_stack, cd_var_stack = trial.simul_func()

    trial.vari_hist(Q_opt_c_stack, la_array_stack, cd_array_stack)
    trial.q_hist(opt_c_stack)

    print_true_1      = True
    print_true_2      = False
    num_var_to_plot_1 = 15

    trial.acc_plot(Q_opt_c_stack, la_var_stack, cd_var_stack, num_var_to_plot_1, print_true_1)
    trial.acc_plot(Q_opt_c_stack, la_var_stack, cd_var_stack, num_var_to_plot_1, print_true_2)
    trial.bl_vari_plot(solar_coef_stack, num_var_to_plot_1)
