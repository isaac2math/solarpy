<center><h2>Bootstrap selection comparison: runtime</h2></center>
<center><h3>created at Jan/20/2019</h3></center>
<center><h3>upadated at Apr/20/2022</h3></center>

<br>


### File structure
> <font size="4.5"> the Python packages and simulation for bolasso and bsolar runtime comparison.

* #### supporting functions

  - **./numerical_result** : the folder of all numerical results, saved as ".p";
  - **bolasso** and **bolasso_cd.py** : the Python package "bolasso" (solved by lars and warm-start pathwise coordinate descent);
  - **bsolar_parallel.py** : the Python package "bsolar" (parallel computing);
  - **costcom.py** : the package to compute the regression error;
  - **debug.sh** : (for macOS and Linux only) the bash file for bug testing of all .py files here.
    * in Mac OS or Linux, open terminal and switch to this folder; run "bash debug.sh" commmand
    * it will produces all the test plots, results and tables;
    * if you find no error during the procedure and the bash file ends normally, there is no bug of all the packages in this folder.
  - **solar.py** : the Python package "solar";
  - **simul_built_in_parallel.py** : the simulation function for bolasso runtime by lars (using the built-in "Sci-kit learn" parallel computing scheme);
  - **simul_cd_parallel.py** : the simulation function for bolasso runtime by warm-start pathwise coordinate descent (using the built-in "Sci-kit learn" parallel computing scheme);
  - **simul_joblib_parallel.py** : the simulation function for bolasso runtime (by lars and warm-start pathwise coordinate descent, under the customized Joblib parallel computing scheme);
* #### simulations

  - **bolasso_cd_runtime.ipynb** : the simulation for bolasso runtime by warm-start pathwise coordinate descent (using the built-in "Sci-kit learn" parallel computing scheme);
  - **bolasso_lars_runtime.ipynb** : the simulation for bolasso runtime by lars (using the built-in "Sci-kit learn" parallel computing scheme);
  - **bsolar_runtime.ipynb** : the simulation for bsolar runtime (by lars and warm-start pathwise coordinate descent, under the customized Joblib parallel computing scheme)
  - **runtime_plot.ipynb** : the plotting script for runtime comparison graph.
