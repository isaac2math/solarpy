<center><h2>Bootstrap selection comparison: subsample selection efficiency</h2></center>
<center><h3>created at Jan/20/2019</h3></center>
<center><h3>upadated at Apr/20/2022</h3></center>

<br>


### File structure
> <font size="4.5"> the Python packages and simulation for bolasso and bsolar comparison on subsample selection efficiency.

* #### supporting functions

  - **./numerical_result** : the folder of all numerical results, saved as ".p";
  - **bolasso_parallel.py** : the Python package "bolasso" (parallel computing);
  - **bsolar.py** : the Python package "bsolar";
  - **costcom.py** : the package to compute the regression error;
  - **debug.sh** : (for macOS and Linux only) the bash file for bug testing of all .py files here.
    * in Mac OS or Linux, open terminal and switch to this folder; run "bash debug.sh" commmand
    * it will produces all the test plots, results and tables;
    * if you find no error during the procedure and the bash file ends normally, there is no bug of all the packages in this folder.
  - **solar** and **solar_parallel.py** : the Python package "solar";
  - **bootstrap_demo_parallel.py** : all the simulation functions (computation and plotting functions) that bsolar and bolasso require in the simulation;
* #### simulations

  - **subsample_frequency_bolasso_bsolar.ipynb** : the simulation for bolasso-bsolar comparison on subsample selection efficiency;