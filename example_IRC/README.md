<center><h2>Example of solar robustness to IRC </h2></center>
<center><h3>created at Jan/20/2019</h3></center>
<center><h3>upadated at Apr/20/2022</h3></center>

<br>

### File structure
> <font size="4.5"> the Python package and simulation for <u>IRC Example, Section 3</u>. </font>

* #### supporting functions
  - **./figures** : the folder of all detailed graphical results, saved as ".pdf";
  - **./numerical_result** : the folder of all numerical results, saved as ".p";
  - **debug.sh** : (for macOS and Linux only) the bash file for bug testing of all .py files here.
    * in Mac OS or Linux, open terminal and switch to this folder; run "bash debug.sh" commmand
    * it will produces all the test plots, results and tables;
    * if you find no error during the procedure and the bash file ends normally, there is no bug of all the packages in this folder.
  - **costcom.py** : the package to compute the regression error;
  - **solar_parallel.py** : the Python package "solar" (parallel computing);
  - **simul_plot_ic.py** : all the simulation functions (computation and plotting functions) that the IRC example requires;
  - **simulator_ic.py** : the data generating package for the IRC example only.

* #### simulations
  
  - **example_IRC.ipynb** : the simulation for the lasso-solar example under different irrepresentable conditions.

<br>
