<center><h2>Demonstration of Python packages "solar", "bsolar", and "bolasso"</h2></center>
<center><h3>created at Jan/20/2019</h3></center>
<center><h3>upadated at Apr/20/2022</h3></center>

<br>

### File structure
> <font size="4.5"> the step-by-step walkthrough of "bolasso_parallel", "solar_parallel" and "bsolar_parallel.py"</font>

* #### supporting functions
 
  - **simulator.py** : the data generating package;
  - **costcom.py** : the package to compute the regression error;
  - **solar.py** : the Python package "solar";
  - **solar_parallel.py** : the Python package "solar" (parallel computing);
  - **debug.sh** : (for macOS and Linux only) the bash file for bug testing of all .py files here.
    * in Mac OS or Linux, open terminal and switch to this folder; run "bash debug.sh" commmand
    * it will produces all the test plots, results and tables;
    * if you find no error during the procedure and the bash file ends normally, there is no bug of all the packages in this folder.
  - **solar_simul_demo.py** : all the simulation functions that the solar demostration needs.

* #### demostrations
  - **bsolar_walkthough**, **bolasso_walkthough** and **solar_demo.ipynb** : the step-by-step demonstration for bsolar, bolasso and solar.
