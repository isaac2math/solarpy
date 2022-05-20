<center><h1>Python package "solarpy"</h1></center>
<center><h3>last updated at Jan/20/2022</h3></center>

<br>

### The corresponding paper can be found at https://arxiv.org/abs/2007.15707.
### A detailed video walkthrough can be found at .

 

<br>

### #0. This Github repository includes
* <font size="4.5"> the Python packages (both parallel and sequential vesions) for
  * solar and solar-cd (cd for coordinate descent);
  * bsolar and bsolar-cd;
  * bolasso and bolasso-cd;
* the step-by-step demostration for solar, bsolar and bolasso;
* the codes, graphs, tables, raw results, and data for
  * examples of data-splitting test and IRC;
  * simulation codes and results of solar, lasso, bsolar, and bolasso, including
    * runtime comparison;
    * sparsity and accuracy comparison;
  * real-world application code and result;

</font>

#### to quickly verify the paper results, all raw results are saved as HTML files in the "raw_results" folder with detailed comments and explanations. See Section "#3. File structure" in this file for details.

<br>

#### This package is programmed based on the following flow.

![Programming logics](flow.png)

<br>

* <font size="4.5"> we program the function for L2 error and data generating procedure based on the functions from Anaconda 3 (an easy-to-use Python enviroment)

* based on the function above, we define the function of solar, bolasso, and bootstrap solar function.

* based on the function above, we define the functions for plotting and calculation in simulations

* based on the function above, we conduct all simulations in ipynb files and save the raw result of simluations as HTML.
</font>

<br>
### #1. Package description


#### #1(a). Ananconda3

* <font size="4.5"> solar and bsolar are developed on Ubuntu 18.04 using Anaconda3 version 2019-03. **I highly recommend use Anaconda3 since it by default satisfies all the dependencies (listed as #2 of this file), relieving you from Python package management and the configuration of the C++/Fortran library.**
</font>

<br>

#### #1(b). Coding style

<font size="4.5"> When I develop this package, I rigorously stick to the following programming paradigm:

* **the comment-code ratio is larger than $1/1$.** 
  I only <u>briefly list the file structure as #3 of this file (explaining the purpose of each file)</u>, since in each ".py" and ".ipynb" file I carefully and thoroughly explain
  * the meaning of every step;
  * the meaning of inputs and output;
  * the purpose of each function;
  * how each step corresponds to the paper.

* **the simulations and examples are done in ".ipynb" files;** 

* **all the ".py" files only contain the supporting function for simulations and examples**.

* at the end of each ".py" file, I add a testing module for debug. You can simple run each ".py" file at terminal. If no bug are reported, the package is bug-free.

* The Python files automatically export the raw simulation/example results as ".html" files, which can be found at the "./raw_results" folder; the numerical results are automatically saved as ".p" files at the "numerical_result" subfolder at each simulation folder.

</font>

<br>

#### #1(c). To make the comparisons fair on a distributed parallel-computation platform

<font size="4.5"> **To make a fair sparsity/accuracy comparison among solar, lasso, bsolar and bolasso,** I specifically code the simulation function "simul_plot_parallel.py" in the following way:

* step 1: before the comparison simulation starts, I reset the Numpy random seed as $0$ (aka **<u>the father seed = 0</u>**).
* step 2: right after step 1,
  * I use the father seed to generate **<u>200 child seeds</u>** via Numpy.
  * I use each child seed for the data generation of one repetition in each $p/n$ scenario of Simulation.
  * E.g., the $i^{th}$ child seed is used to generate the data in $i^{th}$ repetition for bsolar, solar and bolasso.
* This trick guarantees that, **even if you have to run Algorithm A and B on different machines, Algorithm A and Algorithm B are evaluated on exactly the same training / test / validation data in each repetition**. This is necessary for distributive computation.

**To make a fair runtime comparison among bsolar and bolasso**, I specifically code all 3 identically in the following structure.

  * <u>step 1</u>: compute the subsample selection frequency on the sample. Bsolar-3 trains 3 solar and bolasso trains 256 lasso.
  * <u>step 2</u>: based on the subsample selection frequency of each variable, rank all variable decreasingly.
  * <u>step 3</u>: we set the threshold as $f=0.9$ (or 1) and select the variables with subsample selection frequencies larger than that $f$.

  * To show that **the runtime difference is purely due to the fact that we replace lasso with solar in a boostrap ensemble**,
    * we **use exactly the same codes in step 1 to 3** for bsolar and bolasso.
    * **the only difference** is that we use different algorithms (solar or lasso) to estimate subsample selection frequency in the for-loop of step 1.
    * we also **use the same parallel scheme (coded exactly the same)** for bsolar and bolasso.
    * in case someone claims that our parallel scheme is too aggressive, we also report **the runtime of bolasso using the default SciKit-learn parallel scheme, which is optimzed for the trainnig of lasso and, hence, bolasso.**

</font>

<br>

#### #1(d). Replication

* <font size="4.5"> To replicate the simulation **after you read though the detailed explanation and comments in each ".ipynb" file**, you just need to
  * open each ".ipynb" file in <u>Jupyter Lab (or Jupyter notebook)</u>,
  * click the <u>"Kernel"</u> menu
  * click <u>"Restart Kernel and Run All Cells"</u>.
  * **you may want to read the comments in ".ipynb" files carefully before you replicate bolasso simulations (since it could take very long time).**

* Python Package dependence : reported at **#2.**
    
</font>

<br>

### #2. Package dependence

#### the package has been tested with the following dependence:

* <font size="4.5"> **<u>Python 3.7.10</u>**;
* **<u>scikit-learn 0.24.1</u>**;
  * check the version index since scikit-learn will remove the automatic variable normalization feature after version 1.2;
* the Fortran/C++ library **<u>Intel MKL 2021</u>**;
  * Note that Intel MKL is known to be clearly faster than openBLAS on matrix operation;
* **<u>numpy 1.19.2</u>** or, equivalently, **<u>jax.numpy</u>** of **<u>jax 0.2.10</u>**;
  * if you are working on large datasets (such as picture recognition, natural language processing and text mining with dimension $p>1k$ ), one bolasso realization could take hours. Hence, I strongly recommend that
    * use **<u>jax</u>** for much quicker matrix operation in bolasso;
    * use **<u>incomplete cholesky decomposition</u>** and **<u>covariance updating</u>** instead of the default one in Scikit-learn for bolasso.
* **<u>jupyter 1.0.0</u>** and **<u>jupyterlab 3.0.11</u>**;
* **<u>matplotlib 3.3.4</u>**;
* **<u>statsmodels 0.12.2</u>**;
* **<u>tqdm 4.59.0</u>**;
  * *2021/02/13: among all runtime measure packages, tqdm takes the least runtime overhead (80ns). This guarantees the accuracy of runtime measurement.*
* **<u>joblib 1.0.1</u>**;

</font>

<br>

### #3. File structure
> <font size="4.5"> the detailed explanations and comments can be found at the begining and each step of the ".py" or ".ipynb" file. I only introduce the structure here.</font>

#### #3(a) ./raw_results
> <font size="4.5"> all the raw results with detailed explanations (in .html files) </font>

- **Section 2**
  - **solar\_demo** , **bsolar\_walkthrough** and **bolasso\_walkthrough.html** : the step-by-step explanation, evaluation and demostration for the code of solar, bsolar and bolasso.

- **Section 3**
  
  - **example_split.html** : the raw results of the post-solar/lasso data splitting test (Section 3.1 of the paper) with detailed explanations.

  - **example_IRC.html** : the raw results of the variable selection accuracy simulation under different irrepresetible condition settings(Section 3.3 of the paper) with detailed explanations.

- **Section 4**
  
  - **simul_solar_lasso.html** : the sparsity and accuracy results of the simulation (solar and lasso, Section 4.4) with detailed explanations.

  - **sparsity_accuracy_bolasso_bsolar.html** : the sparsity and accuracy results (bsolar and bolasso, Section 4.4) with detailed explanations.

  - **subsample_frequency_bolasso_bsolar.html** : the subsample selection frequency comparison (bsolar vs bolasso, Section 4.5) with detailed explanations.

  - **bsolar_runtime**, **bolasso_lars_runtime**, **runtime_plot**, and **bolasso_cd_runtime.html** : the runtime comparison (Section 4.6) with detailed explanations.

- **Section 5**
  
  - **application\_Houseprice\_linear** and **application\_Houseprice\_log.html** : the raw result of the real-world application (Section 5) with detailed explanations.  

<br>

#### #3(b) ./example_test
> <font size="4.5"> the Python packages and codes for <u>Example 1, Section 3.2</u> </font>

- **simulator.py** : the data generating package.
- **costcom.py** : the package to compute the regression error.
- **solar_parallel.py** : the solar (parallel computation) package.
- **example\_split.ipynb** : the simulation for Example 1.
- **debug.sh** : (for macOS and Linux only) the bash file for bug testing of all .py files here.
  * in Mac OS or Linux, open terminal and switch to this folder; run "bash debug.sh" commmand
  * it will produces all the test plots, results and tables;
  * if you find no error during the procedure and the bash file ends normally, there is no bug of all the packages in this folder.

<br>

#### #3(c) ./example_IRC

> <font size="4.5"> the Python package and simulation for <u>IRC Example, Section 3.4</u>. </font>

- **./figures** : the folder of all detailed graphical results, saved as ".pdf";
- **costcom** and **solar_paralle.py** and **debug.sh** : same as previous;
- **simul_plot_ic.py** : all the simulation functions (computation and plotting functions) that the IRC example requires;
- **simulator_ic.py** : the data generating package for the IRC example only.
- **example_IRC.ipynb** : the simulation for the lasso-solar example under different irrepresentable conditions.

<br>

#### #3(d) ./simul_lasso_solar

> <font size="4.5"> the Python packages and simulation for lasso, solar and solar + holdout at <u>Section 4</u> </font>

- **./figures**, **costcom**, **simulator**, **solar_paralle.py** and **debug.sh**: same as previous;
- **simul_plot.py** : all the simulation functions (computation and plotting functions) that solar and lasso require in the Simulation;
- **simul_solar_lasso.ipynb** : the simulation for lasso, solar and "solar + hold out".

<br>

#### #3(e) ./simul_bolasso_bsolar

> <font size="4.5"> the Python package and simulation for bolasso and bsolar at <u>Section 4</u> </font>

* **./subsample_selection_frequency**
  > <font size="4.5"> the folder for the comparison of subsample selection frequency tables between bsolar and bolasso </font>

  - **costcom**, **simulator**, **solar** and **solar_parallel.py** and **debug.sh** : same as previous;
  - **bolasso_parallel.py** : the bolasso package under the Scikit-learn built-in parallel computing scheme;
  - **bootstrap_demo_parallel.py** : the simulation functions required for subsample selection frequency comparison;
  - **bsolar.py** : the bsolar package under sequential computing scheme;
  - **subsample_frequency_bolasso_bsolar.ipynb** : the simulation for subsample selection frequency comparison.
  
<br>

* **./sparsity_accuracy**
  > <font size="4.5"> the sparsity and accuracy of bsolar and bolasso </font>

  - **costcom**, **simulator**, **solar**, **solar_parallel**, **bolasso_parallel** and **bsolar.py** and **debug.sh** : same as previous;
  - **bsolar_parallel.py** : the bsolar package under customized Joblib parallel computing scheme;
  - **simul_plot_parallel.py** : all the simuation function that sparsity comparison requiress
  - **sparsity_accuracy_bolasso_bsolar.ipynb** : the simulation for sparsity and accuracy comparison.
  
<br>

* **./runtime**
  > <font size="4.5"> the runtime comparison between bsolar and bolasso </font>

  - **costcom.py, simulator.py, debug.sh, solar.py, solar_paralle.py, bolasso_parallel.py, bsolar.py, bsolar_parallel.py** : same as previous;
  - **simul_built_in_parallel** and **simul_joblib_parallel.py** : the simulation functions, respectively, for the Scikit-learn build-in parallel scheme and customized Joblib scheme.
  - **bsolar_runtime**, **bolasso_cd_runtime**, **bolasso_lars_runtime**, and **runtime_plot.ipynb** : the runtime of bsolar and bolasso (solved by lars and warm-start, pathwise coordinate descent).

<br>

#### #3(f) ./application

> <font size="4.5"> the real-world application at <u>Section 5</u> </font>

- **costcom.py, simulator.py, solar.py** : same as previous;
- **Data2010.csv** : data
- **House2010_linear** and **House2010_log.csv** : variables in linear and log forms, generated based on <u>Data2010.csv</u>;
- **application_Houseprice_linear** and **application_Houseprice_log.ipynb**: the real-world application in both linear and log forms;

<br>

#### #3(g) ./demo

> <font size="4.5"> the step-by-step walkthrough of "bolasso_parallel", "solar_parallel" and "bsolar_parallel.py"</font>

- **costcom.py, simulator.py, debug.sh, solar.py, solar_parallel.py** : same as previous.
- **solar_simul_demo.py** : all the simulation functions that the solar demostration needs.
- **bsolar_walkthough**, **bolasso_walkthough** and **solar_demo.ipynb** : the step-by-step demonstration for bsolar, bolasso and solar.
