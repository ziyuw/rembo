## REMBO
This package contains code for the paper "Bayesian Optimization in a Billion 
Dimensions via Random Embeddings". The paper tries to solve the problem of 
doing Bayesian optimization in high dimensions. For more details please read 
the paper (http://www.cs.ubc.ca/~ziyuw/papers/rembo.pdf). 

### INSTALL
The package is written in Matlab. Thus to install, just run startup.m. This 
package could make use of the function "fminsearch" from Matlab. 
This, however, is not necessary.

To run the lpsolve example from the paper, please download the data files from:
http://www.cs.ubc.ca/~ziyuw/AC_blackbox_eval.tar.
And untar the data files to the folder /demos/lpsolve/.

### Configuration files
This package allows one the specify the parameters settings in 
a configuration file. In the configuration file, one can specify
the type of parameters to be optimized. 
Currently, the package supports continuous, discrete, and categorical
parameters. It also allows one to use log scales. 
For an example of such a configuration file, please checkout 
"demos/lpsolve/lpsolve.yaml".
