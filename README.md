# Deep Learning Setup Scripts

Michelle L. Gill, Ph.D.  
michelle@michellelynngill.com  
https://github.com/mlgill/deeplearn_setup_scripts

======================

This is my custom installation and setup script for Ubuntu 16.04. 

I use this on my personal deep learning box and on AWS GPU instances. This script installs a standard set of development tools (gcc, g++, git, make, cmake, tmux, etc.) and either the Anaconda or Miniconda python distribution. 

Three conda environments are created, one called `scienv3` that contains a variety of data science-focused libraries (numpy, scipy, scikit-learn, opencv, jupyter notebook, etc.). Two deep learning libraries are also created (`deeplearn` and `deeplearn_caffe`). Both libraries conda the packages of the `scienv3` library plus theano, tensorflow, and keras. The `deeplearn_caffe` library also contains caffe.
