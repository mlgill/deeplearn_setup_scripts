#!/bin/bash

####################################################################################
## Deep Learning Box Setup Script                                                 ##
## Michelle L. Gill, Ph.D.                                                        ##
## michelle@michellelynngill.com                                                  ##
## GitHub location: https://github.com/mlgill/deeplearn_setup_scripts             ##
## Date updated: 2017/06/11                                                       ##
####################################################################################

# Only setup python packages--useful for rebuilding python environment
python_only="false"

# Setup x11vnc--useful on my home deep learning box
setup_vnc="true"

# Default shell config file--works with bash and zsh
shellrc=".zshrc"

# Choose Anaconda (full distribution) or Miniconda (only install selected packages)
# anaconda_python_version="Anaconda3-latest"
anaconda_python_version="Miniconda3-latest"

# The name for the initial Python library
python_env_name="scienv3"

nvidia_cuda_version="8.0.61"
cudnn_version="5.1-0"


############################# END USER SETTINGS #############################

# BASH prompt colors
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

echo ""
echo -e $BLUE"############################################################################"$NC
echo -e $BLUE"#                     Compatible with Ubuntu 16.04                         #"$NC
echo -e $BLUE"############################################################################"$NC
echo ""
sleep 2

echo ""
echo -e $BLUE"Using: anaconda_python_version       = $anaconda_python_version"$NC
if [[ $python_only == "false" ]]; then
	echo -e $BLUE"       nvidia_cuda_version           = $nvidia_cuda_version"$NC
fi
echo -e $BLUE"       cudnn_version                 = $cudnn_version"$NC
echo ""
sleep 2

# Create download directory
if [[ ! -e $HOME/deeplearning_downloads ]]; then
	mkdir $HOME/deeplearning_downloads
fi
cd $HOME/deeplearning_downloads

if [[ $python_only == "false" ]]; then

	echo ""
	echo -e $BLUE"Installing apt-get packages"$NC
	echo ""
	sleep 1

	# Ensure system is updated and has my toolkit
	sudo apt-get update
	#sudo apt-get --assume-yes upgrade
	sudo apt-get --assume-yes install tmux vim zsh git curl openssh-server build-essential gcc g++ make cmake binutils software-properties-common graphviz 

	if [[ $setup_vnc == 'true' ]]; then

		sudo apt-get --assume-yes x11vnc

		# Configure x11vnc
		# Note that this sets up x11vnc with a global password--not ideal for multi-user or insecure environment
		sudo x11vnc -storepasswd /etc/x11vnc.pass

		echo '[Unit]
Description=Start x11vnc at startup.
After=multi-user.target
[Service]
Type=simple
ExecStart=/usr/bin/x11vnc -auth guess -forever -loop -noxdamage -repeat -rfbauth /etc/x11vnc.pass -rfbport 5900 -shared
[Install]
WantedBy=multi-user.target' | sudo tee /lib/systemd/system/x11vnc.service

		sudo systemctl enable x11vnc.service
		sudo systemctl daemon-reload
	fi

	echo ""
	echo -e $BLUE"Installing cuda drivers"$NC
	echo ""
	sleep 1

	# download and install GPU drivers
	wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${nvidia_cuda_version}-1_amd64.deb" \
	      -O "cuda-repo-ubuntu1604_${nvidia_cuda_version}_amd64.deb"
	sudo dpkg -i cuda-repo-ubuntu1604_${nvidia_cuda_version}_amd64.deb
	sudo apt-get update
	sudo apt-get -y install cuda
	sudo modprobe nvidia
	nvidia-smi
	echo "export PATH=\"/usr/local/cuda/bin:\$PATH\"" >> $HOME/$shellrc
	export PATH="/usr/local/cuda/bin:$PATH"

	# sudo ln -s -f /usr/local/cuda-8.0/targets/x86_64-linux/include/* /usr/local/cuda/include/

	# # Install cudnn libraries
	# Note this is now installed via conda since NVIDA doesn't make CUDNN available for download without a login
	# echo ""
	# echo -e $BLUE"Installing cuDNN drivers"$NC
	# echo ""

	# wget "http://mlgill.co/xxxxxx.tgz" -O "cudnn.tgz"
	# tar -zxf cudnn.tgz
	# cd cuda
	# sudo cp lib64/* /usr/local/cuda/lib64/
	# sudo cp include/* /usr/local/cuda/include/

fi

echo ""
echo -e $BLUE"Installing Anaconda"$NC
echo ""
sleep 1

# Set the default installation path for conda
if [[ $anaconda_python_version == "Anaconda"* ]]; then
	anaconda_short_python_version="anaconda3"
else
	anaconda_short_python_version="miniconda3"
fi

# Install Anaconda for current user
if [[ ! -e "$HOME/$anaconda_short_python_version" ]]; then
    # rm -rf "$HOME/$anaconda_short_python_version"

    if [[ $anaconda_short_python_version == "anaconda"* ]]; then
    	wget_url="archive"
	else
		wget_url="miniconda"
	fi

    wget "https://repo.continuum.io/${wget_url}/${anaconda_python_version}-Linux-x86_64.sh" \ 
     -O "${anaconda_python_version}-Linux-x86_64.sh"
    bash "${anaconda_python_version}-Linux-x86_64.sh" -b
fi

echo "export PATH=\"$HOME/$anaconda_short_python_version/bin:\$PATH\"" >> $HOME/$shellrc
export PATH="$HOME/$anaconda_short_python_version/bin:$PATH"

echo ""
echo -e $BLUE"Setting up root Anaconda environment"$NC
echo ""
sleep 1

# Sometimes I prefer to use conda-forge for better software availability
#conda config --prepend channels conda-forge

conda upgrade -y --all

# Packages for builing conda packages and searching other channels
conda install -y -n root conda-build anaconda-client

# Add CUDNN to the base python environment
conda install -y -n root cudnn=$cudnn_version

# Setup paths and symbolic links to the local environment
sudo ln -s -f $HOME/$anaconda_short_python_version/include/cudnn.h /usr/local/cuda/include/
sudo ln -s -f $HOME/$anaconda_short_python_version/lib/libcudnn* /usr/local/cuda/lib64/

echo "export LIBRARY_PATH=\"/usr/local/cuda/lib64:\$LIBRARY_PATH\"" >> $HOME/$shellrc
echo "export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:\$LD_LIBRARY_PATH\"" >> $HOME/$shellrc
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

echo ""
echo -e $BLUE"Setting up $python_env_name Anaconda environment"$NC
echo ""
sleep 1

# The base conda repo--"scienv3"
conda create -y -n $python_env_name
source activate $python_env_name

conda install -y -n $python_env_name bcolz ipython scipy numpy pandas scikit-learn pillow h5py py-xgboost dask matplotlib seaborn jupyter notebook gensim nltk

# The menpo version of opencv contains SIFT/SURF
conda install -y -n $python_env_name  -c menpo opencv3

echo ""
echo -e $BLUE"Installing NLTK corpora"$NC
echo ""
sleep 1

# Setup NLTK corpora
python -m nltk.downloader -d $HOME/nltk_data all
echo "export NLTK_HOME=\"$HOME/nltk_data\"" >> $HOME/$shellrc
export NLTK_HOME="$HOME/nltk_data"

echo ""
echo -e $BLUE"Configuring Jupyter notebook"$NC
echo ""
sleep 1

# Configure Jupyter
jupyter notebook --generate-config --y

# Personal server that is only accessible via SSH key so no need for token nonsense
cat $HOME/.jupyter/jupyter_notebook_config.py \
| sed "s/#c.NotebookApp.password = ''/c.NotebookApp.password = ''/g" \
| sed "s/#c.NotebookApp.token = '<generated>'/c.NotebookApp.token = ''/g" \
| sed "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" \
> $HOME/.jupyter/jupyter_notebook_config_new.py

mv -f $HOME/.jupyter/jupyter_notebook_config_new.py $HOME/.jupyter/jupyter_notebook_config.py

echo ""
echo -e $BLUE"Installing Keras, Theano, Tensorflow in 'deeplearn' environment"$NC
echo ""
sleep 1

conda create -y -n deeplearn --clone $python_env_name
source activate deeplearn

# Install and configure deep learning libraries 
conda install -y -n deeplearn cudnn=$cudnn_version theano tensorflow keras #caffe-gpu

# PyTorch installs its own CUDNN. Gross--it causes conflicts with theano. 
#conda install -y -n deeplearn -c soumith pytorch torchvision #cuda80

if [[ ! -e $HOME/.keras ]]; then
	mkdir $HOME/.keras
fi
cat > $HOME/.keras/keras.json << EOF
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
EOF

# Install and configure theano
cat > $HOME/.theanorc << EOF
[global]
device = cuda
floatX = float32
[cuda]
root = /usr/local/cuda
[dnn]
enabled = True
include_path = /usr/local/cuda/include
library_path = /usr/local/cuda/lib64
EOF


echo ""
echo -e $BLUE"******************************************************"$NC
echo -e $BLUE"** NOTE: this instance MUST be rebooted before use. **"$NC
echo -e $BLUE"******************************************************"$NC
echo ""

