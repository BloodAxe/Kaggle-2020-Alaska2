# octave to interface matlab codes with python
sudo apt-get install octave octave-control octave-image octave-io octave-optim octave-signal octave-statistics

# Apex for automatic mixed precision
git clone https://github.com/NVIDIA/apex
cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install -v --no-cache-dir ./
cd ..

# OMPI
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
gunzip -c openmpi-4.0.3.tar.gz | tar xf -
cd openmpi-4.0.3
sudo ./configure --prefix=/usr/local
sudo make all install
cd ..

# g++ and gcc
# make a backup
cat /etc/apt/sources.list > sources_tmp.list

# one command including line break, add sources
sudo sh -c "echo 'deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe' >> /etc/apt/sources.list"

# install g++ and gcc
sudo apt-get update
sudo apt-get install g++-4.9 gcc-4.9

# remove those two lines from sources.list
sudo sh -c "cat sources_tmp.list > /etc/apt/sources.list"
rm sources_tmp.list


# NCCL
# You probably need this for training horovod code on multiple machines
# wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnccl2_2.6.4-1+cuda10.0_amd64.deb
# wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnccl-dev_2.6.4-1+cuda10.0_amd64.deb

# sudo dpkg -i libnccl2_2.6.4-1+cuda10.0_amd64.deb libnccl-dev_2.6.4-1+cuda10.0_amd64.deb
# sudo apt-get update

# sudo apt-get install libnccl2=2.6.4-1+cuda10.0 libnccl-dev=2.6.4-1+cuda10.0
# not sure about libcudnn7 libcudnn7-dev, but it should be installed with tensorflow / maybe even torch

# NCCL install horovod
# sudo LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip3 install --no-cache-dir horovod==0.19.1