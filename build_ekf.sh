#!/bin/bash

export DEBIAN_FRONTEND="noninteractive"
export TZ="America/Pacific" 
pwd=$(pwd)

sudo apt update & apt upgrade -y

# install pyton3.12
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
tar -xf Python-3.12.0.tgz
cd Python-3.12.0
./configure --enable-optimizations
make -j 8
make altinstall


sudo apt install -y libboost-dev libzmq5-dev git cmake-curses-gui build-essential m4 wget libaprutil1-dev liblapack-dev libblas-dev libssl-dev

mkdir $pwd/build 
cd $pwd/build 

git clone https://github.com/GMLC-TDC/HELICS 
cd HELICS 

mkdir build 
cd build 
cmake -DHELICS_BUILD_CXX_SHARED_LIB=True -DCMAKE_CXX_STANDARD=20 ../ 
make
sudo make install

cd $pwd/build 
wget http://archive.apache.org/dist/activemq/activemq-cpp/3.9.5/activemq-cpp-library-3.9.5-src.tar.gz 
tar -xzf activemq-cpp-library-3.9.5-src.tar.gz 
cd activemq-cpp-library-3.9.5 
./configure 
make 
sudo make install 

cd $pwd/build 
git clone --branch OEDISI.1.1.2_ekf_fixes https://github.com/GRIDAPPSD/gridappsd-state-estimator 
cd gridappsd-state-estimator 
git clone https://github.com/GRIDAPPSD/SuiteSparse 
git clone https://github.com/GRIDAPPSD/json 
LD_LIBRARY_PATH=/build/gridappsd-state-estimator/SuiteSparse/lib/ 
make -C SuiteSparse LAPACK=-llapack BLAS=-lblas 
make -C state-estimator gadal
cp state-estimator/bin/* $pwd/ekf_federate/
