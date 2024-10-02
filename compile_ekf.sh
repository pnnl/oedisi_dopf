#!/bin/bash

export DEBIAN_FRONTEND="noninteractive"
export TZ="America/Pacific" 
pwd=$(pwd)

cd $pwd/build 
cd gridappsd-state-estimator 
LD_LIBRARY_PATH=/build/gridappsd-state-estimator/SuiteSparse/lib/ 
make -C SuiteSparse LAPACK=-llapack BLAS=-lblas 
make -C state-estimator gadal
cp state-estimator/bin/* $pwd/ekf_federate/
