#!/bin/bash

export DEBIAN_FRONTEND="noninteractive"
export TZ="America/Pacific" 
pwd=$(pwd)

cd $pwd/build 
cd gridappsd-state-estimator 
make -C state-estimator clean
bear --output state-estimator/compile_commands.json -- make -j4 -C state-estimator gadal
cp state-estimator/bin/state-estimator-gadal $pwd/ekf_federate/
