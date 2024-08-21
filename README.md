# OEDISI DOPF
Open Energy Data Initiative - Solar Systems Integration Data and Analytics (OEDI-SI) Distributed Optimal Power Flow

## Docker

Once the container is build and running, follow the link to the jupyter notebook and selet the *workflow.ipynb* notebook and follow the instructions for selecting scenarios and running the co-simulation.

```shell
    docker build -t pnnl-dopf-lindistflow:0.0.0 .
    docker run -it -p 8888:8888 pnnl-dopf-lindistflow:0.0.0
```
#
 
## Setup

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or 

```shell
poetry update
```

### Building EKF
There is one federate that is not a python script that must be built. Use the following script which will build the dependencies and copy the exe files into the ekf_federate directory. You may be prompted to entry your password multiple times.

```shell
./build_ekf.sh
```


## Jupyter Notebook
The following notebook provides a more interactive experiance and a frontend for the container if you are running in docker. Once the notebook is running open the notebook link with it's generated token.


```shell
jupyter notebook workflow.ipynb
```

## Build and Run
Replace the \<scenario\> below to point to the desired scenario folder name

```shell
./run.sh <scenario>
```

or 


```shell
poetry ./run.sh <scenario>
```
