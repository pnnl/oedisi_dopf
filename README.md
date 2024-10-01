# OEDISI DOPF
Open Energy Data Initiative - Solar Systems Integration Data and Analytics (OEDI-SI) Distributed Optimal Power Flow

## Docker

Once the container is build and running, follow the link to the jupyter notebook and selet the *workflow.ipynb* notebook and follow the instructions for selecting scenarios and running the co-simulation.

- \<repo\> should be created in dockerhub before building to make pushing easier later
- \<tag\> follow tagging convention 0.0.0
- \<model\> will designate which model: ieee123, SFO/P1U, SFO/P6U, ...
- \<algo\> will specify witch dockerfile to build

```shell
    docker build -t openenergydatainitiative/<repo>:<tag> --build-arg MODEL=<model> -f Dockerfile.<algo> .
    docker run --rm -it -p 8888:8888 openenergydatainitiative/<repo>:<tag>
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
