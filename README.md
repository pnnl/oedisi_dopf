# OEDISI DOPF
Open Energy Data Initiative - Solar Systems Integration Data and Analytics (OEDI-SI) Distributed Optimal Power Flow

## Docker

To simply run the DOPF algorithm an an example scenaior fill modify the build arg *SCENARIO* to: small, medium, large, or ieee123. ieee123 is the default scenario. Outputs are save in the mounted volumn to local directory.

```shell
    docker build --build-arg SCENARIO=<ieee123> -t oedisi-example:0.0.0 .
    docker volume create --name oedisi_outputs --opt type=none --opt device=${PWD}/outputs --opt o=bind
    docker run -it --mount source=oedisi_outputs,target=/simulation/outputs oedisi-example:0.0.0
```

 
## Setup

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build and Run
Replace the \<scenario\> below to point to the desired scenario folder name

```shell
./run.sh <scenario>
```
