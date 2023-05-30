# OEDISI DOPF
Open Energy Data Initiative - Solar Systems Integration Data and Analytics (OEDI SI) Distributed Optimal Power Flow

## Setup

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Test
the oedisi-cli tool needs to be in the base directory that the system and components are defined within. To test the setup move into the test folder and run the shell script.

```shell
cd test 
./run.sh
```

## Build and Run
Replace the "<path>" below to point to the desired scenario

```shell
oedisi build --component-dict scenario/<path>/components.json --system scenario/<path>/system.json
oedisi run
```

## Helics Kill
If you get a multiple broker error from helics just run the following commands. 

```shell
pkill -9 helics_broker
pkill -9 python
```
