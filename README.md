# OEDISI DOPF
Open Energy Data Initiative - Solar Systems Integration Data and Analytics (OEDI SI) Distributed Optimal Power Flow

## Build and Run

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
oedisi build --component-dict scenario/test/components.json --system scenario/test/system.json
oedisi run
```

## Helics Kill

```shell
pkill -9 helics_broker
pkill -9 python
```
