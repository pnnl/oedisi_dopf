# Anonymizer

This anonymizer loads an OpenDSS model and its connected loadshape/pvshape profiles to generate new anonymized OpenDSS files. 

## Setup

This profile utilizes Differential Privacy with highest level of privacy setting to anonymise the files. Degree of anonymization can be varied manually

```shell
git clone https://github.com/pnnl/oedisi_dopf.git
cd oedisi_dopf/anonymize
poetry update
poetry lock
```

## Run

### Anonymize Files

```shell
python main.py <feeder_name>
```

#### Example:
```
python main.py ieee123
```
### Output Files 
The files are anonymized and saved in the outputs directory
