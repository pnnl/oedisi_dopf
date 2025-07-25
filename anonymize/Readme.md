# Anonymizer

This anonymizer loads an OpenDSS model and its connected loadshape/pvshape profiles to generate new anonymized OpenDSS files. 

## Setup

This profile utilizes Differential Privacy with highest level of privacy setting to anonymise the files. Degree of anonymization can be varied manually

```shell
git clone https://github.com/pnnl/oedisi_dopf.git
cd oedisi_dopf/anonymize
poetry update
```

## Run
The main.py script will anonymize all opendss files found within the input director and save them into the specified output director under the new model name.

```shell
python main.py --model=anon123 --input=./opendss/ieee123 --output=./anon/opendss
```
