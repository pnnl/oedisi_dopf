# Imputation

This imputation algorithm creates a dataset to train on provided OpenDSS model and then the data is used for training and testing, this can later be used for predicting imputed data.

## Setup

This neural network architecture represents a sophisticated deep learning approach specifically designed for power system injection prediction tasks.!This feed-forward neural network employs a progressive compression methodology combined with comprehensive regularization techniques to achieve  robust predictive performance while maintaining computational efficiency.


```shell
git clone https://github.com/pnnl/oedisi_dopf.git
cd oedisi_dopf/imputation
poetry update
poetry lock
```

## Run

### Prepare Training Data
The first step is to extract and develop a training dataset. Run the ./src/data_for_imputation_new.py command to prepapre the training dataset.

```shell
python /src/data_for_imputation_new.py
```


### Train Model
Once the input data has been extracted, the InjPred_Train.py code is run to train the model

```shell
python /src/InjPred_Train.py
```
### Imputation Prediction
To predict the imputed data, the Injection_Prediction.py is run. 
```
### Evaluate 

```shell
poetry run python /src/Injection_Prediction.py
```
