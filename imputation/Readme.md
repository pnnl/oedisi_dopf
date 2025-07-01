# Imputation

This imputation algorithm creates a dataset to train on provided OpenDSS model and then the data is used for training and testing, this can later be used for predicting imputed data.

## Setup

This neural network architecture represents a sophisticated deep learning approach specifically designed for power system injection prediction tasks.!This feed-forward neural network employs a progressive compression methodology combined with comprehensive regularization techniques to achieve 
robust predictive performance while maintaining computational efficiency.


```shell
git clone https://github.com/pnnl/oedisi_dopf.git
cd oedisi_dopf/imputation
poetry update
poetry lock
```

## Run

### Extract Features
The first step is to extract features to train TimeGAN with. Run the extract_features.py command with the path to the model folder and the name of the model that will be used for storing the new synthetic information.

```shell
poetry run python src/feature_extraction.py <model_path> <model_name>
```

#### Example:
```shell
poetry run python src/feature_extraction.py ../builds/lindistflow/ieee123/feeder/opendss ieee123
```
### Train Model
Once the features have been extracted, the main.py function will load the features from the specified model_name and generate new synthetic profiles for each feature.

```shell
poetry run python src/main.py <model_name>
```

#### Example:
```shell
poetry run python src/main.py ieee123
```
### Evaluate 
If you are interested in evaluating the new profiles, you can generate the maximum mean discrepancy (MMD) for random samples and visualize the t-SNE plot for model inspection. run the evaluate.py

```shell
poetry run python src/evaluate.py <model_name>
```

#### Example:
```shell
poetry run python src/evaluate.py ieee123
```
