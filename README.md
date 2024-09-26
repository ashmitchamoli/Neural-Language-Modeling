# Neural-Language-Modeling
In this repository, we will implement 3 Neural Language Models architectures using [PyTorch](https://pytorch.org/). We will train these models on the next word prediction task and report the performance of each model using perplexity scores.

# Code Structure
The source code for all the language models is present in the `language_modeling` module. There are 2 submodules:
1. `utils`: Contains all the utility classes such as `Tokenizer`, `Inferencee` and dataset classes.
2. `models`: Contains all the language model classes such as `LstmLanguageModel`, `AnnLanguageModel` and `TransformerLanguageModel`.
   
Besides these submodules, there is a `BaseLanguageModel` class from which all the `LanguageModel` classes inherhit. The `BaseLanguageModel` class contains all the common methods and attributes that all the language models share. The file `config.py` contains global parameters that are used across the entire module.

# `utils`
## `inference.py`
This file provides the `Inferencer` class. This class takes a `*LanguageModel` object as input and provides several useful methods.
1. `generateSentence(context)`: Generates a sentence out of the given context.
2. `computePerplexity(tokens)`: Computes the perplexity of the given list of tokens and save it to a file if specified.

## `tokenizer.py`
This file provides the `Tokenizer` class which takes path to a text file as input. This class is used to tokenize the input text and convert it into a list of tokens. The `Tokenizer` class provides the following methods:
1. `getTokens()`: Returns a list of tokenized sentences from the input text file or from a string if hinted using the `fromString` parameter.
2. `readText()`: Reads the input text file and returns it as a string.

This class uses the NLTK library for tokenization.

## `datasets.py`
This file contains a dataset class for each of the 3 language models for the next word prediction task, namely, `AnnLanguageModelDataset`, `LstmLanguageModelDataset` and `TransformerLanguageModelDataset`. Each dataset class inherits from the `torch.utils.data.Dataset` class.

# `models`
In this library, code for 3 language models is provided:
1. `AnnLanguageModel`
2. `LstmLanguageModel`
3. `TransformerLanguageModel`

These classes are inherited from the `BaseLanguageModel` class and were implemented using pytorch.

# How to run
## Training
To train models, 3 files have been provided in the root diretory, `annlm_training.py`, `lstmlm_training.py` and `transformerlm_training.py`. To train a model, edit the `modelHyperparams` and `trainingConfig` dict in the file and run the respective script. 

For example, to train an `AnnLanguageModel`, run the script `annlm_training.py` after editing the `modelHyperparams` and `trainingConfig` dict.

Note: if `retrain` is set to False, the model will be loaded automatically if found in `model_checkpoints` folder.

## Perplexity Scores
If available, this file will load the model weights from the `model_checkpoints` folder and compute the perplexity scores for the test and train datasets. The perplexity scores will be saved in `perplexity_scores` folder.

## Data Preprocessing
Scripts have been provided to preprocess the data in the `preprocessing_scripts` directory. To preprocess the data in a different way, just edit the various parameters provided and run the file.

# Analysis
## Perplexity Scores
In the table below, we can see the best perplexity scores obtained for each model on the test and train sets.

| Model | Train Perplexity | Test Perplexity |
| --- | --- | --- |
| Ann LM | 148.20 | 287.37 |
| LSTM LM | 139.91 | 200.25 |
| Transformer LM | 88.85 | 185.86 |

We can see that the transformer model performs the best among the 3, followed by the LSTM model and the ANN model.

## Hyperparameters
The following table shows the hyperparameters vs perplexity scores for the ANN language model.

Activation | Hidden Layer Sizes | Train Perplexity | Test Perplexity
| --- | --- | --- | --- |
| ReLU | [512] | 258.88 | 499.28 |
| ReLU | [1024] | 213.02 | 602.82 |
| ReLU | [512, 1024] | 134.45 | 566.55 |
| Tanh | [512] | 214.23 | 511.63 |
| Tanh | [1024] | 175.39 | 478.03 |
| Tanh | [512, 1024] | 148.20 | 287.37 |

We can see that the best hyperparameters for the ANN model are `Tanh` activation and hidden layer sizes `[512, 1024]`. All the `Tanh` models perform better than the `ReLU` models. This could be because `Tanh` provides more non-linearity in the shallower models while `ReLU` might perform better in deeper models.

# Model Checkpoints
Model checkpoints can be downloaded from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ashmit_chamoli_students_iiit_ac_in/Eotng8QleXlBon85FleTbSwBMBPc9VohEuUscDORbCw07w?e=bm9ecG). Put the downloaded checkpoints in the script directory or specify the `searchPath` parameter in the `*LanguageModel.loadModelWeights()` methods, where `searchPath` is the path to the folder containing the `model_checkpoints` folder.
