# Neural-Language-Modeling
In this repository, we will implement 3 Neural Language Models architectures using [PyTorch](https://pytorch.org/). We will train these models on the next word prediction task and report the performance of each model using perplexity scores.

Please read [Report.pdf](Report.pdf) for analytical details.

# Code Structure
The source code for all the language models is present in the `language_modeling` module. There are 2 submodules:
1. `utils`: Contains all the utility classes such as `Tokenizer`, `Inferencee` and dataset classes.
2. `models`: Contains all the language model classes such as `LstmLanguageModel`, `AnnLanguageModel` and `TransformerLanguageModel`.
   
Besides these submodules, there is a `BaseLanguageModel` class from which all the `LanguageModel` classes inherhit. The `BaseLanguageModel` class contains all the common methods and attributes that all the language models share. The file `config.py` contains global parameters that are used across the entire module.

# utils
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

# models
`AnnLanguageModel`

`LstmLanguageModel`

`TransformerLanguageModel`

# How to run
## Training
To train models, 3 files have been provided in the root diretory, `annlm_training.py`, `lstmlm_training.py` and `transformerlm_training.py`. To train a model, edit the `modelHyperparams` and `trainingConfig` dict in the file and run the respective script. For example, to train an `AnnLanguageModel`, run the script `annlm_training.py` after editing the `modelHyperparams` and `trainingConfig` dict.

Note: if `retrain` is set to False, the model will be loaded automatically if found in `model_checkpoints` folder.

## Perplexity Scores
If available, this file will load the model weights from the `model_checkpoints` folder and compute the perplexity scores for the test and train datasets. The perplexity scores will be saved in `perplexity_scores` folder.

## Data Preprocessing
Scripts have been provided to preprocess the data in the `preprocessing_scripts` directory. To preprocess the data in a different way, just edit the various parameters provided and run the file.
