import torch
import pickle as pkl

import torch.utils

from language_modeling.models import AnnLanguageModel
from language_modeling.utils import AnnLanguageModelDataset
from preprocessing_scripts.pretrained_embeddings import loadPretrained

vocab, trainTokens, valTokens, testTokens = pkl.load(open("data/Auguste_Maquet/data_split.pkl", "rb"))

modelHyperparams = {
	"contextSizePrev": 5,
	"contextSizeNext": 0,
	"activation": "tanh",
	"droupout": 0.1,
	"hiddenLayerSizes": [512, 1024],
	"fineTunePretrained": False
}

trainingConfig = {
	"batchSize": 256,
	"learningRate": 1e-5,
	"epochs": 4,
	"retrain": True
}

if __name__ == "__main__":
	pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
	trainDataset = AnnLanguageModelDataset(trainTokens, vocab, modelHyperparams["contextSizePrev"], modelHyperparams["contextSizeNext"])
	valDataset = AnnLanguageModelDataset(valTokens, vocab, modelHyperparams["contextSizePrev"], modelHyperparams["contextSizeNext"])
	testDataset = AnnLanguageModelDataset(testTokens, vocab, modelHyperparams["contextSizePrev"], modelHyperparams["contextSizeNext"])
	model = AnnLanguageModel(vocabulary=vocab,
							 pretrainedEmbeddings=pretrainedW2v,
							 **modelHyperparams)

	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=trainingConfig["batchSize"], shuffle=True)
	valLoader = torch.utils.data.DataLoader(valDataset, batch_size=64, shuffle=True)
	model.train(trainLoader=trainLoader, valLoader=valLoader, **trainingConfig)
