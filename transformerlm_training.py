import torch
import pickle as pkl

from language_modeling.models import TransformerLanguageModel
from language_modeling.utils import TransformerLanguageModelDataset
from preprocessing_scripts.pretrained_embeddings import loadPretrained

vocab, trainTokens, valTokens, testTokens = pkl.load(open("data/Auguste_Maquet/data_split.pkl", "rb"))

modelHyperparams = {
	"fineTunePretrained": False,
	"numLayers": 1,
	"nhead": 8,
	"dimFeedforward": 2048,
	"activation": "gelu",
	"dropout": 0.0,
	"linearClassifierLayers": [1024]
}

trainingConfig = {
	"batchSize": 64,
	"learningRate": 1e-2,
	"epochs": 3,
	"retrain": True
}

if __name__ == "__main__":	
	pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
	trainDataset = TransformerLanguageModelDataset(trainTokens, vocab)
	valDataset = TransformerLanguageModelDataset(valTokens, vocab)
	testDataset = TransformerLanguageModelDataset(testTokens, vocab)

	model = TransformerLanguageModel(vocabulary=vocab,
							  		 pretrainedEmbeddings=pretrainedW2v,
							  		 **modelHyperparams)

	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=trainingConfig["batchSize"], shuffle=True, collate_fn=trainDataset.customCollate)
	valLoader = torch.utils.data.DataLoader(valDataset, batch_size=4, shuffle=True, collate_fn=valDataset.customCollate)
	model.train(trainLoader, valLoader=valLoader, **trainingConfig)