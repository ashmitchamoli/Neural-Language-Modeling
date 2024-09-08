import torch
import pickle as pkl

from language_modeling.models import LstmLanguageModel
from language_modeling.utils import LstmLanguageModelDataset
from pretrained_embeddings import loadPretrained

vocab, trainTokens, valTokens, testTokens = pkl.load(open("data/Auguste_Maquet/data_split.pkl", "rb"))

modelHyperparams = {
	"hiddenEmbeddingSize": 512,
	"activation": "tanh",
	"numLayers": 2,
	"dropout": 0.0,
	"bidirectional": False,
	"linearClassifierLayers": [1024]
}

trainingConfig = {
	"batchSize": 16,
	"learningRate": 5e-5,
	"epochs": 6
}

if __name__ == "__main__":	
	pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
	trainDataset = LstmLanguageModelDataset(trainTokens, vocab)
	valDataset = LstmLanguageModelDataset(valTokens, vocab)
	testDataset = LstmLanguageModelDataset(testTokens, vocab)

	model = LstmLanguageModel(vocabulary=vocab,
							  pretrainedEmbeddings=pretrainedW2v,
							  **modelHyperparams)

	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=trainingConfig["batchSize"], shuffle=True, collate_fn=trainDataset._customCollate_)
	valLoader = torch.utils.data.DataLoader(valDataset, batch_size=4, shuffle=True, collate_fn=valDataset._customCollate_)
	model.train(trainLoader, valLoader=valLoader, **trainingConfig)