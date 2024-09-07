import torch
import pickle as pkl

from language_modeling.models import AnnLanguageModel
from language_modeling.utils import AnnLanguageModelDataset
from pretrained_embeddings import loadPretrained

vocab, trainTokens, valTokens, testTokens = pkl.load(open("data/Auguste_Maquet/data_split.pkl", "rb"))

modelHyperParams = {
	"contextSizePrev": 5,
	"contextSizeNext": 0,
	"embeddingSize": 512,
	"activation": "relu",
	"droupout": 0.75,
	"hiddenLayerSizes": [],
	"fineTunePretrained": False
}

trainingConfig = {
	"batchSize": 256,
	"learningRate": 0.0005,
	"epochs": 3
}


if __name__ == "__main__":
	pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
	trainDataset = AnnLanguageModelDataset(trainTokens, vocab, modelHyperParams["contextSizePrev"], modelHyperParams["contextSizeNext"])
	valDataset = AnnLanguageModelDataset(valTokens, vocab, modelHyperParams["contextSizePrev"], modelHyperParams["contextSizeNext"])
	testDataset = AnnLanguageModelDataset(testTokens, vocab, modelHyperParams["contextSizePrev"], modelHyperParams["contextSizeNext"])
	model = AnnLanguageModel(vocabulary=vocab,
							pretrainedEmbeddings=pretrainedW2v,
							**modelHyperParams)

	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=trainingConfig["batchSize"], shuffle=True)
	model.train(trainLoader=trainLoader, **trainingConfig)