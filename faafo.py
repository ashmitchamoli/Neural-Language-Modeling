from language_modeling.models import AnnLanguageModel
from language_modeling.utils import Tokenizer, AnnLMDataset

trainTokens, trainVocab = Tokenizer("data/Auguste_Maquet/train.txt").getTokens()
valTokens, valVocab = Tokenizer("data/Auguste_Maquet/val.txt").getTokens()
testTokens, testVocab = Tokenizer("data/Auguste_Maquet/test.txt").getTokens()

trainDataset = AnnLMDataset(trainTokens, trainVocab, 5, 0)
valDataset = AnnLMDataset(valTokens, trainVocab, 5, 0)
testDataset = AnnLMDataset(testTokens, trainVocab, 5, 0)

model = AnnLanguageModel(trainDataset,
						 pretrainedEmbeddings=None,
						 contextSizePrev=5,
						 contextSizeNext=0,
						 embeddingSize=300,
						 activation="tanh")
model.train(valDataset)