from language_modeling.models import AnnLanguageModel
from language_modeling.utils import Tokenizer, AnnLMDataset

trainTokens = Tokenizer("data/Auguste_Maquet/train.txt").getTokens()
valTokens = Tokenizer("data/Auguste_Maquet/val.txt").getTokens()
testTokens = Tokenizer("data/Auguste_Maquet/test.txt").getTokens()

trainDataset = AnnLMDataset(trainTokens)
valDataset = AnnLMDataset(valTokens)
testDataset = AnnLMDataset(testTokens)

model = AnnLanguageModel(vocab=trainDataset.vocabulary)
