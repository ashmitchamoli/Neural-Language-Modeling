import pickle as pkl
from sklearn.model_selection import train_test_split

from language_modeling.utils import Tokenizer

tokenizer = Tokenizer("../data/Auguste_Maquet/Auguste_Maquet.txt")
tokens, vocab = tokenizer.getTokens()

trainRatio = 0.70
testRatio = 0.15
valRatio = 0.15

trainSet, tempSet = train_test_split(tokens, train_size=trainRatio)
testSet, valSet = train_test_split(tempSet, train_size=testRatio/(testRatio + valRatio))
print(len(trainSet), len(testSet), len(valSet))

pkl.dump((vocab, trainSet, testSet, valSet), open("../data/Auguste_Maquet/data_split.pkl", "wb"))