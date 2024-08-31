from language_modeling.utils import Tokenizer
from sklearn.model_selection import train_test_split

tokenizer = Tokenizer("data/Auguste_Maquet.txt")
tokens = tokenizer.getTokens(putEos=False)

trainRatio = 0.7
testRatio = 0.15
valRatio = 0.15

trainSet, tempSet = train_test_split(tokens, train_size=trainRatio)
testSet, valSet = train_test_split(tempSet, train_size=testRatio/(testRatio + valRatio))

print(len(trainSet), len(testSet), len(valSet))

def writeFile(tokens : list[list[str]], filePath : str) -> None:
	with open(filePath, "w") as f:
		for sentence in tokens:
			f.write(" ".join(sentence))
			f.write("\n")

writeFile(trainSet, "data/Auguste_Maquet/train.txt")
writeFile(testSet, "data/Auguste_Maquet/test.txt")
writeFile(valSet, "data/Auguste_Maquet/val.txt")
