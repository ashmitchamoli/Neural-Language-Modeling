import nltk
from bidict import bidict
from itertools import chain

from language_modeling.config import PAD_TOKEN, UNK_TOKEN, EOS_TOKEN

class Tokenizer:
	def __init__(self, filePath : str) -> None:
		self.filePath = filePath

		self.vocabulary = bidict()
		self.vocabSize = 0
		self.numTokens = 0
	
	def readText(self) -> str:
		with open(self.filePath, "r") as f:
			return f.read()
	
	def updateVocab(self, word) -> None:
		if word not in self.vocabulary:
			self.vocabulary[word] = self.vocabSize
			self.vocabSize += 1

	def getTokens(self, putEos : bool = True, replaceUnk : bool = False) -> tuple[list[list[str]], bidict]:
		"""
			:param replaceUnk: if true, replaces all infrequent words with UNK_TOKEN

		returns a list of tokenized sentences.
		"""
		self.vocabulary = bidict()
		text = self.readText()

		sentences = nltk.sent_tokenize(text)
		tokens = [nltk.word_tokenize(sentence) + ([EOS_TOKEN] if putEos else []) for sentence in sentences]

		uniqueWords = set(list(chain(*tokens)))
		for word in uniqueWords:
			self.updateVocab(word)
		self.updateVocab(PAD_TOKEN)
		self.updateVocab(UNK_TOKEN)

		return tokens, self.vocabulary
