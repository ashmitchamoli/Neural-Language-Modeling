import torch

from language_modeling import BaseLanguageModel
from language_modeling.utils import Tokenizer
from language_modeling.config import EOS_TOKEN

class Inferencer:
	def __init__(self, model : BaseLanguageModel) -> None:
		self.model = model
		self.vocabulary = model.vocabulary
	
	def getTokenIndices(self, tokens : list[str]) -> torch.Tensor:
		"""
		Returns a tensor of word indices.
		"""
		return torch.tensor([self.vocabulary[token] for token in tokens])
	
	def generateNextWordIndex(self, contextTensor : torch.Tensor = []) -> int:
		nextWordDistribution = self.model.getNextWordDistribution(contextTensor)
		return self.decodeNextWord(nextWordDistribution)

	def generateSentence(self, context : list[str] = []) -> str:
		"""
		Generates a sentence out of the given context
		"""
		currentTokenIndex = -1
		contextTensor = self.getTokenIndices(context)
		while currentTokenIndex != self.vocabulary[EOS_TOKEN]:
			currentTokenIndex = self.generateNextWord(contextTensor)
			
			# expand current context tensor
			contextTensor = torch.cat([contextTensor, torch.tensor([currentTokenIndex])])
			context.append(self.vocabulary.inverse[currentTokenIndex])
		
		return " ".join(context)

	def decodeNextWord(self, nextWordDistribution : torch.Tensor) -> int:
		"""
		Returns the index of the next word.
		"""
		return nextWordDistribution.argmax().item()

	def __call__(self, context : list[str]) -> str:
		return self.generateSentence(context)