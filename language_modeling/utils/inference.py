import torch
import sys
import gc
from alive_progress import alive_bar as aliveBar

from language_modeling import BaseLanguageModel
from language_modeling.config import EOS_TOKEN, PAD_TOKEN

EPSILON = 1e-20

class Inferencer:
	def __init__(self, model : BaseLanguageModel) -> None:
		self.model = model
		self.vocabulary = model.vocabulary
	
	def getTokenIndices(self, tokens : list[str]) -> torch.Tensor:
		"""
		Returns a tensor of word indices.
		"""
		return torch.tensor([self.vocabulary[token] for token in tokens], dtype=torch.long)
	
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
			currentTokenIndex = self.generateNextWordIndex(contextTensor)
			
			# expand current context tensor
			contextTensor = torch.cat([contextTensor, torch.tensor([currentTokenIndex])])
			context.append(self.vocabulary.inverse[currentTokenIndex])

			print(context[-1], end=" ")
			sys.stdout.flush()
			
		
		return " ".join(context)

	def computePerplexity(self, tokens : list[list[str]]) -> float:
		"""
		Computes the perplexity of the given list of tokens.
		"""
		totalPerplexity = 0
		with aliveBar(len(tokens)) as bar:
			for sentence in tokens:
				logSentenceProb = 0
				tokenIndices = self.getTokenIndices(sentence)

				context = torch.full((len(sentence), len(sentence)), fill_value=self.vocabulary[PAD_TOKEN], dtype=torch.long)
				for i in range(1, len(tokenIndices)):
					context[i, (len(sentence) - i):] = tokenIndices[:i]
				
				probDist = self.model.getNextWordDistribution(context)
				probDist = probDist.to("cpu")

				logSentenceProb = torch.log(probDist[torch.arange(len(tokenIndices)), tokenIndices]).sum()				
				totalPerplexity += torch.exp(-logSentenceProb / len(sentence))

				# del context
				# del probDist
				# del logSentenceProb
				# del tokenIndices

				# # garbage collection
				# gc.collect()
				# with torch.no_grad():
				# 	torch.cuda.empty_cache()

				bar()


		
		return totalPerplexity.item() / len(tokens)

	def decodeNextWord(self, nextWordDistribution : torch.Tensor) -> int:
		"""
		Returns the index of the next word.
		"""
		return nextWordDistribution.argmax().item()

	def __call__(self, context : list[str]) -> str:
		return self.generateSentence(context)