import torch
from language_modeling.utils import AnnLMDataset

class BaseLanguageModel(torch.nn.Module):
	def __init__(self, trainDataset : AnnLMDataset) -> None:
		super().__init__()

		self.trainDataset = trainDataset
		self.vocabulary = self.trainDataset.vocabulary
		self.vocabSize = len(self.vocabulary)

	def saveModel(self, path : str) -> None:
		torch.save(self.state_dict(), path)
	