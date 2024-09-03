import torch
import os
from language_modeling.utils import AnnLanguageModelDataset

class BaseLanguageModel(torch.nn.Module):
	def __init__(self, trainDataset,
			  	 pretrainedEmbeddings : torch.Tensor = None, 
				 fineTunePretrained : bool = False) -> None:
		super().__init__()

		self.trainDataset = trainDataset
		self.vocabulary = self.trainDataset.vocabulary
		self.vocabSize = len(self.vocabulary)

		self.pretrainedEmbeddings = None
		self.pretrainedEmbeddingSize = None
		self.fineTunePretrained = fineTunePretrained
		if pretrainedEmbeddings is not None:
			self.pretrainedEmbeddings = torch.nn.Embedding.from_pretrained(pretrainedEmbeddings, freeze=(not fineTunePretrained))
			self.pretrainedEmbeddingSize = pretrainedEmbeddings.size()[1]
		else:
			self.pretrainedEmbeddings = torch.nn.Embedding(self.vocabSize, 512)
			self.pretrainedEmbeddingSize = 512
	
	def _getEmbeddings_(self, indices : torch.Tensor) -> torch.Tensor:
		if isinstance(self.pretrainedEmbeddings, torch.nn.Embedding):
			return self.pretrainedEmbeddings(indices)
		
		return self.pretrainedEmbeddings[indices]
	
	def _saveModel_(self, path : str) -> None:
		
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)
	
	def _loadModel_(self, path : str) -> bool:
		if os.path.exists(path):
			self.load_state_dict(torch.load(path))
			return True
		else:
			return False
		
	