import torch
import os
from bidict import bidict
from typing import Literal, Union
from alive_progress import alive_bar as aliveBar

from language_modeling.utils import AnnLanguageModelDataset
from language_modeling.config import ANN_MODEL_PATH, PAD_TOKEN
from language_modeling import BaseLanguageModel

class AnnLanguageModel(BaseLanguageModel):
	def __init__(self, vocabulary : bidict, 
			  	 pretrainedEmbeddings : torch.Tensor = None, 
				 fineTunePretrained : bool = False, 
				 contextSizePrev : int = 5, 
				 contextSizeNext : int = 0, 
				 embeddingSize : int = 300, 
				 activation : Literal["tanh", "sigmoid", "relu"] = "relu",
				 droupout : float = 0.75,
				 hiddenLayerSizes : Union[list[int], None] = None) -> None:
		"""
			`pretrainedEmbeddings` are assumed to be indexed with the same vocabulary as `trainDataset`.
			If set to `None`, the pretrained embeddings will be randomly initialized with `torch.nn.Embedding`.
		"""
		super().__init__(vocabulary, pretrainedEmbeddings, fineTunePretrained)

		self.contextSizePrev = contextSizePrev
		self.contextSizeNext = contextSizeNext
		self.embeddingSize = embeddingSize

		self.softmax = torch.nn.Softmax(dim=1)
		self.activation = None
		if activation == "tanh":
			self.activation = torch.nn.Tanh()
		elif activation == "sigmoid":
			self.activation = torch.nn.Sigmoid()
		elif activation == "relu":
			self.activation = torch.nn.ReLU()
		assert self.activation is not None

		self.dropoutLayer = torch.nn.Dropout(droupout)
		self.hiddenLayers = torch.nn.Sequential()
		if hiddenLayerSizes is not None and len(hiddenLayerSizes) > 0:
			self.hiddenLayers.append(torch.nn.Linear((self.contextSizePrev + self.contextSizeNext) * self.pretrainedEmbeddingSize, hiddenLayerSizes[0]))
			for i in range(1, len(hiddenLayerSizes)):
				self.hiddenLayers.append(self.activation)
				self.hiddenLayers.append(torch.nn.Linear(hiddenLayerSizes[i - 1], hiddenLayerSizes[i]))
				if (i%3) == 0:
					self.hiddenLayers.append(self.dropoutLayer)
			self.hiddenLayers.append(self.activation)
			self.hiddenLayers.append(torch.nn.Linear(hiddenLayerSizes[-1], self.vocabSize))
		else:
			self.hiddenLayers.append(torch.nn.Linear((self.contextSizePrev + self.contextSizeNext) * self.pretrainedEmbeddingSize, self.vocabSize))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._modelSaveDir_ = ANN_MODEL_PATH
		self._modelName_ = f"nnlm_{self.activation}_{self.contextSizeNext}_{self.contextSizePrev}_{self.embeddingSize}_{self.pretrainedEmbeddingSize}"

	def forward(self, x : torch.Tensor):
		"""
		x is of shape (batchSize, contextSizePrev + contextSizeNext)
		each row in x contains indices of the tokens in the context
		
		Returns:
			next word probability distribution (batchSize, vocabSize)
			embedding of this word (batchSize, embeddingSize)
		"""
		x = self._getPretrainedEmbeddings_(x) # (batchSize, contextSizePrev + contextSizeNext, pretrainedEmbeddingSize)
		x = x.view(-1, (self.contextSizePrev + self.contextSizeNext) * self.pretrainedEmbeddingSize) # (batchSize, (contextSizePrev + contextSizeNext) * pretrainedEmbeddingSize)

		x = self.hiddenLayers(x) # (bachSize, vocabSize)
		# x = self.softmax(x)
		x = self.activation(x)

		return x, None

	def getNextWordDistribution(self, x : torch.Tensor) -> torch.Tensor:
		"""
		:param x: (batchSize, context) or (context, ). `x` contains indices of context words.
		
		Returns
			(batchSize, vocabSize) or (vocabSize, )
			The probability distribution for the next word.
		"""
		originalDim = x.ndim
		if originalDim:
			# make x of shape (1, context)
			x = x.unsqueeze(0)

		prevContextTensor = None
		if (x.size(1) > self.contextSizePrev):
			prevContextTensor = x[:, :self.contextSizePrev]
		else:
			prevContextTensor = torch.cat(
				[torch.full((x.size(0), self.contextSizePrev - x.size(1)), self.vocabulary[PAD_TOKEN]), x],
				axis=1
			)

		contextTensor = prevContextTensor
		if self.contextSizeNext > 0:
			contextTensor = torch.cat(prevContextTensor, torch.full((x.size(0), self.contextSizeNext), self.vocabulary[PAD_TOKEN]))
	
		with torch.no_grad():
			nextWordDistribution, _ = self.forward(contextTensor)
		
		if originalDim == 1:
			# return a tensor of shape (vocabSize, )
			return nextWordDistribution.squeeze(0)
		return nextWordDistribution