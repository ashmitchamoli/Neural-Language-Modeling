import torch
from bidict import bidict
from typing import Literal, Union, Callable

from language_modeling import BaseLanguageModel
from language_modeling.config import PAD_TOKEN, TRANSFORMER_MODEL_PATH

class TransformerLanguageModel(BaseLanguageModel):
	def __init__(self, vocabulary : bidict, 
			  	 pretrainedEmbeddings : torch.Tensor,
				 fineTunePretrained : bool = False,
				 numLayers : int = 2,
				 nhead : int = 8,
				 dimFeedforward : int = 2048,
				 activation : Union[Literal["relu", "gelu"], Callable[[torch.Tensor], torch.Tensor]] = "relu",
				 dropout : float = 0.2,
				 linearClassifierLayers : list[int] | None = None) -> None:
		
		super().__init__(vocabulary, pretrainedEmbeddings, fineTunePretrained)
		
		# self.encoderLayer = torch.nn.TransformerEncoder(
		# 	d_model=self.pretrainedEmbeddingSize,
		# 	nhead=nhead,
		# 	dim_feedforward=dimFeedforward,
		# 	dropout=dropout,
		# 	activation=activation,
		# 	batch_first=True
		# )
		self.encoder = torch.nn.TransformerEncoderLayer(
			d_model=self.pretrainedEmbeddingSize,
			nhead=nhead,
			dim_feedforward=dimFeedforward,
			dropout=dropout,
			activation=activation,
			batch_first=True
		)
		self.encoderLayer = torch.nn.TransformerEncoder(self.encoder, num_layers=numLayers)

		self.linear = torch.nn.Sequential()
		if linearClassifierLayers is None or len(linearClassifierLayers) == 0:
			self.linear.append(torch.nn.Linear(self.pretrainedEmbeddingSize, self.vocabSize))
		else:
			self.linear = torch.nn.Sequential(torch.nn.Linear(self.pretrainedEmbeddingSize, linearClassifierLayers[0]))
			for i in range(1, len(linearClassifierLayers)):
				self.linear.append(torch.nn.Tanh())
				self.linear.append(torch.nn.Linear(linearClassifierLayers[i - 1], linearClassifierLayers[i]))
			self.linear.append(torch.nn.Tanh())
			self.linear.append(torch.nn.Linear(linearClassifierLayers[-1], self.vocabSize))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._modelSaveDir_ = TRANSFORMER_MODEL_PATH
		self._modelName_ = f"transformerlm_{nhead}_{dimFeedforward}_{self.pretrainedEmbeddingSize}_{activation}_{dropout}_{linearClassifierLayers}"

	def _getPositionalEncoding_(self, maxSequenceLength : int, embeddingSize : int, device : torch.device) -> torch.Tensor:
		pe = torch.arange(0, maxSequenceLength).unsqueeze(1) # (maxSequenceLength, 1)
		temp = torch.arange(0, embeddingSize).unsqueeze(0) # (1, embeddingSize)
		pe = (pe / 10000 ** ((temp - temp%2) / embeddingSize)) # (maxSequenceLength, embeddingSize)
		pe[:, 0::2] = torch.sin(pe[:, 0::2])
		pe[:, 1::2] = torch.cos(pe[:, 1::2])

		return pe.to(device)

	def forward1(self, x : torch.Tensor) -> torch.Tensor:
		"""
		x is of shape (batchSize, maxSequenceLength)
		"""
		x = self._getPretrainedEmbeddings_(x) # (batchSize, maxSequenceLength, pretrainedEmbeddingSize)
		x = x + self._getPositionalEncoding_(x.size(1), self.pretrainedEmbeddingSize, self.device)
		
		# masked attention
		# causalMask is a square matrix with upper traiangle filled with -inf and lower triangle filled with 0 with diagonal also 0
		causalMask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1), device=self.device) # (maxSequenceLength, maxSequenceLength)
		x = self.encoderLayer.forward(x, mask=causalMask, is_causal=True) # (batchSize, sequenceLength, pretrainedEmbeddingSize)
		
		return x

	def forward(self, x) -> torch.Tensor:
		"""
		x is of shape (batchSize, maxSequenceLength)
		"""
		# pick the last word embedding from each sequence. The last word will be the token just before the first <PAD> token
		# find the index of the first PAD token in each sequence
		# firstPadIndices = torch.argmax((x == self.vocabulary[PAD_TOKEN]).int(), dim=1) # (batchSize,)
		# firstPadIndices = firstPadIndices + (firstPadIndices == 0)

		x = self.forward1(x) # (batchSize, maxSequenceLength, pretrainedEmbeddingSize)

		# pick the last word embedding from each sequence
		# x = x[torch.arange(x.size(0)), firstPadIndices - 1, :] # (batchSize, pretrainedEmbeddingSize)

		# linear classifier
		x = self.linear(x) # (batchSize, maxSequenceLength, vocabSize)

		return x.view(-1, x.size(2)), None
	
	def getNextWordDistribution(self, x : torch.Tensor) -> torch.Tensor:
		"""
		:param x: (batchSize, context) or (context, ). `x` contains indices of context words.

		Returns
			(batchSize, vocabSize) or (vocabSize, )
			The probability distribution for the next word.
		"""
		# # self.to(self.device)
		# # if x.ndim == 1:
		# # 	# make x of shape (1, context)
		# # 	x = x.unsqueeze(0)
		
		# # # reverse each row in x
		# # x = x.flip(1)

		# # # since x might be too long, we need to split it into batches
		# # finalOutput = torch.zeros(x.size(0), self.vocabSize, device=self.device)
		# # batchSize = 32
		# # x = list(x.split(batchSize, dim=0))
		# # for i in range(len(x)):
		# # 	xi = x[i].to(self.device) # (batchSize, context)

		# # 	with torch.no_grad():
		# # 		output = self(xi)[0] # (batchSize, vocabSize)

		# # 	finalOutput[i * batchSize : (i + 1) * batchSize, :] = output
		
		# # if finalOutput.size(0) == 1:
		# # 	return torch.nn.Softmax(dim=0)(finalOutput.squeeze(0))
		
		# # return torch.nn.Softmax(dim=1)(finalOutput)
		# if x.ndim == 1:
		# 	# make x of shape (1, context)
		# 	x = x.unsqueeze(0)

		# self.to(self.device)

		# # since x can be too large to load at once, we split it into batches
		# finalOutput = torch.zeros(x.shape[0], self.vocabSize, device=self.device)
		# batchSize = 32
		# x = list(x.split(batchSize, dim=0))
		# for i in range(len(x)):
		# 	xi = x[i].to(self.device)

		# 	with torch.no_grad():
		# 		output = self.linear(self.forward1(xi)[:, -1, :]) # (batchSize, vocabSize)

		# 	finalOutput[i * batchSize : (i + 1) * batchSize, :] = output

		# return torch.nn.Softmax(dim=1)(finalOutput)

		# get the sentence
		sentenceTokens = None
		if x.ndim == 1:
			sentenceTokens = x.unsqueeze(0) # (1, context)
		else:
			sentenceTokens = x[-1, :].unsqueeze(0) # (1, context)

		self.to(self.device)
		sentenceTokens = sentenceTokens.to(self.device)

		with torch.no_grad():
			# get the next word distribution
			output = self.forward1(sentenceTokens) # (1, context, pretrainedEmbeddingSize)
			output = self.linear(output) # (1, context, vocabSize)

		return torch.nn.Softmax(dim=1)(output.view(-1, self.vocabSize))