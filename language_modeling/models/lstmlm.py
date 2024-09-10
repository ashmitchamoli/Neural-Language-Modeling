import torch
from bidict import bidict

from language_modeling import BaseLanguageModel
from language_modeling.config import LSTM_MODEL_PATH

class LstmLanguageModel(BaseLanguageModel):
	def __init__(self, vocabulary : bidict, 
			  	 pretrainedEmbeddings : torch.Tensor = None, 
				 fineTunePretrained : bool = False,
				 hiddenEmbeddingSize : int = 512,
				 numLayers : int = 2,
				 dropout : float = 0.2,
				 bidirectional : bool = False,
				 activation : str = "tanh",
				 linearClassifierLayers : list[int] | None = None) -> None:
		super().__init__(vocabulary, pretrainedEmbeddings, fineTunePretrained)

		self.hiddenEmbeddingSize = hiddenEmbeddingSize
		self.numLayers = numLayers
		self.dropout = dropout
		self.dropoutLayer = torch.nn.Dropout(self.dropout)
		self.bidirectional = bidirectional
		
		self.activation = None
		if activation == "tanh":
			self.activation = torch.nn.Tanh()
		elif activation == "sigmoid":
			self.activation = torch.nn.Sigmoid()
		elif activation == "relu":
			self.activation = torch.nn.ReLU()
		assert self.activation is not None

		self.lstm = torch.nn.LSTM(self.pretrainedEmbeddingSize, 
								  hidden_size=self.hiddenEmbeddingSize,
								  num_layers=self.numLayers,
								  batch_first=True,
								  dropout=self.dropout,
								  bidirectional=self.bidirectional)
		
		self.linear = torch.nn.Sequential()
		if linearClassifierLayers is None or len(linearClassifierLayers) == 0:
			self.linear.append(torch.nn.Linear(self.hiddenEmbeddingSize * (self.bidirectional + 1), self.vocabSize))
		else:
			self.linear = torch.nn.Sequential(torch.nn.Linear(self.hiddenEmbeddingSize * (self.bidirectional + 1), linearClassifierLayers[0]),)
			for i in range(1, len(linearClassifierLayers)):
				self.linear.append(self.activation)
				self.linear.append(torch.nn.Linear(linearClassifierLayers[i - 1], linearClassifierLayers[i]))
				if i%3 == 0:
					self.linear.append(self.dropoutLayer)
			self.linear.append(self.activation)
			self.linear.append(torch.nn.Linear(linearClassifierLayers[-1], self.vocabSize))
		
		self.softmax = torch.nn.Softmax(dim=2)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._modelSaveDir_ = LSTM_MODEL_PATH
		self._modelName_ = f"lstmlm_{self.activation}_{self.hiddenEmbeddingSize}_{self.numLayers}_{self.dropout}_{self.bidirectional}"
	
	def forward1(self, x : torch.Tensor) -> torch.Tensor:
		"""
		:param x: batch of sentences (batchSize, maxSentenceLength) 
		"""
		x = self._getPretrainedEmbeddings_(x) # (batchSize, maxSentenceLength, pretrainedEmbeddingSize)
		embeddings, _ = self.lstm(x) # (batchSize, maxSentenceLength, hiddenEmbeddingSize * (bidirectional + 1))
		x = self.linear(embeddings) # (batchSize, maxSentenceLength, vocabSize)
		# x = self.softmax(x)

		return x, embeddings
	
	def forward(self, x : torch.Tensor) -> torch.Tensor:
		"""
		:param x: batch of sentences (batchSize, maxSentenceLength) 
		"""
		x, _ = self.forward1(x) # (batchSize, maxSentenceLength, vocabSize)

		return x.view(-1, self.vocabSize), None

	def getNextWordDistribution(self, x: torch.Tensor) -> torch.Tensor:
		"""
		:param x: size (batchSize, context) or (context, ). `x` contains indices of context words.
		"""
		if x.ndim == 1:
			# make x of shape (1, context)
			x = x.unsqueeze(0)

		self.to(self.device)

		# since x can be too large to load at once, we split it into batches
		finalOutput = torch.zeros(x.shape[0], self.vocabSize, device=self.device)
		batchSize = 32
		x = list(x.split(batchSize, dim=0))
		for i in range(len(x)):
			xi = x[i].to(self.device)

			with torch.no_grad():
				output = self.forward1(xi)[0][:, -1, :] # (batchSize, vocabSize)

			finalOutput[i * batchSize : (i + 1) * batchSize, :] = output

		return torch.nn.Softmax(dim=1)(finalOutput)
