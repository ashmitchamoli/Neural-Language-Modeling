import torch
import os
from bidict import bidict
from typing import Literal
from alive_progress import alive_bar as aliveBar

from language_modeling import BaseLanguageModel
from language_modeling.utils import LstmLanguageModelDataset
from language_modeling.config import LSTM_MODEL_PATH

class LstmLanguageModel(BaseLanguageModel):
	def __init__(self, vocabulary : bidict, 
			  	 pretrainedEmbeddings : torch.Tensor = None, 
				 fineTunedPretrained : bool = False,
				 hiddenEmbeddingSize : int = 512,
				 numLayers : int = 2,
				 dropout : float = 0.0,
				 bidirectional : bool = False,
				 activation : str = "tanh",
				 linearClassifierLayers : list[int] | None = None) -> None:
		super().__init__(vocabulary, pretrainedEmbeddings, fineTunedPretrained)

		self.hiddenEmbeddingSize = hiddenEmbeddingSize
		self.numLayers = numLayers
		self.dropout = dropout
		self.bidirectional = bidirectional

		self.lstm = torch.nn.LSTM(self.pretrainedEmbeddingSize, 
								  hidden_size=self.hiddenEmbeddingSize,
								  num_layers=self.numLayers,
								  batch_first=True,
								  dropout=self.dropout,
								  bidirectional=self.bidirectional)
		if linearClassifierLayers is None or len(linearClassifierLayers) == 0:
			self.linear = torch.nn.Linear(self.hiddenEmbeddingSize * (self.bidirectional + 1), self.vocabSize)
		else:
			self.linear = torch.nn.Sequential(torch.nn.Linear(self.hiddenEmbeddingSize * (self.bidirectional + 1), linearClassifierLayers[0]),)
			for i in range(1, len(linearClassifierLayers)):
				self.linear.append(torch.nn.Linear(linearClassifierLayers[i - 1], linearClassifierLayers[i]))
			self.linear.append(torch.nn.Linear(linearClassifierLayers[-1], self.vocabSize))
		
		self.activation = None
		if activation == "tanh":
			self.activation = torch.nn.Tanh()
		elif activation == "sigmoid":
			self.activation = torch.nn.Sigmoid()
		elif activation == "relu":
			self.activation = torch.nn.ReLU()
		assert self.activation is not None
		self.softmax = torch.nn.Softmax(dim=2)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self._modelSavePath_ = LSTM_MODEL_PATH
		self._modelName_ = f"lstmlm_{self.activation}_{self.hiddenEmbeddingSize}_{self.numLayers}_{self.dropout}_{self.bidirectional}"
	
	def forward1(self, x : torch.Tensor) -> torch.Tensor:
		"""
		:param x: batch of sentences (batchSize, maxSentenceLength) 
		"""
		x = self._getPretrainedEmbeddings_(x) # (batchSize, maxSentenceLength, pretrainedEmbeddingSize)
		embeddings, _ = self.lstm(x) # (batchSize, maxSentenceLength, hiddenEmbeddingSize * (bidirectional + 1))
		x = self.linear(embeddings) # (batchSize, maxSentenceLength, vocabSize)
		x = self.softmax(x)

		return x, embeddings
	
	def forward(self, x : torch.Tensor) -> torch.Tensor:
		"""
		:param x: batch of sentences (batchSize, maxSentenceLength) 
		"""
		x, _ = self.forward1(x[:, :-1]) # (batchSize, maxSentenceLength, vocabSize)

		return x.view(-1, self.vocabSize), None

	# def train(self, valDataset : LstmLanguageModelDataset, 
	# 		  epochs : int = 5, 
	# 		  verbose : bool = True, 
	# 		  batchSize : int = 64, 
	# 		  learningRate : float = 0.005, 
	# 		  retrain : bool = False,
	# 		  optimizerType : Literal["adam", "sgd", "rmsprop"] = "adam") -> None:
	# 	self.to(self.device)

	# 	if not retrain:
	# 		if self._loadModel_(os.path.join(self._modelSavePath_, self._modelName_)):
	# 			if verbose:
	# 				print(f"Loaded model from {os.path.join(self._modelSavePath_, self._modelName_)}")
	# 			return
	# 		else:
	# 			if verbose:
	# 				print(f"Model checkpoint not found. Training model from scratch...")
		
	# 	trainLoader = torch.utils.data.DataLoader(self.trainDataset, batch_size=batchSize, shuffle=True, collate_fn=self.trainDataset._customCollate_)
	# 	optimizer = None
	# 	if optimizerType == "adam":
	# 		optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
	# 	else:
	# 		raise ValueError("Unknown optimizer.")
		
	# 	criterion = torch.nn.CrossEntropyLoss()

	# 	for epoch in range(epochs):
	# 		totalLoss = 0
	# 		with aliveBar(len(trainLoader), title=f"Epoch {epoch}") as bar:
	# 			for i, (x, y) in enumerate(trainLoader):
	# 				x = x.to(self.device)
	# 				y = y.to(self.device)

	# 				optimizer.zero_grad()
	# 				output = self(x[:, :-1])
	# 				output = output.view(-1, self.vocabSize)
	# 				y = y.view(-1)
	# 				loss = criterion(output, y)
	# 				loss.backward()
	# 				optimizer.step()

	# 				totalLoss += loss.item()
	# 				bar.text(f"\nAvg Loss: {totalLoss/(i+1):.3f}")

	# 				bar()
			
	# 		avgLoss = totalLoss / len(trainLoader)
	# 		if verbose:	
	# 			print(f"Epoch {epoch} completed. log(Perplexity): {avgLoss:.3f}")
		
	# 	self._saveModel_(os.path.join(self._modelSavePath_, self._modelName_ + ".pth"))
	# 	if verbose:
	# 		print("Model saved.")

	# 	return