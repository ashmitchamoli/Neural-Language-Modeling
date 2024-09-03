import torch
from typing import Literal
from alive_progress import alive_bar as aliveBar

from language_modeling.utils import AnnLMDataset
from language_modeling.config import ANN_MODEL_PATH

class AnnLanguageModel(torch.nn.Module):
	def __init__(self, trainDataset : AnnLMDataset, pretrainedEmbeddings : torch.Tensor = None, fineTunePretrained : bool = False, contextSizePrev : int = 5, contextSizeNext : int = 0, embeddingSize : int = 300, activation : str = "tanh") -> None:
		"""
			`pretrainedEmbeddings` are assumed to be indexed with the same vocabulary as `trainDataset`.
			If set to `None`, the pretrained embeddings will be randomly initialized with `torch.nn.Embedding`.
		"""
		super().__init__()

		self.trainDataset = trainDataset
		self.vocabulary = self.trainDataset.vocabulary
		self.vocabSize = len(self.vocabulary)

		self.pretrainedEmbeddings = None
		self.pretrainedEmbeddingSize = None
		if pretrainedEmbeddings is not None:
			self.pretrainedEmbeddings = torch.nn.Embedding.from_pretrained(pretrainedEmbeddings, freeze=(not fineTunePretrained))
			self.pretrainedEmbeddingSize = pretrainedEmbeddings.size()[1]
		else:
			self.pretrainedEmbeddings = torch.nn.Embedding(self.vocabSize, 512)
			self.pretrainedEmbeddingSize = 512

		self.contextSizePrev = contextSizePrev
		self.contextSizeNext = contextSizeNext
		self.embeddingSize = embeddingSize

		self.hidden1 = torch.nn.Linear((self.contextSizePrev + self.contextSizeNext) * self.pretrainedEmbeddingSize, embeddingSize)
		self.hidden2 = torch.nn.Linear(embeddingSize, self.vocabSize)

		self.softmax = torch.nn.Softmax(dim=1)
		self.activation = None
		if activation == "tanh":
			self.activation = torch.nn.Tanh()
		elif activation == "sigmoid":
			self.activation = torch.nn.Sigmoid()
		elif activation == "relu":
			self.activation = torch.nn.ReLU()
		assert self.activation is not None

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def _getEmbeddings_(self, indices : torch.Tensor) -> torch.Tensor:
		if isinstance(self.pretrainedEmbeddings, torch.nn.Embedding):
			return self.pretrainedEmbeddings(indices)
		
		return self.pretrainedEmbeddings[indices]

	def forward(self, x : torch.Tensor):
		"""
		x is of shape (batchSize, contextSizePrev + contextSizeNext)
		each row in x contains indices of the tokens in the context
		
		Returns:
			next word probability distribution (batchSize, vocabSize)
			embedding of this word (batchSize, embeddingSize)
		"""
		x = self._getEmbeddings_(x) # (batchSize, contextSizePrev + contextSizeNext, pretrainedEmbeddingSize)
		x = x.view(-1, (self.contextSizePrev + self.contextSizeNext) * self.pretrainedEmbeddingSize) # (batchSize, (contextSizePrev + contextSizeNext) * pretrainedEmbeddingSize)

		x = self.hidden1(x) # (batchSize, embeddingSize)
		embedding = self.activation(x)
		x = self.hidden2(embedding) # (batchSize, vocabSize)

		return self.softmax(x), embedding
	
	def train(self, valDataset : AnnLMDataset, 
			  epochs : int = 10, 
			  verbose : bool = True, 
			  batchSize : int = 64, 
			  learningRate : float = 0.005, 
			  retrain : bool = False,
			  optimizerType : Literal["adam", "sgd", "rmsprop"] = "adam") -> None:
		self.to(self.device)
		
		trainLoader = torch.utils.data.DataLoader(self.trainDataset, batch_size=batchSize, shuffle=True)
		optimizer = None
		if optimizerType == "adam":
			optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
		else:
			raise ValueError("Unknown optimizer.")
		
		criterion = torch.nn.CrossEntropyLoss()

		for epoch in range(epochs):
			totalLoss = 0
			with aliveBar(len(trainLoader), title=f"Epoch {epoch}") as bar:
				for i, (x, y) in enumerate(trainLoader):
					x = x.to(self.device)
					y = y.to(self.device)

					optimizer.zero_grad()
					output, _ = self(x)
					loss = criterion(output, y)
					loss.backward()
					optimizer.step()

					totalLoss += loss.item()
					bar.text(f"\nAvg Loss: {totalLoss/(i+1):.3f}")

					bar()
			
			avgLoss = totalLoss / len(trainLoader)
			if verbose:	
				print(f"Epoch {epoch} completed. log(Perplexity): {avgLoss:.3f}")
		
		self.saveModel(ANN_MODEL_PATH)
		if verbose:
			print("Model saved.")

		return

	def saveModel(self, path : str) -> None:
		torch.save(self.state_dict(), path)