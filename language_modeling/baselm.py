import torch
import os
from bidict import bidict
from typing import Literal
from alive_progress import alive_bar as aliveBar

class BaseLanguageModel(torch.nn.Module):
	def __init__(self, vocabulary : bidict,
			  	 pretrainedEmbeddings : torch.Tensor = None, 
				 fineTunePretrained : bool = False) -> None:
		super().__init__()

		self.vocabulary = vocabulary
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
	
	def _getPretrainedEmbeddings_(self, indices : torch.Tensor) -> torch.Tensor:
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
		
	def train(self, trainLoader : torch.utils.data.DataLoader, 
			  epochs : int = 5, 
			  verbose : bool = True, 
			  batchSize : int = 64, 
			  learningRate : float = 0.005, 
			  retrain : bool = False,
			  optimizerType : Literal["adam", "sgd", "rmsprop"] = "adam") -> None:
		self.to(self.device)

		if not retrain:
			if self._loadModel_(os.path.join(self._modelSavePath_, self._modelName_)):
				if verbose:
					print(f"Loaded model from {os.path.join(self._modelSavePath_, self._modelName_)}")
				return
			else:
				if verbose:
					print(f"Model checkpoint not found. Training model from scratch...")
		
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
		
		self._saveModel_(os.path.join(self._modelSavePath_, self._modelName_ + ".pth"))
		if verbose:
			print("Model saved.")

		return
	
	def getNextWordDistribution(self, x : torch.Tensor) -> torch.Tensor:
		"""
		:param x: (batchSize, context) or (context, ). `x` contains indices of context words.
		
		Returns
			(batchSize, vocabSize) or (vocabSize, )
			The probability distribution for the next word.
		"""
		pass
	