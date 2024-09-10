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

		self._modelSaveDir_ = None
		self._modelName_ = None

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
			self.load_state_dict(torch.load(path, weights_only=True))
			return True
		else:
			return False

	def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
		pass

	# pylint: disable=arguments-differ
	def train(self, trainLoader : torch.utils.data.DataLoader, 
			  valLoader : torch.utils.data.DataLoader = None,
			  epochs : int = 5, 
			  verbose : bool = True, 
			  batchSize : int = 64, 
			  learningRate : float = 0.005, 
			  retrain : bool = False,
			  optimizerType : Literal["adam", "sgd", "rmsprop"] = "adam",
			  ignorePadding : int = None) -> None:
		self.to(self.device)

		if not retrain:
			if self._loadModel_(os.path.join(self._modelSaveDir_, self._modelName_ + '.pth')):
				if verbose:
					print(f"Loaded model from {os.path.join(self._modelSaveDir_, self._modelName_ + '.pth')}")
				return
			else:
				if verbose:
					print("Model checkpoint not found. Training model from scratch...")

		optimizer = None
		if optimizerType == "adam":
			optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
		else:
			raise ValueError("Unknown optimizer.")
		
		if ignorePadding is not None:
			criterion = torch.nn.CrossEntropyLoss(ignore_index=ignorePadding)
		else:
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

			if valLoader is not None:
				valLoss = 0
				valAcc = 0
				with torch.no_grad():
					with aliveBar(len(valLoader), title="Validation") as bar:
						for x, y in valLoader:
							x = x.to(self.device)
							y = y.to(self.device)

							output, _ = self(x)
							loss = criterion(output, y)

							valLoss += loss.item()
							preds = torch.argmax(output, dim=1)
							if ignorePadding is not None:
								preds = preds[y != ignorePadding]
								y = y[y != ignorePadding]
							valAcc += torch.sum(preds == y).item() / len(y)

							bar()

				valLoss /= len(valLoader)
				valAcc /= len(valLoader)

			if verbose and valLoader is not None:
				print(f"Epoch {epoch} completed. Training loss: {avgLoss:.3f} | Validation loss: {valLoss:.3f} | Validation acc: {valAcc:.3f}")
			elif verbose:
				print(f"Epoch {epoch} completed. Training loss: {avgLoss:.3f}")

		self._saveModel_(os.path.join(self._modelSaveDir_, self._modelName_ + ".pth"))
		if verbose:
			print("Model saved.")

		return

	def loadModelWeights(self) -> None:
		if self._loadModel_(os.path.join(self._modelSaveDir_, self._modelName_ + '.pth')):
			print(f"Loaded model from {os.path.join(self._modelSaveDir_, self._modelName_ + '.pth')}")
		else:
			print("Model checkpoint not found. Train the model from scratch.")

	def getNextWordDistribution(self, x : torch.Tensor) -> torch.Tensor:
		"""
		:param x: (batchSize, context) or (context, ). `x` contains indices of context words.
		
		Returns
			(batchSize, vocabSize) or (vocabSize, )
			The probability distribution for the next word.
		"""
	