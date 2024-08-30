import torch
from bidict import bidict
from alive_progress import alive_bar as aliveBar

from language_modeling.utils import annlm_dataset

class AnnLanguageModel(torch.nn.Module):
	def __init__(self, vocab : bidict, pretrainedEmbeddings : torch.Tensor, contextSizePrev : int = 5, contextSizeNext : int = 0, embeddingSize : int = 300, activation : str = "tanh") -> None:
		super.__init__()

		self.vocab = vocab
		self.vocabSize = len(vocab)
		self.pretrainedEmbeddings = pretrainedEmbeddings
		self.pretrainedEmbeddingSize = pretrainedEmbeddings.shape[1]
		self.contextSizePrev = contextSizePrev
		self.contextSizeNext = contextSizeNext
		self.embeddingSize = embeddingSize

		self.hidden1 = torch.nn.Linear((self.contextSizePrev + self.contextSizeNext) * self.pretrainedEmbeddings.shape[1], embeddingSize)
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

	def forward(self, x : torch.Tensor):
		"""
		x is of shape (batchSize, contextSizePrev + contextSizeNext)
		each row in x contains indices of the tokens in the context
		
		Returns:
			next word probability distribution (batchSize, vocabSize)
			embedding of this word (batchSize, embeddingSize)
		"""
		x = self.pretrainedEmbeddings[x] # (batchSize, contextSizePrev + contextSizeNext, pretrainedEmbeddingSize)
		x = x.view(-1, (self.contextSizePrev + self.contextSizeNext) * self.pretrainedEmbeddingSize) # (batchSize, (contextSizePrev + contextSizeNext) * pretrainedEmbeddingSize)

		x = self.hidden1(x) # (batchSize, embeddingSize)
		embedding = self.activation(embedding)
		x = self.hidden2(embedding) # (batchSize, vocabSize)

		return self.softmax(x), embedding
	
	def train(self, trainDataset : annlm_dataset, 
		   	  valDataset : annlm_dataset, 
			  epochs : int = 10, 
			  verbose : bool = True, 
			  batchSize : int = 32, 
			  learningRate : float = 0.001, 
			  retrain : bool = False) -> None:
		self.to(self.device)
		
		trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
		optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
		criterion = torch.nn.CrossEntropyLoss()

		for epoch in range(epochs):
			with aliveBar(len(trainLoader), title=f"Epoch {epoch}") as bar:
				for i, (x, y) in enumerate(trainLoader):
					x = x.to(self.device)
					y = y.to(self.device)

					optimizer.zero_grad()
					output, _ = self(x)
					loss = criterion(output, y)
					loss.backward()
					optimizer.step()

					bar()
				
			print(f"Epoch {epoch} completed.")
		return