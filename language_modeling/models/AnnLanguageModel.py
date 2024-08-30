import torch
from bidict import bidict

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