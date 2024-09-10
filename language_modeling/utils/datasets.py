import torch
from bidict import bidict
from itertools import chain

from language_modeling.config import PAD_TOKEN, EOS_TOKEN, UNK_TOKEN

class AnnLanguageModelDataset(torch.utils.data.Dataset):
	def __init__(self, tokens : list[list[str]], vocabulary : bidict, contextSizePrev : int = 5, contextSizeNext : int = 0) -> None:
		super().__init__()

		self.tokens = list(chain(*tokens))
		self.vocabulary = vocabulary
		self.contextSizePrev = contextSizePrev
		self.contextSizeNext = contextSizeNext
		self.vocabSize = len(vocabulary)
	
	def __len__(self) -> int:
		return len(self.tokens)

	def _getPrevContext_(self, index : int) -> list[int]:
		prevContext = [self.vocabulary[PAD_TOKEN]] * self.contextSizePrev

		for i in range(1, self.contextSizePrev+1):
			if index - i < 0:
				break
			if self.tokens[index - i] == EOS_TOKEN:
				break

			if self.tokens[index - i] not in self.vocabulary:
				prevContext[self.contextSizePrev - i] = self.vocabulary[UNK_TOKEN]
			else:
				prevContext[self.contextSizePrev - i] = self.vocabulary[self.tokens[index - i]]

		return prevContext

	def _getNextContext_(self, index : int) -> list[int]:
		nextContext = [self.vocabulary[PAD_TOKEN]] * self.contextSizeNext

		for i in range(1, self.contextSizeNext+1):
			if index + i >= len(self.tokens):
				break
			if self.tokens[index + i] == EOS_TOKEN:
				nextContext[i - 1] = self.vocabulary[EOS_TOKEN]
				break

			if self.tokens[index + i] not in self.vocabulary:
				nextContext[i - 1] = self.vocabulary[UNK_TOKEN]
			else:
				nextContext[i - 1] = self.vocabulary[self.tokens[index + i]]
			
		return nextContext

	def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
		prevContext = self._getPrevContext_(index)
		nextContext = self._getNextContext_(index)

		return torch.tensor(prevContext + nextContext), torch.tensor(self.vocabulary[self.tokens[index]])

class LstmLanguageModelDataset(torch.utils.data.Dataset):
	def __init__(self, tokens : list[list[str]], vocabulary : bidict) -> None:
		super().__init__()

		self.tokens = tokens
		self.vocabulary = vocabulary
		self.vocabSize = len(vocabulary)
	
	def __len__(self) -> int:
		return len(self.tokens)
	
	def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
		sentence = self.tokens[index]

		encodedSentence = [self.vocabulary[word] for word in sentence]
		
		return torch.tensor(encodedSentence[:-1]), torch.tensor(encodedSentence[1:])

	def customCollate(self, batch : list[torch.Tensor]) -> torch.Tensor:
		X, y = zip(*batch)

		X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
		y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])

		return X, y.view(-1)

class TransformerLanguageModelDataset(torch.utils.data.Dataset):
	def __init__(self, tokens : list[list[str]], vocabulary : bidict) -> None:
		super().__init__()

		self.tokens = tokens
		self.vocabulary = vocabulary
		self.vocabSize = len(vocabulary)

		self.dataset = self._prepareDataset_(tokens, vocabulary)
	
	@staticmethod
	def _prepareDataset_(tokens : list[list[str]], vocabulary : bidict) -> torch.utils.data.Dataset:
		dataset = []
		for sentence in tokens:
			context = [vocabulary[PAD_TOKEN]]
			for token in sentence:
				dataset.append((torch.tensor(context + [vocabulary[PAD_TOKEN]]), torch.tensor(vocabulary[token])))
				context = context + [vocabulary[token]]

		return dataset
	
	def __len__(self) -> int:
		return len(self.dataset)
	
	def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
		return self.dataset[index]
	
	def customCollate(self, batch : list[torch.Tensor]) -> torch.Tensor:
		X, y = zip(*batch)

		X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
		
		return X, torch.stack(y)