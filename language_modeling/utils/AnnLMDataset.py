import torch
from bidict import bidict
from itertools import chain

from language_modeling.config import PAD_TOKEN, EOS_TOKEN, UNK_TOKEN

class AnnLMDataset(torch.utils.data.Dataset):
	def __init__(self, tokens : list[list[str]], vocabulary : bidict, contextSizePrev : int = 5, contextSizeNext : int = 0) -> None:
		super.__init__()

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

		