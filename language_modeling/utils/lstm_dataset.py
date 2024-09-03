import torch
from bidict import bidict

from language_modeling.config import PAD_TOKEN, EOS_TOKEN, UNK_TOKEN

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
		
		return torch.tensor(encodedSentence), torch.tensor(encodedSentence[1:])

	def _customCollate_(self, batch : list[torch.Tensor]) -> torch.Tensor:
		X, y = zip(*batch)

		X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
		y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])

		return X, y