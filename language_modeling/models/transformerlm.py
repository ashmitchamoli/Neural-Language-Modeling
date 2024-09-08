import torch
from bidict import bidict
from typing import Literal, Union, Callable

from language_modeling import BaseLanguageModel

class TransformerDecoder(BaseLanguageModel):
	def __init__(self, vocabulary : bidict, 
			  	 pretrainedEmbeddings : torch.Tensor,
				 fineTunedPretrained : bool,
				 nhead : int = 8,
				 dimFeedforward : int = 2048,
				 activation : Union[Literal["relu", "gelu"], Callable[[torch.Tensor], torch.Tensor]] = "relu",
				 dropout : float = 0.2) -> None:
		super().__init__(vocabulary, pretrainedEmbeddings, fineTunedPretrained)
		
		self.encoderLayer = torch.nn.TransformerEncoderLayer(
			d_model=self.pretrainedEmbeddingSize,
			nhead=nhead,
			dim_feedforward=dimFeedforward,
			dropout=dropout,
			activation=activation,
			batch_first=True
		)

	def forward(self, x) -> torch.Tensor:
		"""
		x is of shape (batchSize, sequenceLength)
		"""