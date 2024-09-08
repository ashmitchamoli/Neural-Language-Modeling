import torch
import pickle as pkl

from language_modeling.models import AnnLanguageModel
from language_modeling.utils import Inferencer
from pretrained_embeddings import loadPretrained
from annlm_training import modelHyperparams, vocab, testTokens

if __name__ == "__main__":
	pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
	model = AnnLanguageModel(vocabulary=vocab,
							 pretrainedEmbeddings=pretrainedW2v,
							 **modelHyperparams)
	model.loadModelWeights()

	inferencer = Inferencer(model)
	# sentence = inferencer.generateSentence(["The", "bride", "blushed", ",", "while"])
	# print(sentence)

	print(inferencer.computePerplexity(testTokens))

