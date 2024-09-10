from language_modeling.models import AnnLanguageModel
from language_modeling.utils import Inferencer
from preprocessing_scripts.pretrained_embeddings import loadPretrained
from annlm_training import modelHyperparams, vocab, testTokens, trainTokens

if __name__ == "__main__":
	pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
	model = AnnLanguageModel(vocabulary=vocab,
							 pretrainedEmbeddings=pretrainedW2v,
							 **modelHyperparams)
	model.loadModelWeights()

	inferencer = Inferencer(model)

	testPerplexity = inferencer.computePerplexity(testTokens)
	print(f"Test perplexity: {testPerplexity}")

	trainPerplexity = inferencer.computePerplexity(trainTokens)
	print(f"Train perplexity: {trainPerplexity}")
