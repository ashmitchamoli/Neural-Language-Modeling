from language_modeling.models import TransformerLanguageModel
from language_modeling.utils import Inferencer
from preprocessing_scripts.pretrained_embeddings import loadPretrained
from transformerlm_training import modelHyperparams, vocab, testTokens, trainTokens

if __name__ == "__main__":
	pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
	model = TransformerLanguageModel(vocabulary=vocab,
							  pretrainedEmbeddings=pretrainedW2v,
							  **modelHyperparams)
	model.loadModelWeights()

	inferencer = Inferencer(model)
	testPerplexity = inferencer.computePerplexity(testTokens, True, "perplexity_scores/2021101114-transformerLM1-test-perplexity.txt")
	print(f"Test perplexity: {testPerplexity}")

	trainPerplexity = inferencer.computePerplexity(trainTokens, True, "perplexity_scores/2021101114-transformerLM1-train-perplexity.txt")
	print(f"Train perplexity: {trainPerplexity}")
