from annlm_training import trainingConfig as nnlmTrainingConfig

from language_modeling.models import AnnLanguageModel


nnlm = AnnLanguageModel.loadModel("nnlm_tanh_0_5_512_300")
