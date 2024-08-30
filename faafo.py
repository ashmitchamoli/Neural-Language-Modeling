from language_modeling.models import AnnLanguageModel
from language_modeling.utils import Tokenizer, AnnLMDataset
from sklearn.model_selection import train_test_split

tokenizer = Tokenizer("data/Auguste_Maqueture.txt")
tokens = tokenizer.getTokens()

