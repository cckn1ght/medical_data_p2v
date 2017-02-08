from gensim.models import Phrases, Word2Vec
from gensim.models.word2vec import LineSentence


def build_model(min_count, threshold):
    sentences = LineSentence('./cleaned_data/w2v_models/22m_s3.txt')
    bigram = Phrases(
        sentences, min_count=min_count, threshold=threshold)
    trigram = Phrases(
        bigram[sentences], min_count=min_count, threshold=threshold)
    bigram.save('./cleaned_data/bigramPhraseModel')
    trigram.save('./cleaned_data/trigramPhraseModel')
    model = Word2Vec(trigram[bigram[sentences]],
                     size=200, window=5, min_count=10, workers=4, max_vocab_size=200000)
    model.save('./cleaned_data/p2vModel')

build_model(5, 10)
