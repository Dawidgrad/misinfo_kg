import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk

class LDA:
    def __init__(self, docs):
        nltk.download('wordnet')
        self.stopwords = set(STOPWORDS)
        self.stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')

        lda_words, lda_triples = self.preprocess(docs)
        self.dictionary = gensim.corpora.Dictionary(lda_triples)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in lda_triples]

    def lemmatize_stemming(self, text):
        return self.stemmer.stem(self.lemmatizer.lemmatize(text, pos='v'))

    def preprocess(self, docs):
        lda_words = []
        lda_triples = []
        for item in docs:
            tokenised = simple_preprocess(item)
            lda_words.append(list(filter(lambda x: x not in self.stopwords, tokenised)))

            lda_triples.append(item)

        return lda_words, lda_triples

    def get_topics(self, num_topics=10, passes=2, workers=2):
        lda_model = gensim.models.LdaMulticore(self.bow_corpus, num_topics=num_topics, id2word=self.dictionary, passes=passes, workers=workers)
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
