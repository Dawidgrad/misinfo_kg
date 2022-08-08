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

        processed_docs = self.preprocess(docs)
        self.dictionary = gensim.corpora.Dictionary(processed_docs)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

    def lemmatize_stemming(self, text):
        return self.stemmer.stem(self.lemmatizer.lemmatize(text, pos='v'))

    def preprocess(self, text):
        result = []
        for item in text:
            tokenised = simple_preprocess(item)
            result.append(list(filter(lambda x: x not in self.stopwords, tokenised)))
        return result

    def get_topics(self, num_topics=10, passes=2, workers=2):
        lda_model = gensim.models.LdaMulticore(self.bow_corpus, num_topics=num_topics, id2word=self.dictionary, passes=passes, workers=workers)
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
