from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk

class LDA:
    def __init__(self, triples, subjects, objects, verbs):
        nltk.download('wordnet')
        self.stopwords = set(STOPWORDS)
        self.stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')

        # Prepare dictionaries and corpuses for all LDA models
        self.triples = triples
        self.subjects = subjects
        self.objects = objects
        self.verbs = verbs
        self.lda_words = self.preprocess(triples)

        self.dictionary_triple = Dictionary([triples])
        self.corpus_triple = [self.dictionary_triple.doc2bow([text]) for text in triples]

        self.dictionary_bow = Dictionary(self.lda_words)
        self.corpus_bow = [self.dictionary_bow.doc2bow(text) for text in self.lda_words]

        self.dictionary_subject = Dictionary([subjects])
        self.dictionary_verb = Dictionary([verbs])
        self.dictionary_object = Dictionary([objects])
        self.corpus_subject = [self.dictionary_subject.doc2bow([text]) for text in subjects]
        self.corpus_verb = [self.dictionary_verb.doc2bow([text]) for text in verbs]
        self.corpus_object = [self.dictionary_object.doc2bow([text]) for text in objects]

    # Preprocess tokenised words
    def preprocess(self, triples):
        lda_words = list()
        for item in triples:
            tokenised = simple_preprocess(item)
            lda_words.append(list(filter(lambda x: x not in self.stopwords, tokenised)))

        return lda_words

    # Perform LDA using the BOW-based model
    def get_topics_bow(self, num_topics=10, passes=2, workers=2):
        lda_model = LdaMulticore(self.corpus_bow, num_topics=num_topics, id2word=self.dictionary_bow, passes=passes, workers=workers)
        print('\n\nBOW\n')
        self.print_topics(lda_model)
        flat_lda_words = [word for sublist in self.lda_words for word in sublist]
        self.get_coherence_score(lda_model, self.dictionary_bow, flat_lda_words)

    # Perform LDA using the Triple-based model
    def get_topics_triple(self, num_topics=10, passes=2, workers=2):
        lda_model = LdaMulticore(self.corpus_triple, num_topics=num_topics, id2word=self.dictionary_triple, passes=passes, workers=workers)
        print('\n\nTriples\n')
        self.print_topics(lda_model)
        self.get_coherence_score(lda_model, self.dictionary_triple, self.triples)

    # Perform LDA using SVO-based model
    def get_topics_svo(self, num_topics=10, passes=2, workers=2):
        lda_model = LdaMulticore(self.corpus_subject, num_topics=num_topics, id2word=self.dictionary_subject, passes=passes, workers=workers)
        print('\n\nSubjects\n')
        self.print_topics(lda_model)
        self.get_coherence_score(lda_model, self.dictionary_subject, self.subjects)

        lda_model = LdaMulticore(self.corpus_verb, num_topics=num_topics, id2word=self.dictionary_verb, passes=passes, workers=workers)
        print('\n\nVerbs\n')
        self.print_topics(lda_model)
        self.get_coherence_score(lda_model, self.dictionary_verb, self.verbs)

        lda_model = LdaMulticore(self.corpus_object, num_topics=num_topics, id2word=self.dictionary_object, passes=passes, workers=workers)
        print('\n\nObjects\n')
        self.print_topics(lda_model)
        self.get_coherence_score(lda_model, self.dictionary_object, self.objects)

    # Calculate and print coherence (possible coherence metrics: c_v, u_mass, c_uci)
    def get_coherence_score(self, lda_model, dictionary, data, coherence_metric='c_v'):
        coherence_model_lda = CoherenceModel(model=lda_model, texts=[data], dictionary=dictionary, coherence=coherence_metric)
        
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score:', coherence_lda)
        print()

    def print_topics(self, lda_model):
        for idx, topic in lda_model.print_topics(-1):
            print()
            print('Topic: {} \nWords: {}'.format(idx + 1, topic))
        print()

