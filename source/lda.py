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
    def get_topics_bow(self, num_topics, alpha, passes=2, workers=2):
        lda_model = LdaMulticore(self.corpus_bow, num_topics=num_topics, id2word=self.dictionary_bow, passes=passes, workers=workers, alpha=alpha)
        flat_lda_words = [word for sublist in self.lda_words for word in sublist]
        coherence = self.get_coherence_score(lda_model, self.dictionary_bow, flat_lda_words, processes=workers)
        self.print_topics(lda_model, f'BOW - Topics = {num_topics}, Alpha = {alpha}', coherence)

    # Perform LDA using the Triple-based model
    def get_topics_triple(self, num_topics, alpha, passes=2, workers=2):
        lda_model = LdaMulticore(self.corpus_triple, num_topics=num_topics, id2word=self.dictionary_triple, passes=passes, workers=workers, alpha=alpha)
        coherence = self.get_coherence_score(lda_model, self.dictionary_triple, self.triples, processes=workers)
        self.print_topics(lda_model, f'Triples - Topics = {num_topics}, Alpha = {alpha}', coherence)

    # Perform LDA using SVO-based model
    def get_topics_svo(self, num_topics, alpha, passes=2, workers=2):
        lda_model = LdaMulticore(self.corpus_subject, num_topics=num_topics, id2word=self.dictionary_subject, passes=passes, workers=workers, alpha=alpha)
        coherence = self.get_coherence_score(lda_model, self.dictionary_subject, self.subjects)
        self.print_topics(lda_model, f'Subject - Topics = {num_topics}, Alpha = {alpha}', coherence)

        lda_model = LdaMulticore(self.corpus_verb, num_topics=num_topics, id2word=self.dictionary_verb, passes=passes, workers=workers, alpha=alpha)
        coherence = self.get_coherence_score(lda_model, self.dictionary_verb, self.verbs)
        self.print_topics(lda_model, f'Verb - Topics = {num_topics}, Alpha = {alpha}', coherence)

        lda_model = LdaMulticore(self.corpus_object, num_topics=num_topics, id2word=self.dictionary_object, passes=passes, workers=workers, alpha=alpha)
        coherence = self.get_coherence_score(lda_model, self.dictionary_object, self.objects, processes=workers)
        self.print_topics(lda_model, f'Object - Topics = {num_topics}, Alpha = {alpha}', coherence)

    # Calculate and print coherence (possible coherence metrics: c_v, u_mass, c_uci)
    def get_coherence_score(self, lda_model, dictionary, data, coherence_metric='c_v', processes=2):
        coherence_model_lda = CoherenceModel(model=lda_model, texts=[data], dictionary=dictionary, coherence=coherence_metric, processes=processes) 
        coherence_lda = coherence_model_lda.get_coherence()
        
        return coherence_lda

    def print_topics(self, lda_model, name, coherence_score):
        # Write topics to a file in source/output_files folder
        with open('source/output_files/lda_analysis.txt', 'a', encoding='utf-8') as f:
            f.write(name + '\n')
            for idx, topic in lda_model.print_topics(num_words=10):
                f.write('Topic {}: {}'.format(idx + 1, topic))
                f.write('\n')
            f.write('\n')
            f.write(f'Coherence Score: {coherence_score}')
            f.write('\n\n\n')
        print('Topics written to source/output_files/lda_analysis.txt')
        print()
