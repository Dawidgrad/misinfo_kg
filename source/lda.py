from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel, LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class LDA:
    def __init__(self, triples, subjects, objects, verbs):
        nltk.download('wordnet')
        self.stopwords = set(STOPWORDS)
        self.stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

        # Prepare dictionaries and corpuses for all LDA models
        self.bow_data, self.triples, self.subjects, self.verbs, self.objects = self.preprocess_data(triples, subjects, verbs, objects)

        # Plot a word cloud based on self.triples
        flattened_triples = [item for sublist in self.triples for item in sublist]
        joined_triples = [triple.replace(' ', 'v') for triple in flattened_triples]
        print(joined_triples)
        WordCloud(width=1200, height=600).generate(' '.join(joined_triples)).to_file('wordcloud.png')

        # Create dictionary and corpus for all models
        self.dictionary_triple = Dictionary(self.triples)
        self.corpus_triple = [self.dictionary_triple.doc2bow(doc) for doc in self.triples]

        self.dictionary_bow = Dictionary(self.bow_data)
        self.corpus_bow = [self.dictionary_bow.doc2bow(text) for text in self.bow_data]

        self.dictionary_subject = Dictionary(self.subjects)
        self.dictionary_verb = Dictionary(self.verbs)
        self.dictionary_object = Dictionary(self.objects)
        self.corpus_subject = [self.dictionary_subject.doc2bow(text) for text in self.subjects]
        self.corpus_verb = [self.dictionary_verb.doc2bow(text) for text in self.verbs]
        self.corpus_object = [self.dictionary_object.doc2bow(text) for text in self.objects]

    def preprocess_data(self, docs_triples, docs_subjects, docs_verbs, docs_objects):
        bow_data = list()
        triples_data = list()
        subjects_data = list()
        verbs_data = list()
        objects_data = list()

        # Preprocess text data for every type of model
        for doc_tuple in zip(docs_triples, docs_subjects, docs_verbs, docs_objects):
            tokenised_bow = list()
            tokenised_triples = list()
            tokenised_subject = list()
            tokenised_verb = list()
            tokenised_object = list()

            for triple in doc_tuple[0]:
                triple = triple.replace(' is ', ' ')
                tokenised = simple_preprocess(triple)
                if len(tokenised) < 3:
                    continue
                tokenised_bow += tokenised
                tokenised_triples.append(' '.join(tokenised))

            for subject in doc_tuple[1]:
                tokenised_subject += simple_preprocess(subject)

            for verb in doc_tuple[2]:
                tokenised_verb += simple_preprocess(verb)

            for obj in doc_tuple[3]:
                tokenised_object += simple_preprocess(obj)

            bow_data.append(list(filter(lambda x: x not in self.stopwords, tokenised_bow)))
            triples_data.append(tokenised_triples)
            subjects_data.append(tokenised_subject)
            verbs_data.append(tokenised_verb)
            objects_data.append(tokenised_object)

            # Remove empty lists from the data
            bow_data = list(filter(None, bow_data))
            triples_data = list(filter(None, triples_data))
            subjects_data = list(filter(None, subjects_data))
            verbs_data = list(filter(None, verbs_data))
            objects_data = list(filter(None, objects_data))

        return bow_data, triples_data, subjects_data, verbs_data, objects_data

    # Perform LDA using the BOW-based model
    def get_topics_bow(self, num_topics, alpha='auto', passes=1, workers=2):
        lda_model = LdaModel(self.corpus_bow, num_topics=num_topics, id2word=self.dictionary_bow, passes=passes, alpha=alpha)
        coherence = self.get_coherence_score(lda_model, self.dictionary_bow, self.bow_data, processes=workers)
        self.print_topics(lda_model, f'BOW - Topics = {num_topics}, Alpha = {alpha}, Passes = {passes}', coherence)

        return coherence

    # Perform LDA using the Triple-based model
    def get_topics_triple(self, num_topics, alpha='auto', passes=1, workers=2):
        lda_model = LdaModel(self.corpus_triple, num_topics=num_topics, id2word=self.dictionary_triple, passes=passes, alpha=alpha)
        coherence = self.get_coherence_score(lda_model, self.dictionary_triple, self.triples, processes=workers)
        self.print_topics(lda_model, f'Triples - Topics = {num_topics}, Alpha = {alpha}, Passes = {passes}', coherence)

        return coherence, lda_model

    # Perform LDA using SVO-based model
    def get_topics_svo(self, num_topics, alpha='auto', passes=1, workers=2):
        lda_model = LdaModel(self.corpus_subject, num_topics=num_topics, id2word=self.dictionary_subject, passes=passes, alpha=alpha)
        coherence_s = self.get_coherence_score(lda_model, self.dictionary_subject, self.subjects)
        self.print_topics(lda_model, f'Subject - Topics = {num_topics}, Alpha = {alpha}, Passes = {passes}', coherence_s)

        lda_model = LdaModel(self.corpus_verb, num_topics=num_topics, id2word=self.dictionary_verb, passes=passes, alpha=alpha)
        coherence_v = self.get_coherence_score(lda_model, self.dictionary_verb, self.verbs)
        self.print_topics(lda_model, f'Verb - Topics = {num_topics}, Alpha = {alpha}, Passes = {passes}', coherence_v)

        lda_model = LdaModel(self.corpus_object, num_topics=num_topics, id2word=self.dictionary_object, passes=passes, alpha=alpha)
        coherence_o = self.get_coherence_score(lda_model, self.dictionary_object, self.objects, processes=workers)
        self.print_topics(lda_model, f'Object - Topics = {num_topics}, Alpha = {alpha}, Passes = {passes}', coherence_o)

        return (coherence_s + coherence_v + coherence_o) / 3.0

    # Calculate and print coherence (possible coherence metrics: c_v, u_mass, c_uci)
    def get_coherence_score(self, lda_model, dictionary, data, coherence_metric='c_v', processes=2):
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=dictionary, coherence=coherence_metric, processes=processes) 
        coherence_lda = coherence_model_lda.get_coherence()

        return coherence_lda

    def print_topics(self, lda_model, name, coherence_score):
        # Write topics to a file in source/output_files folder
        print(f'\nModel trained: {name}')
        with open('source/output_files/lda_analysis.txt', 'a', encoding='utf-8') as f:
            f.write(name + '\n')
            for idx, topic in lda_model.print_topics(num_words=5):
                f.write('Topic {}: {}'.format(idx + 1, topic))
                f.write('\n')
            f.write('\n')
            f.write(f'Coherence Score: {coherence_score}')
            f.write('\n\n\n')
        print('Topics written to source/output_files/lda_analysis.txt')
