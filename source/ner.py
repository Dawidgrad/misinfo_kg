from abc import ABC, abstractmethod
import subprocess
import sys
import spacy

class NamedEntityRecogniser(ABC):

    @abstractmethod
    def get_entities(self):
        pass

    @abstractmethod
    def convert_format(self):
        pass

class Spacy(NamedEntityRecogniser):
    def __init__(self):
        # Download the en_core_web_sm model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    def get_entities(self, sentences): 
        # Load the model
        nlp = spacy.load('en_core_web_sm')
        entities = list()
        # ignored_labels = ['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'WORK_OF_ART']
        ignored_labels = []

        # Get the NER tags
        for sentence in sentences:
            doc = nlp(sentence)
            for ent in doc.ents:
                if ent.label_ not in ignored_labels:
                    entities = entities + [(ent, ent.label_)]
        
        return entities
    
    def convert_format(self):
        return super().convert_format()