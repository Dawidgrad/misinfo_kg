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

    def get_entities(self, transcripts): 
        # Load the model
        nlp = spacy.load('en_core_web_sm')
        entities = list()
        ignored_labels = ['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'WORK_OF_ART']

        # Get the NER tags
        for single_transcript in transcripts:
            for segment in single_transcript:
                doc = nlp(segment)
                for ent in doc.ents:
                    if ent.label_ not in ignored_labels:
                        entities = entities + [([ent.start_char, ent.end_char], ent.label_)]
                entities.append('segment_end')
            entities.append('transcript_end')
        
        return entities