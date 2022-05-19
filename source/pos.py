from abc import ABC, abstractmethod
from textblob import TextBlob
import spacy

class POSTagger(ABC):

    @abstractmethod
    def get_tags(self, sentences):
        pass

class TextblobTagger(POSTagger):

    def get_tags(self, sentences):
        tags = []

        # Extract the pos tags
        for sentence in sentences:
            blob = TextBlob(sentence)
            tags.append(blob.tags)

        return tags

class SpacyTagger(POSTagger):

    def get_tags(self, sentences):
        tags = []
        nlp = spacy.load("en_core_web_sm")

        for sentence in sentences:
            doc = nlp(sentence)
            for token in doc:
                tags = tags + [(token.text, token.pos_)]

        return tags

