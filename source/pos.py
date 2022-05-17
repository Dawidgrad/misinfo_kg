from abc import ABC, abstractmethod
from textblob import TextBlob

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