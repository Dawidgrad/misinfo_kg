from abc import ABC, abstractmethod
from ratelimit import limits, sleep_and_retry
import requests
import subprocess
import spacy
import json
import sys

class NamedEntityRecogniser(ABC):

    @abstractmethod
    def get_entities(self, sentences):
        pass

class Spacy(NamedEntityRecogniser):
    def __init__(self):
        # Download the en_core_web_sm model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    def get_entities(self, sentences): 
        # Load the model
        nlp = spacy.load('en_core_web_sm')
        entities = list()
        ignored_labels = ['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'DATE']

        # Get the NER tags
        for sentence in sentences:
            doc = nlp(sentence)
            for ent in doc.ents:
                if ent.label_ not in ignored_labels:
                    entities.append(ent)
        
        return entities

class Gate(NamedEntityRecogniser):
    def __init__(self, key_id, password) -> None:
        self.key_id = key_id
        self.password = password

    # Rate limit API calls to GATE
    @sleep_and_retry
    @limits(calls=1, period=1)
    def call_gate_api(self, sentence):
        results = []
        endpoint_url = 'https://cloud-api.gate.ac.uk/process/annie-named-entity-recognizer'
        headers = {
            "Content-Type": "text/plain"
        }

        response = requests.post(endpoint_url,
                                auth=(self.key_id, self.password),
                                data = sentence.encode('utf-8'),
                                headers = headers)
        results.append(response.text)

        return results

    def get_entities(self, sentences):
        entities = list()
       
        # Get the NER tags
        for sentence in sentences:
            raw_data = self.call_gate_api(sentence)[0]
            dict_output = json.loads(raw_data)['entities']
            text = json.loads(raw_data)['text']

            if 'Location' in dict_output:
                for item in dict_output['Location']:
                    entities.append(text[item['indices'][0]:item['indices'][1]])

            if 'Person' in dict_output:
                for item in dict_output['Person']:
                    entities.append(text[item['indices'][0]:item['indices'][1]])

            if 'Organization' in dict_output:
                for item in dict_output['Organization']:
                    entities.append(text[item['indices'][0]:item['indices'][1]])

        return entities
    
class Flair(NamedEntityRecogniser):
    def __init__(self):
        pass
    
    def get_entities(self, sentences): 
        pass

class DeepPavlov(NamedEntityRecogniser):
    def __init__(self):
        pass
    
    def get_entities(self, sentences): 
        pass