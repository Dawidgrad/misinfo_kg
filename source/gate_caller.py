import ast
import requests
from ratelimit import limits, sleep_and_retry

class GateCaller:
    def __init__(self, key_id, password) -> None:
        self.key_id = key_id
        self.password = password
        
    # Rate limit API calls to GATE
    @sleep_and_retry
    @limits(calls=1, period=1)
    def call_yodie(self, text):
        endpoint_url = 'https://cloud-api.gate.ac.uk/process/yodie-en'
        headers = {
            "Content-Type": "text/plain"
        }

        response = requests.post(endpoint_url,
                                auth=(self.key_id, self.password),
                                data = text.encode('utf-8'),
                                headers = headers)

        return self.process_yodie_output(ast.literal_eval(response.text))

    # Extract DBPedia entities from yodie output
    def process_yodie_output(self, yodie_output):
        ne_links = []

        if 'entities' in yodie_output and 'Mention' in yodie_output['entities']:
            for mention in yodie_output['entities']['Mention']:
                inst = mention['inst']
                indices = mention['indices']
                text = yodie_output['text'][indices[0]:indices[1]]
                ne_links.append((inst, text))

        return ne_links
