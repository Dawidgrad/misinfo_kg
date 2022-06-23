import requests
from ratelimit import limits, sleep_and_retry

class GateCaller:
    def __init__(self, key_id, password) -> None:
        self.key_id = key_id
        self.password = password
        
    # Rate limit API calls to GATE
    @sleep_and_retry
    @limits(calls=1, period=1)
    def call_gate_api(self, text):
        results = []
        endpoint_url = 'https://cloud-api.gate.ac.uk/process/yodie-en'
        headers = {
            "Content-Type": "text/plain"
        }

        response = requests.post(endpoint_url,
                                auth=(self.key_id, self.password),
                                data = text.encode('utf-8'),
                                headers = headers)
        results.append(response.text)

        return results