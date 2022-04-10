import pandas as pd
import subprocess

class Classification:
    def __init__(self, data_path, stanford_path) -> None:
        self.data_path = data_path
        self.stanford_path = stanford_path

    def run(self):
        # Retrieve the misinformation data
        covid_data = pd.read_json(self.data_path)
        covid_data = covid_data[['Claim', 'Explaination']]
        print(covid_data.shape)

        # self.data.drop('Debunk_Html', inplace=True, axis=1)
        # self.data.drop('Source_PageTextOriginal', inplace=True, axis=1)
        # self.data.drop('Source_PageTextEnglish', inplace=True, axis=1)
        # self.data[['Claim', 'Explaination']].head(50).to_csv(r'C:\Users\gradd\Documents\Software Development\ifcn_covid_kg\source\outputs\ifcn_head.csv')

        # Run the Stanford CoreNLP java package
        args = ['java', '-mx1g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', './test_i.txt', './test_o.txt']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT)

        return None
