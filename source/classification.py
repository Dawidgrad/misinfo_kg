from source.ner import Spacy
from source.utils import silent_remove
import pandas as pd
import subprocess

class Classification:
    def __init__(self, working_dir, stanford_path) -> None:
        self.working_dir = working_dir
        self.stanford_path = stanford_path

    def run(self):
        # Retrieve semantic triples
        # self.ukraine_misinfo()
        self.covid_misinfo()

        # Read the sentences from the input files
        # sentences = 

        # named_entities = self.extract_ne(sentences)

        return None

    def prepare_data(self, data, column_name):
        # Filter for a specific column
        data = data[[f'{column_name}']]
        data = data.reset_index()

        # Split the data into smaller batches so they can be passed to Stanford's package
        # This is to avoid using too much memory at once
        filenames = []
        file_idx = 0

        # Make sure a new file is created to avoid appending to the output from previous run
        new_filename = f'{self.working_dir}\source\input_files\\{column_name}_{file_idx}.txt'
        silent_remove(new_filename)
        file = open(new_filename, 'a', encoding='utf-8')
        filenames.append(new_filename)

        # Get column data
        for idx, row in data.iterrows():
            cell = row[f'{column_name}'].strip().replace('\n', ' ')
            file.write(f'{cell}\n')

            if (idx + 1) % 50 == 0:
                file.close()
                file_idx += 1

                new_filename = f'{self.working_dir}\source\input_files\\{column_name}_{file_idx}.txt'
                silent_remove(new_filename)

                file = open(new_filename, 'a', encoding='utf-8')
                filenames.append(new_filename)
        
        # Create filelist
        filelist_path = f'{self.working_dir}\source\input_files\\filelist.txt'
        silent_remove(filelist_path)
        with open(filelist_path, 'a') as file:
            for filename in filenames:
                file.write(f'{filename}\n')

        return None
    
    def ukraine_misinfo(self):
        # Retrieve the misinformation data        
        data = pd.read_json(f'{self.working_dir}\source\\resources\warDisinfoClaims.json')
        self.prepare_data(data, 'claim')

        # Run the Stanford CoreNLP java package over all previously created files
        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist', 
                f'{self.working_dir}\source\input_files\\filelist.txt', '-output',  
                f'{self.working_dir}\source\output_files\openie_output_ukraine_claims.txt', '-tokenize.options', 'untokenizable=noneDelete']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()
        
        return None

    def covid_misinfo(self):
        # Retrieve the misinformation data   
        data = pd.read_json(f'{self.working_dir}\source\\resources\IFCN_COVID19_12748.json')

        # Prepare and process claims
        self.prepare_data(data, 'Claim')

        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist', 
                f'{self.working_dir}\source\input_files\\filelist.txt', '-output',  
                f'{self.working_dir}\source\output_files\openie_output_covid_claims.txt', '-tokenize.options', 'untokenizable=noneDelete']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()

        # Prepare and process explanations
        self.prepare_data(data, 'Explaination')
        
        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist', 
                f'{self.working_dir}\source\input_files\\filelist.txt', '-output',  
                f'{self.working_dir}\source\output_files\openie_output_covid_explanations.txt', '-tokenize.options', 'untokenizable=noneDelete']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()

        return None

    def extract_ne(self, sentences):
        spacy_ner = Spacy()
        entities = spacy_ner.get_entities(sentences)

        return entities
