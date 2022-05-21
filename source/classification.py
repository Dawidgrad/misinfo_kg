import neuralcoref
from source.ner import Spacy
from source.pos import SpacyTagger, TextblobTagger
from source.utils import silent_remove
import pandas as pd
import subprocess
import spacy

class Classification:
    def __init__(self, working_dir, stanford_path) -> None:
        self.working_dir = working_dir
        self.stanford_path = stanford_path

    def run(self):
        # TODO Clear the input & output files
        # TODO Command line interface to select between datasets (+ future options)
        # TODO Change name of the Classification class to something more suitable
        dataset_name = 'Ukraine'

        # Retrieve semantic triples
        if (dataset_name == 'Ukraine'):
            filenames = self.ukraine_misinfo()
            output_name = 'openie_output_ukraine_claims'
        elif (dataset_name == 'Covid'):
            filenames = self.covid_misinfo()
            output_name = 'openie_output_covid_claims'
            # TODO Handle explanations too?

        # Convert the output of OpenIE to a csv file
        read_output = pd.read_csv(f'{self.working_dir}/source/output_files/{output_name}.txt', encoding='ISO-8859-1',
                                    sep='\t', names=['Confidence', 'Subject', 'Verb', 'Object'])
        read_output.to_csv(f'{self.working_dir}/source/output_files/{output_name}.csv')

        # Extract named entities from the data
        named_entities = self.extract_ne(filenames)
        print('Named entities')
        print(f'{named_entities}\n')
        print(f'Number of found entities in the entire dataset: {len(named_entities)}')

        # Extract part of speech tags from the data
        pos_tags = self.extract_pos(filenames)
        print('Part of speech tags')
        print(f'{pos_tags}')

        nlp = spacy.load('en')
        neuralcoref.add_to_pipe(nlp)
        doc = nlp(u'My sister has a dog. She loves him.')
        print(doc._.has_coref)
        print(doc._.coref_clusters)

        self.coreference_resolution()
        self.verb_lemmatisation()

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
            file.write(f'{cell}.\n')

            if (idx + 1) % 1 == 0:
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

        return filenames
    
    def ukraine_misinfo(self):
        # Retrieve the misinformation data        
        data = pd.read_json(f'{self.working_dir}\source\\resources\warDisinfoClaims.json')
        filenames = self.prepare_data(data, 'claim')

        # Run the Stanford CoreNLP java package over all previously created files
        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist', 
                f'{self.working_dir}\source\input_files\\filelist.txt', '-output',  
                f'{self.working_dir}\source\output_files\openie_output_ukraine_claims.txt', '-tokenize.options', 'untokenizable=noneDelete']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()
        
        return filenames

    def covid_misinfo(self):
        # Retrieve the misinformation data   
        data = pd.read_json(f'{self.working_dir}\source\\resources\IFCN_COVID19_12748.json')

        # Prepare and process claims
        filenames = self.prepare_data(data, 'Claim')

        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist', 
                f'{self.working_dir}\source\input_files\\filelist.txt', '-output',  
                f'{self.working_dir}\source\output_files\openie_output_covid_claims.txt', '-tokenize.options', 'untokenizable=noneDelete']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()

        # Prepare and process explanations
        filenames = filenames + self.prepare_data(data, 'Explaination')
        
        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist', 
                f'{self.working_dir}\source\input_files\\filelist.txt', '-output',  
                f'{self.working_dir}\source\output_files\openie_output_covid_explanations.txt', '-tokenize.options', 'untokenizable=noneDelete']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()

        return filenames

    def extract_ne(self, filenames):
        sentences = []

        # Get the sentences from input files
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    sentences.append(line)

        # Get the entities from the sentences
        spacy_ner = Spacy()
        entities = spacy_ner.get_entities(sentences)

        return entities

    def extract_pos(self, filenames):
        sentences = []

        # Get the sentences from input files
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    sentences.append(line)

        # Get the pos tags from the sentences
        textblob_pos = SpacyTagger()
        pos_tags = textblob_pos.get_tags(sentences)

        return pos_tags

    def coreference_resolution(self):

        return None

    def verb_lemmatisation(self):

        return None

    def clean_up_files(self):

        return None