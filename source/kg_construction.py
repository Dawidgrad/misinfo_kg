import os
import neuralcoref
from tqdm import tqdm
from source.gate_caller import GateCaller
from source.knowledge_graph import KnowledgeGraph
from source.ner import Spacy
from source.pos import SpacyTagger, TextblobTagger
from source.utils import silent_remove
import pandas as pd
import subprocess
import spacy
import nltk

class KGConstruction:
    def __init__(self, working_dir, stanford_path, api_key, api_password) -> None:
        self.working_dir = working_dir
        self.stanford_path = stanford_path
        self.api_key = api_key
        self.api_password = api_password
        nltk.download('omw-1.4')

    def run(self):
        # Clear the input & output files
        self.clean_up_files()
        
        # TODO Command line interface to select between datasets (+ future options)
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
        output_data = pd.read_csv(f'{self.working_dir}/source/output_files/{output_name}.txt', encoding='ISO-8859-1',
                                    sep='\t', names=['Confidence', 'Subject', 'Verb', 'Object'])
        output_data.to_csv(f'{self.working_dir}/source/output_files/{output_name}.csv')

        # Extract named entities from the data
        named_entities = self.extract_ne(filenames)

        ne_dict = dict()
        for ne in named_entities:
            ne = str(ne[0])
            if ne in ne_dict:
                ne_dict[ne] += 1
            else: 
                ne_dict[ne] = 1

        # Named entity recognition and disambiguation against DBpedia
        ne_links = self.ne_disambiguation(filenames)
        print(ne_links)

        # Extract part of speech tags from the data
        # pos_tags = self.extract_pos(filenames)

        # TODO incorporate POS somehow

        # data = self.coreference_resolution(output_data)
        # data = self.verb_lemmatisation(data)
        
        # TODO incorporate coreference resolution
        # TODO incorporate verb lemmatisation

        knowledge_graph = KnowledgeGraph()

        for index, row in output_data.iterrows():
            for ne in ne_dict:
                # TODO Different ways to align the NER and triples?
                if ne in row['Subject'] and row['Object'] in ne_dict:
                    knowledge_graph.add_relation(row['Subject'], row['Verb'], row['Object'])
        
        knowledge_graph.export_csv(self.working_dir)
            
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

    def ne_disambiguation(self, filenames):
        # Call GATE Yodie
        gate = GateCaller(self.api_key, self.api_password)

        yodie_outputs = []

        # Get the sentences from input files
        for filename in tqdm(filenames):
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    yodie_outputs.append(gate.call_yodie(line))

        return yodie_outputs

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

    def coreference_resolution(self, data):
        resolved_data = data.reset_index()
        nlp = spacy.load('en')
        neuralcoref.add_to_pipe(nlp)

        for index, row in resolved_data.iterrows():
            print(row['Verb'])

        # doc = nlp(u'My sister has a dog. She loves him.')
        # print(doc._.has_coref)
        # print(doc._.coref_clusters)

        return resolved_data

    def verb_lemmatisation(self, data):
        lemmatised_data = data.reset_index()
        wnl = nltk.stem.WordNetLemmatizer()

        for index, row in lemmatised_data.iterrows():
            print(wnl.lemmatize(row['Verb'], pos='v'))

        return lemmatised_data

    def clean_up_files(self):
        # Delete all files in input_files directory
        for filename in os.listdir(f'{self.working_dir}\source\input_files'):
            silent_remove(f'{self.working_dir}\source\input_files\\{filename}')

        # Delete all files in output_files directory
        for filename in os.listdir(f'{self.working_dir}\source\output_files'):
            silent_remove(f'{self.working_dir}\source\output_files\\{filename}')
