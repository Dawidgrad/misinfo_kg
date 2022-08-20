import os
import neuralcoref
from tqdm import tqdm
from source.lda import LDA
from source.yodie import Yodie
from source.knowledge_graph import KnowledgeGraph
from source.ner import DeepPavlov, Flair, Gate, Spacy
from source.pos import SpacyTagger
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
        # -------------------- Preparing the data --------------------
        # Clear the input & output files
        self.clean_up_files()
        
        # TODO Command line interface to select between datasets (+ future options)
        dataset_name = 'Ukraine'

        # Retrieve semantic triples using OpenIE
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

        # Remove rows with empty subject, object or verb
        output_data = output_data[output_data['Subject'].notnull()]
        output_data = output_data[output_data['Object'].notnull()]
        output_data = output_data[output_data['Verb'].notnull()]

        # -------------------- LDA --------------------
        self.perform_lda(output_data)

        # -------------------- NER --------------------
        # Extract named entities from the data using various NER packages
        ne_dict = dict()

        spacy = Spacy()
        ne_dict = self.extract_ne(spacy, ne_dict, filenames)

        flair = Flair()
        ne_dict = self.extract_ne(flair, ne_dict, filenames)

        deeppavlov = DeepPavlov()
        ne_dict = self.extract_ne(deeppavlov, ne_dict, filenames)

        gate = Gate(self.api_key, self.api_password)
        ne_dict = self.extract_ne(gate, ne_dict, filenames)

        # -------------------- Entity Linking --------------------
        # Entity linking DBpedia 
        ne_links = self.ne_disambiguation(ne_dict.keys())
        print(ne_links)

        # -------------------- Knowledge Graph --------------------
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
            # Make sure to ignore NaN values
            if isinstance(row[f'{column_name}'], str):
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
    
    # Use OpenIE to extract the triples from the Russo-Ukrainian war misinformation data
    def ukraine_misinfo(self):
        # Retrieve the misinformation data        
        data = pd.read_json(f'{self.working_dir}\source\\resources\stratcom-data.json')
        filenames = self.prepare_data(data, 'summary')

        # Run the Stanford CoreNLP java package over all previously created files
        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist', 
                f'{self.working_dir}\source\input_files\\filelist.txt', '-output',  
                f'{self.working_dir}\source\output_files\openie_output_ukraine_claims.txt', '-tokenize.options', 'untokenizable=noneDelete']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()
        
        return filenames

    # Use OpenIE to extract the triples from covid-19 misinformation data
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

    # Perform LDA with BOW, Triples, and SVO models
    def perform_lda(self, output_data):
        triples = subjects = objects = verbs = list()
        for index, row in output_data.iterrows():
            triples.append(row['Subject'] + ' ' + row['Verb'] + ' ' + row['Object'])
            subjects.append(row['Subject'])
            objects.append(row['Object'])
            verbs.append(row['Verb'])
            
        lda = LDA(triples, subjects, objects, verbs)

        # LDA performed on SVO separately
        lda.get_topics_svo(num_topics=5, passes=2, workers=8)

        # LDA performed on BOW
        lda.get_topics_bow(num_topics=5, passes=2, workers=8)

        # LDA performed on triples
        lda.get_topics_triple(num_topics=5, passes=2, workers=8)

    def extract_ne(self, ner, ne_dict, filenames):
        sentences = []

        # Get the sentences from input files
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    sentences.append(line)

        # Get the entities from the sentences
        entities = ner.get_entities(sentences)

        for ne in entities:
            if ne in ne_dict:
                ne_dict[ne] += 1
            else: 
                ne_dict[ne] = 1

        return ne_dict

    def ne_disambiguation(self, entities):
        # Call GATE Yodie
        yodie = Yodie(self.api_key, self.api_password)
        yodie_outputs = {}

        # # Get the sentences from input files
        # for filename in filenames:
        #     with open(filename, 'r', encoding='utf-8') as file:
        #         for line in file:
        #             yodie_outputs.append(yodie.call(line))
        
        print('Disambiguating NEs...')
        for entity in tqdm(entities):
            entity_link = yodie.call(entity)
            if entity_link:
                yodie_outputs[entity_link[0][1]] = entity_link[0][0]
        print()

        return yodie_outputs

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

    def clean_up_files(self):
        # Delete all files in input_files directory
        for filename in os.listdir(f'{self.working_dir}\source\input_files'):
            silent_remove(f'{self.working_dir}\source\input_files\\{filename}')

        # Delete all files in output_files directory
        for filename in os.listdir(f'{self.working_dir}\source\output_files'):
            silent_remove(f'{self.working_dir}\source\output_files\\{filename}')
