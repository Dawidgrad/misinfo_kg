import os
from pydoc import resolve
import neuralcoref
from tqdm import tqdm
from source.lda import LDA
from source.yodie import Yodie
from source.knowledge_graph import KnowledgeGraph
from source.ner import Flair, Gate, Spacy
from source.utils import silent_remove
import pandas as pd
import subprocess
import spacy
import nltk

class KGConstruction:
    def __init__(self, working_dir, stanford_path, api_key, api_password, lda, prepare_files) -> None:
        self.working_dir = working_dir
        self.stanford_path = stanford_path
        self.api_key = api_key
        self.api_password = api_password
        self.lda = lda
        self.prepare_files = prepare_files
        nltk.download('omw-1.4')

        # Prepare coreference resolution
        self.nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(self.nlp)

    def run(self):
        if self.prepare_files:
            # -------------------- Preparing the data --------------------
            # Clear the input & output files
            self.clean_up_files()
            
            # TODO Command line interface to select between datasets (+ future options)
            dataset_name = 'Ukraine'

            print(f'Preparing data...')
            # Retrieve semantic triples using OpenIE
            if (dataset_name == 'Ukraine'):
                filenames = self.ukraine_misinfo()
                output_name = 'openie_output_ukraine_claims'
            elif (dataset_name == 'Covid'):
                filenames = self.covid_misinfo()
                output_name = 'openie_output_covid_claims'
                # TODO Handle explanations too?
        
        output_name = 'openie_output_ukraine_claims' # Remove later
        # Convert the output of OpenIE to a csv file
        output_data = pd.read_csv(f'{self.working_dir}/source/output_files/{output_name}.txt', encoding='ISO-8859-1',
                                    sep='\t', names=['Confidence', 'Subject', 'Verb', 'Object'])
        output_data.to_csv(f'{self.working_dir}/source/output_files/{output_name}.csv')

        # Remove rows with empty subject, object or verb
        output_data = output_data[output_data['Subject'].notnull()]
        output_data = output_data[output_data['Object'].notnull()]
        output_data = output_data[output_data['Verb'].notnull()]

        # -------------------- LDA --------------------
        if self.lda:
            self.perform_lda(output_data)

        else:
            # -------------------- NER --------------------
            # Extract named entities from the data using various NER packages
            ne_dict = dict()

            spacy = Spacy()
            ne_dict = self.extract_ne(spacy, ne_dict, filenames)

            flair = Flair()
            ne_dict = self.extract_ne(flair, ne_dict, filenames)

            gate = Gate(self.api_key, self.api_password)
            ne_dict = self.extract_ne(gate, ne_dict, filenames)

            # -------------------- Entity Linking --------------------
            # Entity linking DBpedia 
            ne_links = self.ne_disambiguation(ne_dict.keys())
            print(ne_links)

            # -------------------- Knowledge Graph --------------------
            knowledge_graph = KnowledgeGraph()

            for row in output_data.iterrows():
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
        for idx, row in tqdm(data.iterrows()):
            # Make sure to ignore NaN values
            if isinstance(row[f'{column_name}'], str):
                cell = row[f'{column_name}'].strip().replace('\n', ' ')
                doc = self.nlp(cell)
                file.write(f'{cell}. Dummy is a dummy.\n') 
                # file.write(f'{doc._.coref_resolved}.\n') # Use disambiguated text   

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
        print(filenames)

        # for idx, file in tqdm(enumerate(filenames)):
        #     # Run the Stanford CoreNLP java package over all previously created files
        #     args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', file,
        #             '-output', f'{self.working_dir}\source\output_files\openie_output_ukraine_claims_{idx}.txt',
        #             '-tokenize.options', 'untokenizable=noneDelete', '-max_entailments_per_clause', '1']
        #     args = ' '.join(args)

        #     self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()

        # Run the Stanford CoreNLP java package over all previously created files
        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', 
                '-filelist', f'{self.working_dir}\source\input_files\\filelist.txt',
                '-output', f'{self.working_dir}\source\output_files\openie_output_ukraine_claims.txt',
                '-tokenize.options', 'untokenizable=noneDelete', '-max_entailments_per_clause', '1']
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
        triples = list()
        subjects = list()
        objects = list()
        verbs = list()
        doc_triples = list()
        doc_subjects = list()
        doc_objects = list()
        doc_verbs = list()

        for row in output_data.itertuples():
            if row[2] == 'Dummy':
                triples.append(doc_triples)
                subjects.append(doc_subjects)
                objects.append(doc_objects)
                verbs.append(doc_verbs)
                doc_triples = list()
                doc_subjects = list()
                doc_objects = list()
                doc_verbs = list()
                continue
            doc_triples.append(row[2] + ' ' + row[3] + ' ' + row[4])
            doc_subjects.append(row[2])
            doc_objects.append(row[4])
            doc_verbs.append(row[3])
            
        lda = LDA(triples, subjects, objects, verbs)

        # Remove the old output file before staring the LDA process
        silent_remove(f'{self.working_dir}\source\output_files\lda_analysis.txt')

        args = [[5, 1], [5, 10], [5, 100], [10, 1], [10, 10], [10, 100], [20, 1], [20, 10], [20, 100]]
        # args = [[2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1], [11, 1], [12, 1], [13, 1], [14, 1], [15, 1], [16, 1], [17, 1], [18, 1], [19, 1], [20, 1], [21, 1], [22, 1], [23, 1], [24, 1], [25, 1], [26, 1], [27, 1], [28, 1], [29, 1], [30, 1], [31, 1], [32, 1], [33, 1], [34, 1], [35, 1], [36, 1], [37, 1], [38, 1], [39, 1], [40, 1], [41, 1], [42, 1], [43, 1], [44, 1], [45, 1], [46, 1], [47, 1], [48, 1], [49, 1], [50, 1]]

        # LDA performed on triples
        [lda.get_topics_triple(num_topics=arg[0], passes=arg[1], workers=2) for arg in args]

        # LDA performed on SVO separately
        [lda.get_topics_svo(num_topics=arg[0], passes=arg[1], workers=2) for arg in args]

        # LDA performed on BOW
        [lda.get_topics_bow(num_topics=arg[0], passes=arg[1], workers=2) for arg in args]

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
        
        print('Disambiguating NEs...')
        for entity in tqdm(entities):
            entity_link = yodie.call(entity)
            if entity_link:
                yodie_outputs[entity_link[0][1]] = entity_link[0][0]
        print()

        return yodie_outputs

    def clean_up_files(self):
        # Delete all files in input_files directory
        for filename in os.listdir(f'{self.working_dir}\source\input_files'):
            silent_remove(f'{self.working_dir}\source\input_files\\{filename}')

        # Delete all files in output_files directory
        for filename in os.listdir(f'{self.working_dir}\source\output_files'):
            silent_remove(f'{self.working_dir}\source\output_files\\{filename}')
