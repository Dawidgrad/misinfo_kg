from enum import Enum
import os, errno

class Mode(Enum):
    LDA = 1
    CONSTRUCTION = 2
    DISAMBIGUATION = 3

class DatasetName(Enum):
    UKRAINE = 1
    COVID = 2

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def prepare_data(dataset):
    print(f'Preparing data...')
    # Retrieve semantic triples using OpenIE
    if (dataset == DatasetName.UKRAINE):
        filenames = self.ukraine_misinfo()
        output_name = 'openie_output_ukraine_claims'
    elif (dataset == DatasetName.COVID):
        filenames = self.covid_misinfo()
        output_name = 'openie_output_covid_claims'