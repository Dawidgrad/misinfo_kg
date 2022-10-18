
def prepare_data(dataset):
    print(f'Preparing data...')
    # Retrieve semantic triples using OpenIE
    if (dataset == DatasetName.UKRAINE):
        filenames = self.ukraine_misinfo()
        output_name = 'openie_output_ukraine_claims'
    elif (dataset == DatasetName.COVID):
        filenames = self.covid_misinfo()
        output_name = 'openie_output_covid_claims'

# Use OpenIE to extract the triples from the Russo-Ukrainian war misinformation data
def ukraine_misinfo(self):
    # Retrieve the misinformation data
    data = pd.read_json(
        f'{self.working_dir}\source\\resources\stratcom-data.json')
    filenames = self.prepare_data(data, 'summary')

    # Run the Stanford CoreNLP java package over all previously created files
    args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE',
            '-filelist', f'{self.working_dir}\source\input_files\\filelist.txt',
            '-output', f'{self.working_dir}\source\output_files\openie_output_ukraine_claims.txt',
            '-tokenize.options', 'untokenizable=noneDelete', '-max_entailments_per_clause', '1']
    args = ' '.join(args)

    self.process = subprocess.Popen(
        args, shell=True, stderr=subprocess.STDOUT).wait()

    return filenames

# Use OpenIE to extract the triples from covid-19 misinformation data
def covid_misinfo(self):
    # Retrieve the misinformation data
    data = pd.read_json(
        f'{self.working_dir}\source\\resources\IFCN_COVID19_12748.json')

    # Prepare and process claims
    filenames = self.prepare_data(data, 'Claim')

    args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist',
            f'{self.working_dir}\source\input_files\\filelist.txt', '-output',
            f'{self.working_dir}\source\output_files\openie_output_covid_claims.txt', '-tokenize.options', 'untokenizable=noneDelete']
    args = ' '.join(args)

    self.process = subprocess.Popen(
        args, shell=True, stderr=subprocess.STDOUT).wait()

    # Prepare and process explanations
    filenames = filenames + self.prepare_data(data, 'Explaination')

    args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist',
            f'{self.working_dir}\source\input_files\\filelist.txt', '-output',
            f'{self.working_dir}\source\output_files\openie_output_covid_explanations.txt', '-tokenize.options', 'untokenizable=noneDelete']
    args = ' '.join(args)

    self.process = subprocess.Popen(
        args, shell=True, stderr=subprocess.STDOUT).wait()

    return filenames