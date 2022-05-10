from source.utils import silent_remove
import pandas as pd
import subprocess

class Classification:
    def __init__(self, working_dir, stanford_path) -> None:
        self.working_dir = working_dir
        self.stanford_path = stanford_path

    def run(self):
        # Retrieve the misinformation data
        # data = pd.read_json(f'{self.working_dir}\source\\resources\IFCN_COVID19_12748.json')
        # data = data[['Claim', 'Explaination']]
        # data = data.reset_index()

        data = pd.read_json(f'{self.working_dir}\source\\resources\warDisinfoClaims.json')
        data = data[['Claim', 'Title']]
        data = data.reset_index()

        # self.data[['Claim', 'Explaination']].head(50).to_csv(r'C:\Users\gradd\Documents\Software Development\ifcn_covid_kg\source\outputs\ifcn_head.csv')

        # Split the covid data into smaller batches so they can be passed to Stanford's package
        # This is to avoid using too much memory at once
        filenames = []
        file_idx = 0

        # Make sure a new file is created to avoid appending to the output from previous run
        new_filename = f'{self.working_dir}\source\input_files\\file_{file_idx}.txt'
        silent_remove(new_filename)
        file = open(new_filename, 'a', encoding='utf-8')

        for idx, row in data.iterrows():
            claim = row['Claim'].strip().replace('\n', ' ')
            # Ignore explanation for now
            # explanation = row['Explaination'].strip() 
            
            file.write(f'{claim}\n')

            if (idx + 1) % 50 == 0:
                file.close()
                file_idx += 1
                filenames.append(new_filename)

                new_filename = f'{self.working_dir}\source\input_files\\file_{file_idx}.txt'
                silent_remove(new_filename)

                file = open(new_filename, 'a', encoding='utf-8')
        
        filelist_path = f'{self.working_dir}\source\input_files\\filelist.txt'
        silent_remove(filelist_path)
        with open(filelist_path, 'a') as file:
            for filename in filenames:
                file.write(f'{filename}\n')

        # Run the Stanford CoreNLP java package over all previously created files
        args = ['java', '-mx8g', '-cp', self.stanford_path, 'edu.stanford.nlp.naturalli.OpenIE', '-filelist', 
                f'{self.working_dir}\source\input_files\\filelist.txt', '-output',  
                f'{self.working_dir}\source\output_files\openie_output.txt', '-tokenize.options', 'untokenizable=noneDelete']
        args = ' '.join(args)

        self.process = subprocess.Popen(args, shell=True, stderr=subprocess.STDOUT).wait()
        
        print('Hello there')
        return None
