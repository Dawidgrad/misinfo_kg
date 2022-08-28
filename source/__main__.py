from source.kg_construction import KGConstruction
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for the classification script')
    # TODO Change to use Python's default arguments
    parser.add_argument('-w', '--working_dir', help='Absolute path to the application\'s working directory', required=True)
    parser.add_argument('-s', '--stanford_path', help='Absolute path to the StanfordCoreNLP module', required=True)
    parser.add_argument('-k', '--api_key', help='Key ID to GATE Cloud', required=True)
    parser.add_argument('-p', '--api_password', help='Password to GATE Cloud', required=True)
    args = parser.parse_args()
    
    # Check if the user wants to prepare input files
    print('Would you like to prepare input files?')
    print('1) Yes')
    print('2) No')
    choice = input('Enter corresponding number: ')

    # Validate user input and pass the choice to KGConstruction class
    if choice == '1':
        prepare_files = True
    elif choice == '2':
        prepare_files = False
    else:
        print('Invalid choice')
        exit()

    # Ask user what they want to run: 1) LDA 2) Graph Construction
    print('\nWhat would you like to do?')
    print('1) LDA')
    print('2) Graph Construction')
    choice = input('Enter corresponding number: ')

    # Validate user input and pass the choice to KGConstruction class
    if choice == '1':
        print('You have chosen LDA\n')
        misinfo_classification = KGConstruction(args.working_dir, args.stanford_path, args.api_key, args.api_password, lda=True, prepare_files=prepare_files)
    elif choice == '2':
        print('You have chosen Graph Construction\n')
        misinfo_classification = KGConstruction(args.working_dir, args.stanford_path, args.api_key, args.api_password, lda=False, prepare_files=prepare_files)
    else:
        print('Invalid choice')
        exit()

    misinfo_classification.run()