from source.classification import Classification
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for the classification script')
    parser.add_argument('-w', '--working_dir', help='Absolute path to the application\'s working directory', required=True)
    parser.add_argument('-s', '--stanford_path', help='Absolute path to the StanfordCoreNLP module', required=True)
    args = parser.parse_args()

    misinfo_classification = Classification(args.working_dir, args.stanford_path)
    misinfo_classification.run()