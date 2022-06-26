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

    misinfo_classification = KGConstruction(args.working_dir, args.stanford_path, args.api_key, args.api_password)
    misinfo_classification.run()