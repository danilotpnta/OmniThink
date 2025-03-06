import os
import numpy as np
from argparse import ArgumentParser

from FActScore.factscore.atomic_facts import AtomicFactGenerator, normalize_answer
from trim import process_document

def knowledge_density_grade(response, api_path):        
    lines = response.split('\n')
    filtered_lines = [line for line in lines if not line.strip().startswith('#') ]
    response =  '\n'.join(filtered_lines)

    generator = AtomicFactGenerator(api_path, "./FActScore/factscore/.cache/factscore/demos")
    atomic_facts, _ = generator.run(response)

    all_facts = []
    for _, facts in atomic_facts:
        normalized_facts = [normalize_answer(fact) for fact in facts]
        all_facts += normalized_facts

    num_splits = int(len(response)/3000)
    split_facts = np.array_split(all_facts, num_splits)

    deduplicated_splits = [generator.deduplicate_atomic_facts(split.tolist()) for split in split_facts]

    combined_facts = []
    for deduplicated_part in deduplicated_splits:
        combined_facts += deduplicated_part

    deduplicated_facts = generator.deduplicate_atomic_facts(combined_facts)
    
    return len(deduplicated_facts)/len(response)

def main(args):
    stor_path = args.articlepath
    api_path = args.api_path

    # Replace with your file paths
    file_paths = []
    for dirs in os.listdir(stor_path):
        txt_path = os.path.join(stor_path, dirs)
        file_paths.append(txt_path)
        
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            response = f.read()
        print(knowledge_density_grade(response, api_path))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--articlepath', type=str, default='../results/article',
                        help='Directory to store the articles.')
    parser.add_argument('--api_path', type=str, default='./key',
                        help='Directory to store the model.')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to use for processing files.')
    main(parser.parse_args())