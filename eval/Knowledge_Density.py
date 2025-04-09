import os
import numpy as np
from argparse import ArgumentParser

from FActScore.factscore.atomic_facts import AtomicFactGenerator, normalize_answer
from trim import process_document
from ..src.tools.lm import *

def deduplicate_atomic_facts(all_facts):
    """Deduplicate atomic facts by normalizing them and removing duplicates."""

    prompt = """
    You need to remove atomic knowledge facts with repetitive content from the following list. 
    Even if the expressions differ but convey the same fact, they should be considered duplicates and deleted.
    Here are the facts:

    """ + "\n".join(["- " + fact for fact in all_facts]) + """

    Please output only the deduplicated atomic facts, formatted as follows:
    - atomic fact 1
    - atomic fact 2
    - atomic fact 3
    ...
    """
    lm = OpenAIModel_dashscope(model='gpt-4o', max_tokens=2000)
    output = lm(prompt)[0]

    output = re.findall(r"-\s*(.*)", output)
    
    # Return deduplicated atomic facts
    return output

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

    deduplicated_splits = [deduplicate_atomic_facts(split.tolist()) for split in split_facts]

    combined_facts = []
    for deduplicated_part in deduplicated_splits:
        combined_facts += deduplicated_part

    deduplicated_facts = deduplicate_atomic_facts(combined_facts)
    
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