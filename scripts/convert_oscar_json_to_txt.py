#!/usr/bin/env python3

"""
Convert OSCAR dataset from json format to txt format
Output file will have the following format:
    One sentence per line
    An empty line is used as a separator between documents
"""

import argparse
import json
import re
from tqdm import tqdm
from typing import List


def convert_oscar_json_to_txt(
    input_files: list[str],
    output_file: str,
    feature: str,
    seperators: str = '.?!',
) -> None:
    documents: List[str] = []
    total_sents = 0
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for document in tqdm(json_data, desc='Reading json file', unit=' documents'):
                sents = re.split(fr'(?<=[{seperators}])\s+', document[feature])
                sents = [sent.strip() for sent in sents]
                sents = [sent for sent in sents if sent]
                documents.append('\n'.join(sents))
                total_sents += len(sents)

    with open(output_file, 'w', encoding='utf-8') as f:
        for document in documents:
            f.write(document)
            f.write('\n\n')

    print(f'Wrote {total_sents} sentences to {output_file}')

def main():
    parser = argparse.ArgumentParser(
        description='Convert OSCAR dataset from json format to txt format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Path to the json files',
        type=str,
    )
    parser.add_argument(
        '-o',
        '--output-file',
        help='Path to place the output text file',
        type=str,
        default='./output.txt',
    )
    parser.add_argument(
        '-f',
        '--feature',
        help='Feature to extract from the json file',
        type=str,
        default='text',
    )
    parser.add_argument(
        '-s',
        '--seperators',
        help='Seperators to use for splitting sentences in a document',
        type=str,
        default='.?!',
    )

    args = parser.parse_args()
    convert_oscar_json_to_txt(args.input_files, args.output_file, args.feature, args.seperators)


if __name__ == '__main__':
    main()
