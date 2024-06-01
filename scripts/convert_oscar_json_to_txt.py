#!/usr/bin/env python3

"""
Convert OSCAR dataset from json format to txt format
Output file will have the following format:
    One sentence per line
    An empty line is used as a separator between documents
"""

import argparse
import json
from tqdm import tqdm
from typing import List


def convert_oscar_json_to_txt(args: argparse.Namespace) -> None:
    documents: List[str] = []
    for input_file in args.input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for document in tqdm(json_data, desc='Reading json file'):
                sentences = document[args.feature].split(args.seperator)
                sentences = [sent.strip() for sent in sentences]
                sentences = [sent for sent in sentences if sent]
                documents.append('\n'.join(sentences))

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for document in documents:
            f.write(document)
            f.write('\n\n')

    total_lines = len(documents)
    print(f'File saved in {args.output_file}')
    print(f'Total lines: {total_lines}')

def main():
    parser = argparse.ArgumentParser(
        description='Convert OSCAR dataset from json format to txt format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_file',
        nargs='+',
        help='Path to the input json files',
        type=str,
    )
    parser.add_argument(
        '-o',
        '--output-file',
        help='Path to place the output txt file',
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
        '--seperator',
        help='Seperator to use for splitting sentences in a document',
        type=str,
        default='.',
    )

    args = parser.parse_args()
    convert_oscar_json_to_txt(args)


if __name__ == '__main__':
    main()
