#!/usr/bin/env python3

"""
Extract text from json files
File should be a json file with the following format:
{
    "<data_field>": [
        {
            "<key_1>": "...",
            "<key_2>": "..."
        },
        ...
    ]
}
"""

import argparse
import json
from numpy import require
from tqdm import tqdm
from typing import List


def extract_text_from_json(
    input_files: list[str],
    output_file: str = './output.txt',
    field: str = 'data',
    keys: list[str] = ['source'],
) -> None:
    if isinstance(input_files, str):
        input_files = [input_files]
    content = []
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if not field in json_data:
                raise ValueError(f'Data field {field} not found in {input_file}')
            for item in json_data[field]:
                for key in keys:
                    if key in item:
                        content.append(item[key])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

    total_lines = len(content)
    print(f'File saved in {output_file}')
    print(f'Total lines: {total_lines}')

def main():
    parser = argparse.ArgumentParser(
        description='Convert OSCAR dataset from json format to txt format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_files',
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
        '--field',
        help='Data field to extract from',
        type=str,
        default='data',
    )
    parser.add_argument(
        '--keys',
        nargs='+',
        help='Keys to extract from each item',
        type=str,
        default='source',
    )

    args = parser.parse_args()
    extract_text_from_json(
        args.input_files,
        output_file=args.output_file,
        field=args.field,
        keys=args.keys,
    )


if __name__ == '__main__':
    main()
