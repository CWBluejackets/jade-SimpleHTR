#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from typing import Tuple
from datetime import datetime, timedelta

from panoptes_client import Panoptes, Project, SubjectSet, Subject

PROJECT_ID = 18285
MAX_BATCH_SIZE = 1000


def load_manifest(manifest_path) -> dict:
    print(f'Loading manifest from {manifest_path}')
    with open(manifest_path, 'r') as f:
        manifest0 = json.load(f)
    return manifest0


def create_batches(infix: str, manifest0: dict) -> Tuple[dict, dict]:
    """
    :param infix: batch name prefix
    :param manifest0: manifest dict
    :return: batches dict batch_name -> list of tuples (filename, metadata)
    """
    sequence = 0
    sequence_ids0 = {}
    vessel_year_batches = defaultdict(list)
    for filename, metadata in manifest0.items():
        vessel = metadata['vessel']
        year = metadata['#muster_year'] or 0
        vessel_year_batches[vessel, year].append((filename, metadata))
    results = defaultdict(list)
    print(f'Number of vessel-year combinations: {len(vessel_year_batches)}')
    for (vessel, year), big_batch in vessel_year_batches.items():
        for i in range(0, len(big_batch), MAX_BATCH_SIZE):
            sequence_id = f'{sequence:05d}'
            batch0 = big_batch[i: i + MAX_BATCH_SIZE]
            batch_name0 = f'{vessel} {year:04d} {infix}_{sequence_id}'
            sequence += 1
            results[batch_name0] = batch0
            sequence_ids0[sequence_id] = batch_name0
    return results, sequence_ids0


def make_batch_filename(manifest_path: str, seq_id: str) -> str:
    base = os.path.splitext(manifest_path)[0]
    return f'{base}_BATCH_{seq_id}.json'


def write_batch_manifest(filename: str, batch_name0: str, batch0:dict) -> None:
    data = {'batch_name': batch_name0,
            'contents': batch0}
    print(f'Writing: {filename} <- {batch_name0}')
    with open(filename, 'w') as f:
        json.dump(data, f, indent=1, sort_keys=True)
    return


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate Zooniverse manifest")

    oparser.add_argument('-m', dest='manifest_path',
                         type=str,
                         required=True,
                         metavar='PATH/manifest.json',
                         help='path to JSON manifest file in image directory')

    oparser.add_argument('-s', dest='subject_set_infix',
                         type=str,
                         required=True,
                         metavar='NAME',
                         help='infix for subject set name')

    oparser.add_argument('-n', dest='dry_run',
                         default=False,
                         action='store_true',
                         help='do not generate output files')

    options = oparser.parse_args()

    START_TIME = datetime.now()

    manifest = load_manifest(options.manifest_path)
    input_directory = os.path.dirname(options.manifest_path)
    nbr_subjects = len(manifest)
    print(f'Number of subjects: {nbr_subjects}')

    named_batches, sequence_ids = create_batches(options.subject_set_infix, manifest)
    print(f'Number of batches: {len(named_batches)}')

    if options.dry_run:
        print('Not generating batch manifests')
    else:
        for sequence_id, batch_name in sequence_ids.items():
            batch_filename = make_batch_filename(options.manifest_path, sequence_id)
            write_batch_manifest(batch_filename, batch_name, named_batches[batch_name])

