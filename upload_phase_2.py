#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from typing import Tuple

from panoptes_client import Panoptes, Project, SubjectSet, Subject

PROJECT_ID = 18285
MAX_BATCH_SIZE = 1000


def create_subject_set(batch_name0, batch0, input_directory0, project) -> Tuple[SubjectSet, int]:
    subject_set0 = SubjectSet()
    subject_set0.links.project = project
    subject_set0.display_name = batch_name0
    print(f'Creating subject set: {batch_name0}')
    subjects = []
    nbr_subjects0 = len(batch0)
    nbr_added = 0
    # note: a batch is now a list of tuples (filename, metadata)
    for image_fn, metadata in batch0:
        nbr_added += 1
        image_path = os.path.join(input_directory0, image_fn)
        print(f'Adding image ({nbr_added}/{nbr_subjects0}) from {image_path}')
        subject = Subject()
        subject.links.project = project
        subject.add_location(image_path)
        subject.metadata.update(metadata)
        subject.save()
        subjects.append(subject)
    subject_set0.save()
    subject_set0.add(subjects)
    return subject_set0, nbr_subjects0


def load_manifest(manifest_path) -> dict:
    print(f'Loading manifest from {manifest_path}')
    with open(manifest_path, 'r') as f:
        manifest0 = json.load(f)
    return manifest0


def create_batches(prefix: str, manifest0: dict) -> dict:
    """
    :param prefix: batch name prefix
    :param manifest0: manifest dict
    :return: batches dict batch_name -> list of tuples (filename, metadata)
    """
    sequence = 0
    vessel_batches = defaultdict(list)
    for filename, metadata in manifest0.items():
        vessel_batches[metadata['vessel']].append((filename, metadata))
    results = defaultdict(list)
    for vessel, big_batch in vessel_batches.items():
        for i in range(0, len(big_batch), MAX_BATCH_SIZE):
            batch0 = big_batch[i: i + MAX_BATCH_SIZE]
            batch_name0 = f'{prefix}_{vessel}_{sequence:05d}'
            sequence += 1
            results[batch_name0] = batch0
    return results


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate Zooniverse manifest")

    oparser.add_argument('-u', dest='user_id',
                         type=str,
                         metavar='USERID',
                         help='Zooniverse userid')

    oparser.add_argument('-p', dest='password',
                         type=str,
                         metavar='PASSWORD',
                         help='Zooniverse password')

    oparser.add_argument('-c', dest='auth_file',
                         type=str,
                         metavar='PATH/zooniverse.json',
                         help='path to JSON Zooniverse credentials file')

    oparser.add_argument('-m', dest='manifest_path',
                         type=str,
                         required=True,
                         metavar='PATH/manifest.json',
                         help='path to JSON manifest file in image directory')

    oparser.add_argument('-s', dest='subject_set_prefix',
                         type=str,
                         required=True,
                         metavar='NAME',
                         help='prefix for subject set name')

    oparser.add_argument('-n', dest='dry_run',
                         default=False,
                         action='store_true',
                         help='do not update Zooniverse')

    options = oparser.parse_args()

    if options.auth_file:
        print(f'Loading {options.auth_file}')
        with open(options.auth_file, 'r') as f:
            auth = json.load(f)
    options.user_id = options.user_id or auth['user_id']
    options.password = options.password or auth['password']

    manifest = load_manifest(options.manifest_path)
    input_directory = os.path.dirname(options.manifest_path)
    nbr_subjects = len(manifest)
    print(f'Number of subjects: {nbr_subjects}')

    Panoptes.connect(username=options.user_id, password=options.password)
    cwb_project = Project.find(PROJECT_ID)
    print('Connection and project OK')

    named_batches = create_batches(options.subject_set_prefix, manifest)
    print(f'Number of batches: {len(named_batches)}')
    subjects_done = 0

    for batch_name, batch in named_batches.items():
        subject_set, nbr_created = create_subject_set(batch_name, batch, input_directory, cwb_project)
        subjects_done += nbr_created
        print(f'Subjects processed: {subjects_done} / {nbr_subjects}')
        if not options.dry_run:
            print(f'Updating Zooniverse project with set {subject_set.display_name}')
            cwb_project.add_subject_sets([subject_set])
            cwb_project.save()
        else:
            print('Not updating Zooniverse project')
