#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from typing import Tuple
from datetime import datetime

from panoptes_client import Panoptes, Project, SubjectSet, Subject

PROJECT_ID = 18285
MAX_BATCH_SIZE = 1000


def create_subject_set(batch_name0, batch0, input_directory0, project) -> Tuple[SubjectSet, int, dict]:
    subject_set0 = SubjectSet()
    subject_set0.links.project = project
    subject_set0.display_name = batch_name0
    print(f'Creating subject set: {batch_name0}')
    subjects = []
    nbr_subjects0 = len(batch0)
    nbr_added = 0
    updated_metadata_by_id = {}
    # note: a batch is now a list of tuples (filename, metadata)
    for image_fn, metadata in batch0:
        nbr_added += 1
        image_path = os.path.join(input_directory0, image_fn)
        # https://stackoverflow.com/questions/5419389/how-to-overwrite-the-previous-print-to-stdout
        print(f'Adding image ({nbr_added}/{nbr_subjects0}) from {image_path}', end='\r')
        subject = Subject()
        subject.links.project = project
        subject.add_location(image_path)
        subject.metadata.update(metadata)
        subject.save()
        subject_id = subject.id
        subjects.append(subject)
        updated_metadata = metadata.copy()
        # get subject ID and save for later
        updated_metadata['image_path'] = image_path
        updated_metadata['subject_id'] = subject_id
        updated_metadata_by_id[subject_id] = updated_metadata
    print()
    subject_set0.save()
    subject_set0.add(subjects)
    return subject_set0, nbr_subjects0, updated_metadata_by_id


def load_manifest(manifest_path) -> dict:
    print(f'Loading manifest from {manifest_path}')
    with open(manifest_path, 'r') as f:
        manifest0 = json.load(f)
    return manifest0


def make_updated_path(manifest_path: str) -> str:
    stripped = os.path.splitext(manifest_path)[0]
    updated_path = f'{stripped}_updated.json'
    return updated_path


def create_batches(infix: str, manifest0: dict) -> dict:
    """
    :param infix: batch name prefix
    :param manifest0: manifest dict
    :return: batches dict batch_name -> list of tuples (filename, metadata)
    """
    sequence = 0
    vessel_year_batches = defaultdict(list)
    for filename, metadata in manifest0.items():
        vessel = metadata['vessel']
        year = metadata['#muster_year'] or 0
        vessel_year_batches[vessel, year].append((filename, metadata))
    results = defaultdict(list)
    print(f'Number of vessel-year combinations: {len(vessel_year_batches)}')
    for (vessel, year), big_batch in vessel_year_batches.items():
        for i in range(0, len(big_batch), MAX_BATCH_SIZE):
            batch0 = big_batch[i: i + MAX_BATCH_SIZE]
            batch_name0 = f'{vessel} {year:04d} {infix}_{sequence:05d}'
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

    oparser.add_argument('-s', dest='subject_set_infix',
                         type=str,
                         required=True,
                         metavar='NAME',
                         help='infix for subject set name')

    oparser.add_argument('-n', dest='dry_run',
                         default=False,
                         action='store_true',
                         help='do not update Zooniverse')

    options = oparser.parse_args()

    START_TIME = datetime.now()

    # Load authorization file if specified but do NOT overwrite CLI options
    if options.auth_file:
        print(f'Loading {options.auth_file}')
        with open(options.auth_file, 'r') as f:
            auth = json.load(f)
    options.user_id = options.user_id or auth['user_id']
    options.password = options.password or auth['password']

    # It won't work without these
    if not (options.user_id and options.password):
        raise ValueError('Zooniverse userid and password are required')

    manifest = load_manifest(options.manifest_path)
    input_directory = os.path.dirname(options.manifest_path)
    nbr_subjects = len(manifest)
    print(f'Number of subjects: {nbr_subjects}')

    Panoptes.connect(username=options.user_id, password=options.password)
    cwb_project = Project.find(PROJECT_ID)
    print('Connection and project OK')

    named_batches = create_batches(options.subject_set_infix, manifest)
    print(f'Number of batches: {len(named_batches)}')
    subjects_done = 0

    updated_manifest = dict()

    for batch_name, batch in named_batches.items():
        subject_set, nbr_created, updated_metadata_batch = create_subject_set(batch_name, batch,
                                                                              input_directory, cwb_project)
        updated_manifest.update(updated_metadata_batch)
        subjects_done += nbr_created
        print(f'Subjects processed: {subjects_done} / {nbr_subjects}')
        if not options.dry_run:
            print(f'Updating Zooniverse project with set {subject_set.display_name}')
            cwb_project.add_subject_sets([subject_set])
            cwb_project.save()
        else:
            print('Not updating Zooniverse project')
        elapsed_time = datetime.now() - START_TIME
        # drop the microseconds
        elapsed_time_str = str(datetime.now() - START_TIME).split('.')[0]
        seconds_per_subject = elapsed_time.total_seconds() / subjects_done
        print(f'Elapsed time: {elapsed_time_str} ({seconds_per_subject:.2f} seconds/subject)')
        print()

    output_path = make_updated_path(options.manifest_path)
    print(f'Writing {output_path}')
    with open(output_path, 'w') as f:
        json.dump(updated_manifest, f, indent=1, sort_keys=True)

