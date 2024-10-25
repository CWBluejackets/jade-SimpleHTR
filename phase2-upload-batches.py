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
        updated_metadata['subject_id'] = int_subject_id(subject_id)
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


def int_subject_id(subject_id):
    try:
        result = int(subject_id)
    except ValueError:
        print(f"Not convertible to int: {subject_id}")
        result = subject_id
    return result


def make_updated_path(manifest_path: str) -> str:
    stripped = os.path.splitext(manifest_path)[0]
    updated_path = f'{stripped}_updated.json'
    return updated_path


def upload_batches(batches):
    subjects_done = 0
    manifest_with_ids = dict()
    for batch_name, batch in batches.items():
        subject_set, nbr_created, updated_metadata_batch = create_subject_set(batch_name, batch,
                                                                              input_directory, cwb_project)
        manifest_with_ids.update(updated_metadata_batch)
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
        finish_time = timedelta(seconds=(nbr_subjects * elapsed_time.total_seconds() / subjects_done)) + START_TIME
        finish_str = str(finish_time).split('.')[0]
        print(f'Estimated finish time: {finish_str}')
        print()
    return manifest_with_ids


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

    oparser.add_argument('batch_files', metavar='BATCH_01234.json',
                         nargs='+',
                         help='batch files')

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

    print(f'Number of batchs: {len(options.batch_files)}')

    Panoptes.connect(username=options.user_id, password=options.password)
    cwb_project = Project.find(PROJECT_ID)
    print('Connection and project OK')

    for input_file in options.batch_files:
        pass



    updated_manifest = upload_batches(named_batches)

    output_path = make_updated_path(options.manifest_path)
    print(f'Writing {output_path}')
    with open(output_path, 'w') as f:
        json.dump(updated_manifest, f, indent=1, sort_keys=True)

