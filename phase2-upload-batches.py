#!/usr/bin/env python3
import argparse
import json
import os
import traceback
from collections import defaultdict
from typing import Tuple
from datetime import datetime, timedelta

from panoptes_client import Panoptes, Project, SubjectSet, Subject
from panoptes_client.panoptes import PanoptesAPIException

PROJECT_ID = 18285
MAX_BATCH_SIZE = 1000


def create_subject_set(batch_name0, batch0, input_directory0, project) -> Tuple[SubjectSet, int, dict]:
    subject_set = SubjectSet()
    subject_set.links.project = project
    subject_set.display_name = batch_name0
    print(f'Creating subject set: {batch_name0}')
    subjects = []
    nbr_subjects = len(batch0)
    nbr_added = 0
    updated_metadata_by_id = {}
    # note: a batch is now a list of tuples (filename, metadata)
    for image_fn, metadata in batch0:
        nbr_added += 1
        image_path = os.path.join(input_directory0, image_fn)
        # https://stackoverflow.com/questions/5419389/how-to-overwrite-the-previous-print-to-stdout
        print(f'Adding image ({nbr_added}/{nbr_subjects}) from {image_path}', end='\r')
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
    subject_set.save()
    subject_set.add(subjects)
    return subject_set, nbr_subjects, updated_metadata_by_id


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


def upload_batch(connection, directory, batch_name0, file_list):
    subject_set0, nbr_subjects0, updated_metadata_by_id = create_subject_set(batch_name0, file_list,
                                                                             directory, connection)
    print(f'updating with set {subject_set0.display_name}')
    cwb_project.add_subject_sets([subject_set0])
    cwb_project.save()
    updated_manifest0 = {'contents': updated_metadata_by_id,
                         'batch_name': batch_name0}
    return updated_manifest0


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

    nbr_batches = len(options.batch_files)
    print(f'Number of batches: {nbr_batches}')

    Panoptes.connect(username=options.user_id, password=options.password)
    cwb_project = Project.find(PROJECT_ID)
    print('Connection and project OK')

    for i, input_file in enumerate(options.batch_files):
        print(f'Reading {input_file}')
        with open(input_file, 'r') as f:
            batch_manifest = json.load(f)
        manifest_dir = os.path.dirname(input_file)
        output_file = make_updated_path(input_file)
        batch_name = batch_manifest['batch_name']
        contents = batch_manifest['contents']
        try:
            updated_manifest = upload_batch(cwb_project, manifest_dir, batch_name, contents)
            print(f'Writing {output_file}')
            with open(output_file, 'w') as f:
                json.dump(updated_manifest, f, indent=1, sort_keys=True)
        except PanoptesAPIException as e:
            print(f'Failed during upload of batch {batch_name} from {input_file}')
            print(traceback.format_exc())
            print(f'Quitting!')
            break

        elapsed_time = datetime.now() - START_TIME
        elapsed_time_str = str(datetime.now() - START_TIME).split('.')[0]
        seconds_per_batch = elapsed_time.total_seconds() / (i + 1)
        print(f'Elapsed time: {elapsed_time_str} ({seconds_per_batch:.2f} seconds/batch)')
        finish_time = timedelta(seconds=(nbr_batches * elapsed_time.total_seconds() / (i + 1))) + START_TIME
        finish_str = str(finish_time).split('.')[0]
        print(f'Estimated finish time: {finish_str}')
