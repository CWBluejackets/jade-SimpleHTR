#!/usr/bin/env python3
import argparse
import csv
import json
import os
import traceback

from panoptes_client import Panoptes, Project, SubjectSet, Subject, Workflow
from panoptes_client.panoptes import PanoptesAPIException

PROJECT_ID = 18285


def create_subject_set(options0, project) -> SubjectSet:
    subject_set0 = SubjectSet()
    subject_set0.links.project = project
    subject_set0.display_name = options0.subject_set_name
    subjects = []
    input_directory = os.path.dirname(options0.manifest_path)
    manifest = load_manifest(options0.manifest_path)
    nbr_subjects = len(manifest)
    nbr_added = 0
    for image_fn, metadata in manifest.items():
        nbr_added += 1
        image_path = os.path.join(input_directory, image_fn)
        print(f'Adding image ({nbr_added}/{nbr_subjects}) from {image_path}')
        subject = Subject()
        subject.links.project = project
        subject.add_location(image_path)
        subject.metadata.update(metadata)
        subject.save()
        subjects.append(subject)
    subject_set0.save()
    subject_set0.add(subjects)
    print('Subject set complete')
    return subject_set0


def load_manifest(manifest_path) -> dict:
    print(f'Loading manifest from {manifest_path}')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    return manifest


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate Zooniverse manifest",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-u', dest='user_id',
                         type=str,
                         metavar='USERID',
                         help='Zooniverse userid')

    oparser.add_argument('-p', dest='password',
                         type=str,
                         metavar='PASSWORD',
                         help='Zooniverse password')

    oparser.add_argument('-m', dest='manifest_path',
                         type=str,
                         metavar='PATH/manifest.json',
                         help='path to JSON manifest file in image directory')

    oparser.add_argument('-w', dest='workflow_id',
                         type=str,
                         required=True,
                         metavar='WORKFLOW',
                         help='workflow ID')

    oparser.add_argument('-s', dest='subject_set_name',
                         type=str,
                         required=True,
                         metavar='NAME',
                         help='subject set name')

    oparser.add_argument('-n', dest='dry_run',
                         default=False,
                         action='store_true',
                         help='do not update Zooniverse')

    options = oparser.parse_args()

    Panoptes.connect(username=options.user_id, password=options.password)
    cwb_project = Project.find(PROJECT_ID)
    print('Connection and project OK')
    workflow = Workflow.find(options.workflow_id)
    subject_set = create_subject_set(options, cwb_project)
    if not options.dry_run:
        print('Updating Zooniverse workflow')
        workflow.links.subject_sets.add(subject_set)
        # DO NOT workflow.save() for some reason
        print('Updating Zooniverse project')
        cwb_project.add_subject_sets([subject_set])
        cwb_project.save()
    else:
        print('Not updating Zooniverse workflow')
