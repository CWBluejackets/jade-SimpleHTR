#!/usr/bin/env python3
import argparse
import csv
import os

from panoptes_client import Panoptes, Project, SubjectSet, Subject

PROJECT_ID = 18285


def create_subject_set(options) -> SubjectSet:
    input_directory = os.path.dirname(options.manifest_path)
    # TODO
    return None


def load_manifest(manifest_path) -> dict:
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        




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
                         metavar='PATH',
                         help='path to manifest file in image directory')

    oparser.add_argument('-2', dest='workflow_id',
                         type=str,
                         metavar='WORKFLOW',
                         help='workflow ID')

    options = oparser.parse_args()

    Panoptes.connect(username=options.user_id, password=options.password)
    cwb_project = Project.find(PROJECT_ID)

    subject_set = create_subject_set(options)
