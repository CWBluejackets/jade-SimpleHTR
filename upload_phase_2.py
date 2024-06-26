#!/usr/bin/env python3
import argparse
import os

from panoptes_client import Panoptes, Project, SubjectSet, Subject




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
                         help='path to manifest in image directory')

    options = oparser.parse_args()

    input_directory = os.path.dirname(options.manifest_path)
