#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict, Counter
from datetime import datetime


class CustomJsonEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    # custom encoder to get numpy types through
    def default(self, obj):
        if isinstance(obj, datetime):
            return datetime.strftime(obj, '%Y-%m-%d %H:%M:%S')
        else:
            return super(CustomJsonEncoder, self).default(obj)


def process_zoon_row(csv_row):
    """
    Turn the eccentric CSV with embedded JSON strings into one big JSON.
    """
    output = {}
    for k, v in csv_row.items():
        if k in ['metadata', 'subject_data']:
            output[k] = json.loads(v)
        elif k == 'annotations':
            output[k] = process_zoon_annotations(v)
        else:
            output[k] = v
    finish_time = output['metadata']['finished_at'].split('.')[0]
    output['timestamp'] = datetime.strptime(finish_time, '%Y-%m-%dT%H:%M:%S')
    # '2024-01-17T12:53:54.831Z'
    return output


def process_zoon_annotations(annotation_list_json):
    """
    Turn the annotation format right-side-out.
    """
    annotation_list = json.loads(annotation_list_json)
    output = {}
    for annotation1 in annotation_list:
        task = annotation1['task']
        if task in ['T0', 'T1']:
            output[annotation1['task']] = annotation1['value']
    return output


def reconcile_annotations(classifications1):
    """
    @return was_correct, error, voted_value, list of classifications
    """
    task_values = defaultdict(list)
    for classification1 in classifications1:
        for k, v in classification1['annotations'].items():
            task_values[k].append(v)
    t0vote = vote(task_values['T0'])
    if t0vote == 'Yes':
        result = True, False, None, task_values
    elif t0vote == 'No':
        result = False, False, vote(task_values['T1']), task_values
    else:
        # could be box alignment or other errors
        result = False, True, None, task_values
    return result


def vote(things):
    if len(things) == 0:
        result = None
    elif len(things) == 1:
        result = things[0]
    else:
        c = Counter(things)
        result = c.most_common(1)[0][0]
    # crude, doesn't handle ties
    return result


def extra_filename(fn):
    """
    Stick '_debug' in the filename.
    """
    dir = os.path.dirname(fn)
    base, suffix = os.path.splitext(os.path.basename(fn))
    return os.path.join(dir, f'{base}_debug{suffix}')


oparser = argparse.ArgumentParser(description="analyse the annotations processed from the exports",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-i', dest='input_file',
                     type=str,
                     required=True,
                     metavar='INPUT.csv',
                     help='input file')

oparser.add_argument('-o', dest='output_file',
                     type=str,
                     required=True,
                     metavar='OUTPUT.json',
                     help='output file')

options = oparser.parse_args()

debug_file = extra_filename(options.output_file)

input_data = []
print(f'Reading {options.input_file}')
with open(options.input_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        input_data.append(process_zoon_row(row))

print(f'Classifications: {len(input_data)}')

data_by_subject = defaultdict(list)
for item in input_data:
    data_by_subject[item['subject_ids']].append(item)

full_output = {}
corrections = {}
print(f'Subjects: {len(data_by_subject)}')

for subject, classifications in data_by_subject.items():
    # TODO filter out stuff like [unclear]
    was_correct, error, correction, task_summary = reconcile_annotations(classifications)
    full_output[subject] = {'was_correct': was_correct,
                            'error': error,
                            'correction': correction,
                            'summary': task_summary,
                            'classifications': classifications}
    if not (was_correct or error):
        corrections[subject] = correction

print(f'Subjects for correction: {len(corrections)}')

print(f'Writing {options.output_file}')
with open(options.output_file, 'w') as f:
    json.dump(corrections, f, indent=1, cls=CustomJsonEncoder)

print(f'Writing {debug_file}')
with open(debug_file, 'w') as f:
    json.dump(full_output, f, indent=1, cls=CustomJsonEncoder)
