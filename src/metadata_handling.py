import csv
import os.path
import re


WHITESPACE = re.compile(r'\s+')
INTEGER = re.compile(r'\d+')

# columns:
# "Ship name","ship#","volume#","sheet#","Muster Type","Alignment issue?",
#    "Non-standard?","Muster date","Notes","date"


def get_image_filename(muster_info):
    volume = muster_info["volume"]
    sheet = muster_info["sheet"]
    vessel = muster_info["vessel"].lower()
    vessel = WHITESPACE.sub('_', vessel)
    nst = '.NST' if muster_info['non_standard'] else ''
    return f"{volume:03d}_{vessel}_{sheet:04d}{nst}.JPG"


def yes_boolean(yesno: str) -> bool:
    return yesno.strip().lower().startswith('y')


def extract_number(type_name: str) -> str:
    """
    Convert 'Type 3' to the int 3.
    """
    int_str = INTEGER.search(type_name)
    if int_str:
        result = int_str.group(0)
    else:
        result = '0'
    return result


class Metadata:

    def __init__(self, metadata_file):
        self.metadata = dict()
        with open(metadata_file, "r") as f:
            reader = csv.DictReader(f, restval='')
            self.process_reader(reader)
        return

    def process_reader(self, reader: csv.DictReader):
        muster_date = ''
        for row in reader:
            # if the muster date is not listed, use the previous one
            muster_date_specified = row['Muster date'].strip()
            if not muster_date_specified:
                muster_date = muster_date_specified
            bad_alignment = yes_boolean(row['Alignment issue?'])
            non_standard = yes_boolean(row['Non-standard?'])
            type_name = row['Muster Type']
            item = {"vessel": row['Ship name'].strip(),
                    "vessel_no": int(row['ship#']),
                    "volume": int(row['volume#']),
                    "sheet": int(row['sheet#']),
                    "muster_date_specified": muster_date_specified,
                    "muster_date": muster_date,
                    "type_name": type_name,
                    "type": extract_number(type_name),
                    "bad_alignment": bad_alignment,
                    "non_standard": non_standard}
            image_filename = get_image_filename(item)
            item['image_filename'] = image_filename
            item['good'] = not (bad_alignment or non_standard)
            self.metadata[image_filename] = item
        return

    def get_metadata(self, image_path):
        image_file = os.path.basename(image_path)
        return self.metadata.get(image_file, None)

