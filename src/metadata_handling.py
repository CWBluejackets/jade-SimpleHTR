import csv
import os.path
import re

WHITESPACE = re.compile(r'\s+')
INTEGER = re.compile(r'\d+')
DOUBLE_INITIAL = re.compile(r'\b([a-z])\s*([a-z])\b')
INDEX = re.compile(r'(\d\d\d)_(\S+)_(\d\d\d\d)')
NON_LETTER = re.compile(r'[^a-z]+')

# Vessels misspelled in the Master Sheet Entry Log pulldown menu:
VESSEL_CORRECTIONS = {'adriondack': 'adirondack',
                      'agamenticus': 'agamanticus',
                      'cincinatti': 'cincinnati',
                      'henryjames': 'henryjanes',
                      'hollycock': 'hollyhock',
                      'howquait': 'howquah',
                      'lackwanna': 'lackawanna',
                      'farallones': 'farrallones'
                      }

# columns:
# "Ship name","ship#","volume#","sheet#","Muster Type","Alignment issue?",
#    "Non-standard?","Muster date","Notes","date"


def normalize_vessel(raw_vessel: str) -> str:
    vessel0 = raw_vessel.strip().lower()
    vessel1 = VESSEL_CORRECTIONS.get(vessel0, vessel0)
    vessel2 = NON_LETTER.sub('', vessel1)
    return vessel2


def get_image_filename(muster_info):
    volume = muster_info["volume"]
    sheet = muster_info["sheet"]
    vessel = muster_info["vessel"]
    nst = '.NST' if muster_info['non_standard'] else ''
    return f"{volume:03d}_{vessel}_{sheet:04d}{nst}.JPG"


def make_index_key_from_metadata(muster_info) -> str:
    """
    Key to match part of filename, e.g., '117_essex_0058'
    """
    volume = muster_info["volume"]
    sheet = muster_info["sheet"]
    # this is the normalized vessel name
    vessel = muster_info["vessel"]
    return f"{volume:03d}_{vessel}_{sheet:04d}"


def make_index_key_from_pathname(path: str) -> str:
    filename = os.path.basename(path)
    # 117_essex_0058.jpg
    # 120_estrella_0005.JPG
    # 114_e_b_hale_0023.JPG -> 114_ebhale_0025
    index_match = INDEX.match(filename)
    if index_match:
        volume = index_match.group(1)
        vessel = normalize_vessel(index_match.group(2))
        sheet = index_match.group(3)
        result = f'{volume}_{vessel}_{sheet}'
    else:
        result = None
    return result


def yes_boolean(yesno: str) -> bool:
    return yesno.strip().lower().startswith('y')


def extract_form_type(type_name: str) -> str:
    """
    Convert 'Type 3' to '3'.
    """
    int_str = INTEGER.search(type_name)
    if int_str:
        result = int_str.group(0)
    else:
        result = '0'
    return result


def force_to_int(str_containing_int: str) -> int:
    try:
        result = int(str_containing_int)
    except ValueError:
        match = INTEGER.search(str_containing_int)
        if match:
            result = int(match.group(0))
        else:
            result = -1
        print(f'Invalid int: {str_containing_int} -> {result}')
    return result


class Metadata:

    def __init__(self, metadata_file, inventory_file):
        self.metadata = dict()
        self.inventory = dict()
        self.vessel_mismatches = set()
        with open(inventory_file, "r", encoding="utf-8-sig") as f:
            print(f'Loading inventory from {inventory_file}')
            reader = csv.DictReader(f, restval='')
            self.process_inventory(reader)
        with open(metadata_file, "r", encoding="utf-8-sig") as f:
            print(f'Loading metadata from {metadata_file}')
            reader = csv.DictReader(f, restval='')
            self.process_metadata(reader)
        for vessel in sorted(self.vessel_mismatches):
            print(f'Mismatched vessel: {vessel}')
        print(f'Loaded inventory ({len(self.inventory)}) and metadata ({len(self.metadata)})')
        return

    def process_metadata(self, reader: csv.DictReader):
        muster_date_running = ''
        for row in reader:
            # if the muster date is not listed, use the previous one
            muster_date_specified = row['Muster date'].strip()
            if muster_date_specified:
                muster_date_running = muster_date_specified
            bad_alignment = yes_boolean(row['Alignment issue?'])
            # was 'Non-standard?'
            try:
                non_standard = yes_boolean(row['Non Type 2/3'])
            except KeyError:
                non_standard = yes_boolean(row['Non-standard?'])
            type_name = row['Muster Type']
            raw_vessel = row['Ship name']
            vessel = normalize_vessel(raw_vessel)
            volume = force_to_int(row['volume#'])
            nara_url = self.get_nara_url(vessel, volume)
            item = {"vessel_raw": raw_vessel,
                    "vessel": vessel,
                    "vessel_ui": raw_vessel.title(),
                    "vessel_no": int(row['ship#']),
                    "volume": volume,
                    "sheet": int(row['sheet#']),
                    "muster_date_specified": muster_date_specified,
                    "muster_date": muster_date_specified or muster_date_running,
                    "type_name": type_name,
                    "type": extract_form_type(type_name),
                    "bad_alignment": bad_alignment,
                    "non_standard": non_standard,
                    "nara_url": nara_url
                    }
            image_filename = get_image_filename(item)
            key = make_index_key_from_metadata(item)
            item['image_filename'] = image_filename
            item['good'] = not (bad_alignment or non_standard)
            self.metadata[key] = item
        return

    def process_inventory(self, reader: csv.DictReader):
        """
        Build the database of NARA URLs (for Zooniverse metadata)
        """
        for row in reader:
            vessel = normalize_vessel(row['Ship Name'])
            # Wabash 519A
            volume = force_to_int(row['Volume'])
            nara_url = row['NARA URL']
            self.inventory[(vessel, volume)] = nara_url
        return

    def get_metadata(self, index_key):
        return self.metadata.get(index_key, None)

    def get_nara_url(self, vessel: str, volume: int) -> str:
        try:
            result = self.inventory[(vessel, volume)]
        except KeyError:
            self.vessel_mismatches.add(vessel)
            result = '_UNKNOWN_'
        return result
