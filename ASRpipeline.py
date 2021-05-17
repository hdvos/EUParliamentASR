from diarization.Diarization import MyDiarizer, DiarizationBookkeep, DiarizationBookkeepSegment
from segmentation.Segmentation import SegmentTurnsFromBookkeep

from pprint import pprint
from dataclasses import dataclass, asdict
import os
from shutil import rmtree

import json

def prepare_output_folder(foldername:str):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    else:
        rmtree(foldername)


# Params #TODO: add as cmd line parameters
STAGE = 2   # a parameter that enables skipping a step

SAVE_DIARIZATION_WAVFILES = True
DIARIZATION_WAVFILES_LOCATION = '/home/hugo/MEGA/work/ASR/build_pipeline/diarization/test_wav_out'
if not os.path.exists(DIARIZATION_WAVFILES_LOCATION):
    os.makedirs(DIARIZATION_WAVFILES_LOCATION)

DIARIZATION_BOOKKEEP_FILE = "test_diarization_bookkeep.json"

WAVFILE = '/home/hugo/MEGA/work/ASR/build_pipeline/testfile2.wav'


def do_diarization(wavfile_path):
    assert os.path.exists(wavfile_path)
    diarizer = MyDiarizer(wavfile_path)
    diarization_results = diarizer.diarize()
    diarization_bookeep = diarizer.split_wavfile(DIARIZATION_WAVFILES_LOCATION, DIARIZATION_BOOKKEEP_FILE)
    pprint(asdict(diarization_results))
    pprint(asdict(diarization_bookeep))
    return diarization_bookeep

def read_diarization_bookkeep(filename:str):
    print('read diarization bookkeep')
    with open(filename, 'rt') as f:
        diarization_bookkeep_json = json.load(f)
    
    diarization_bookeep = DiarizationBookkeep(diarization_bookkeep_json['original_file'], [])
    # diarization_bookeep.original_file = diarization_bookkeep_json['original_file']

    for segment in diarization_bookkeep_json['segments']:
        diarization_bookeep.segments.append(DiarizationBookkeepSegment(
            start=segment['start'],
            end=segment['end'],
            speaker=segment['speaker'],
            turn_i=segment['turn_i'],
            filename=segment['filename']
        ))
    return diarization_bookeep


def do_segmentation(diarization_bookkeep):
    SegmentTurnsFromBookkeep(diarization_bookkeep)


if __name__ == "__main__":
    if STAGE == 1:
        testwav = WAVFILE
        diarization_bookeep = do_diarization(testwav)

    if STAGE <= 2:
        try: 
            diarization_bookkeep
        except NameError:
            diarization_bookkeep=read_diarization_bookkeep(DIARIZATION_BOOKKEEP_FILE)
            print("diarization bookkeep loaded")
    
        do_segmentation(diarization_bookkeep)
        