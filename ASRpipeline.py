from diarization.Diarization import MyDiarizer
from pprint import pprint
from dataclasses import dataclass, asdict
import os
from shutil import rmtree

def prepare_output_folder(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    else:
        rmtree(foldername)


# Params #TODO: add as cmd line parameters
STAGE = 1   # a parameter that enables skipping a step

SAVE_DIARIZATION_WAVFILES = True
DIARIZATION_WAVFILES_LOCATION = 'diarization_wavfiles'
if not os.path.exists(DIARIZATION_WAVFILES_LOCATION):
    os.makedirs(DIARIZATION_WAVFILES_LOCATION)

WAVFILE = '/home/hugo/MEGA/work/ASR/build_pipeline/testfile2.wav'


if __name__ == "__main__":
    if STAGE <= 1:
        testwav = WAVFILE
        assert os.path.exists(testwav)
        diarizer = MyDiarizer(testwav)
        diarization_results = diarizer.diarize()
        diarizer.split_wavfile('/home/hugo/MEGA/work/ASR/build_pipeline/diarization/test_wav_out', 'test_bookkeep.json')
        pprint(asdict(diarization_results))