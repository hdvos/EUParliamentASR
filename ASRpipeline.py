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
STAGE = 2   # a parameter that enables skipping a step

SAVE_DIARIZATION_WAVFILES = True
DIARIZATION_WAVFILES_LOCATION = '/home/hugo/MEGA/work/ASR/build_pipeline/diarization/test_wav_out'
if not os.path.exists(DIARIZATION_WAVFILES_LOCATION):
    os.makedirs(DIARIZATION_WAVFILES_LOCATION)

WAVFILE = '/home/hugo/MEGA/work/ASR/build_pipeline/testfile2.wav'


def do_diarization(wavfile_path):
    assert os.path.exists(wavfile_path)
    diarizer = MyDiarizer(wavfile_path)
    diarization_results = diarizer.diarize()
    diarization_bookeep = diarizer.split_wavfile(DIARIZATION_WAVFILES_LOCATION, 'test_bookkeep.json')
    pprint(asdict(diarization_results))
    pprint(asdict(diarization_bookeep))
    return diarization_bookeep

if __name__ == "__main__":
    if STAGE == 1:
        testwav = WAVFILE
        diarization_bookeep = do_diarization(testwav)

    if STAGE <= 2:
        try: 
            diarization_bookkeep
        except NameError