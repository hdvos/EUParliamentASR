from diarization.Diarization2 import MyDiarizer #, DiarizationBookkeep, DiarizationBookkeepSegment
from segmentation.audio_segmenter_3 import segment_wavfile
from transcription import Transcribe2
from librosa import get_duration
from pprint import pprint
from dataclasses import dataclass, asdict
import os
from shutil import rmtree
import time
import json

def prepare_output_folder(foldername:str):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    else:
        rmtree(foldername)


# def my_get_duration(wavname):
#     y, sr = librosa.load(librosa.ex('trumpet'))

import argparse

parser = argparse.ArgumentParser(description='Pre-process and transcribe audio files.')
parser.add_argument("wavfile", help="the wavfile that needs to be processed.")
parser.add_argument('-rttm_output', type=str, default=None, 
                    help="Where to store the diarization bookkeep file when it is created, or where to read it from if stage >=2")                                        
parser.add_argument("-segmentation_output_folder", type=str, default="segmentation_wavfiles",
                    help="the folder where the results of segmentation will be stored.")
parser.add_argument("-segmentation_bookkeep_file", type=str, default=None,
                    help="The json file with the segmentation bookkeep.")
parser.add_argument("-asr_model", type=str, default="facebook/wav2vec2-large-100k-voxpopuli",
                    help="The asr model.")
parser.add_argument("-kenlm_model", type=str, default="facebook/wav2vec2-large-100k-voxpopuli",
                    help="The asr model.")

args = parser.parse_args()
# print(args.accumulate(args.integers))


process_id = os.path.splitext(os.path.basename(args.wavfile))[0]

SEGMENTATION_OUTPUT_FOLDER = args.segmentation_output_folder
WAVFILE = args.wavfile



# Params #TODO: add as cmd line parameters
# STAGE = args.stage   # a parameter that enables skipping a step

# # SAVE_DIARIZATION_WAVFILES = args.save_diarization_wav
# # DIARIZATION_WAVFILES_LOCATION = 'diarization/test_wav_out'
# if not os.path.exists(args.diarization_wavfiles_location):
#     os.makedirs(args.diarization_wavfiles_location)

# DIARIZATION_WAVFILES_LOCATION = args.diarization_wavfiles_location


# DIARIZATION_BOOKKEEP_FILE = "test_diarization_bookkeep.json"

# if not args.diarization_bookkeep_file:
#     if args.stage > 1:
#         raise RuntimeError("Must provide a diarization bookkeepfile if stage is larger than 1")
#     if not os.path.exists("diarization_bookkeep_files"):
#         os.makedirs("diarization_bookkeep_files")
#     diarization_filename = f"{process_id}_diarization_bookkeep.json"

#     DIARIZATION_BOOKKEEP_FILE = os.path.join("diarization_bookkeep_files", diarization_filename)
# else:
#     DIARIZATION_BOOKKEEP_FILE = os.path.join("diarization_bookkeep_files", args.diarization_bookkeep_file)
# WAVFILE = args.wavfile



# if not args.segmentation_bookkeep_file:
#     if args.stage > 2:
#         raise RuntimeError("Must provide a segmentation bookkeepfile if stage is larger than 2")
#     if not os.path.exists("segmentation_bookkeep_files"):
#         os.makedirs("segmentation_bookkeep_files")
#     SEGMENTATION_BOOKKEEP_FILE = os.path.join("segmentation_bookkeep_files", f"{process_id}_segmentation_bookkeep.json")
# else: 
#     SEGMENTATION_BOOKKEEP_FILE = os.path.join("segmentation_bookkeep_files", args.segmentation_bookkeep_file)



def do_diarization(wavfile_path):
    assert os.path.exists(wavfile_path)
    diarizer = MyDiarizer(wavfile_path)
    diarization_results = diarizer.diarize()
    # diarization_bookeep = diarizer.split_wavfile(DIARIZATION_WAVFILES_LOCATION, DIARIZATION_BOOKKEEP_FILE)
    pprint(asdict(diarization_results))
    # pprint(asdict(diarization_bookeep))
    return diarization_results

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


def do_segmentation(diarization_bookkeep, output_root:str, bookkeep_json_file:str):
    segmentationbookkeep = SegmentTurnsFromBookkeep(diarization_bookkeep, output_root, bookkeep_json_file)
    return segmentationbookkeep


# TODO: timeregistration dataclass
@dataclass
class time_registration:
    audio_duration:float
    diarization_duration:float
    segmentation_duration:float
    recognition_duration:float

if __name__ == "__main__":

    # timereg.write(f"Audiofile is {get_duration(filename = WAVFILE)} seconds.\n")
    diarization_start=time.time()
    testwav = WAVFILE
    diarization_results = do_diarization(testwav)
    diarization_duration = time.time() - diarization_start
    
    print(diarization_results)
    # Diarization results: rttm filename, time took


    # # timereg.write(f"Stage 1 took {time.time() - stage_1_start:.2f} seconds.\n")

    segments_metadata, segmentationbookkeep = segment_wavfile(testwav, SEGMENTATION_OUTPUT_FOLDER)
    pprint(segmentationbookkeep)
    
    # # timereg.write(f"Stage 2 took {time.time() - stage_2_start:.2f} seconds.\n")

    # stage_3_start = time.time()

    # try: 
    #     segmentationbookkeep
    # except NameError:
    #     raise NotImplementedError("Implement reader")
    Transcribe2.MODEL_NAME = args.asr_model
    print(Transcribe2.MODEL_NAME)
    Transcribe2.TranscribeFromBookkeep(segmentationbookkeep, json_folder='json_test', csv_folder='csv_test')
    # Transcribe.TranscribeFromBookkeep(segmentationbookkeep, 'voxpopuli_transcriptions_json', 'transcriptions_csv', "s")
    # # TODO Also implement reader for segmentation
    # # timereg.write(f"Stage 3 took {time.time() - stage_3_start:.2f} seconds.\n")

    # # TODO: recombine