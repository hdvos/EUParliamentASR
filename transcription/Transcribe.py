import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os,sys
from segmentation.Segmentation import LengthSegmentationBookkeep, LengthSegmentationBookkeepSegment
from dataclasses import dataclass, asdict
import json
from csv import writer

@dataclass
class InferenceResultSegment():
    wavfile:str

    absolute_start_seconds:float
    absolute_start_frames:int
    
    absolute_end_seconds:float
    absolute_end_frames:int

    speaker:str

    inferred_text:str

@dataclass
class InferenceResults():
    original_filename:str
    segments:list

# https://www.analyticsvidhya.com/blog/2021/02/hugging-face-introduces-the-first-automatic-speech-recognition-model-wav2vec2/

testfile = "/data/voshpde/audiofiles/20140701.wav"
testfile_resampled = "/data/voshpde/audiofiles/20140701_resampled.wav"


# https://unix.stackexchange.com/questions/274144/sox-convert-the-wav-file-with-required-properties-in-single-command 
os.system(f"sox {testfile} -r 16000 {testfile_resampled}")

#+++++++++
print('+++++++++load tokenizer+++++++++')
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
print('+++++++++load model+++++++++')
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

print('+++++++++save tokenizer+++++++++')
tokenizer.save_pretrained('/tokenizer/')
print('+++++++++save model+++++++++')
model.save_pretrained('/model/')



#load any audio file of your choice


#load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")





with open("test_transcripts.txt", 'wt') as out:
    out.write("/data/voshpde/asr_huggingface/transcriptions.txt")


def get_text(filename:str):
    speech, rate = librosa.load(testfile_resampled, sr=16000)

    input_values = tokenizer(speech, return_tensors = 'pt').input_values
    #Store logits (non-normalized predictions)
    logits = model(input_values).logits

    #Store predicted id's
    predicted_ids = torch.argmax(logits, dim =-1)
    #decode the audio to generate text
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription


def transcribe_wavfile(filename:str):
    if not os.path.exists(filename):
        raise RuntimeError(f"Cannot transcribe {filename}. File does not exist.")

    text = get_text(filename)
    return text


    # print(transcriptions)

def results_to_json(results:list, filename:str):
    if not filename.endswith(".json"):
        raise ValueError(f"Filename must have the .json extension")

    with open(filename, 'wt') as out:
        json.dump(asdict(results), out)


def results_to_csv(results:list, filename:str):
    if not filename.endswith(".csv"):
        raise ValueError(f"Filename must have the .csv extension")

    with open(filename, 'wt') as out:
        mywriter = writer(out, delimiter = '\t')
        mywriter.writerow(["speaker", 'inferred_text', 'duration' , 'absolute_start_seconds', 'absolute_end_seconds', 'segment_wavfile', 'original_wavfile'])
        for result in results
            mywriter.writerow([result.speaker, result.inferred_text, result.absolute_end_seconds-result.absolute_start_seconds , result.absolute_start_seconds, result.absolute_end_seconds, result.wavfile, result.original_wavfile])



def TranscribeFromBookkeep(segmentationbookkeep:LengthSegmentationBookkeep):
    results = InferenceResults(
        original_filename = segmentationbookkeep.original_filename,
        segments = []
    )

    for segment in segmentationbookkeep.segments:
        text = trascibe_wavfile(segment.time_segmented_filename)
        inference_result = InferenceResultSegment(
            wavfile = segment.time_segmented_filename,
            original_wavfile = results.original_filename,
            
            absolute_start_seconds = segment.absolute_start_seconds,
            absolute_start_frames = segment.absolute_start_frames,
            
            absolute_end_seconds = segment.absolute_end_seconds,
            absolute_end_frames = segment.absolute_end_seconds,

            speaker = segment.speaker,
            inferred_text = text

        )
        results.segments.append(inference_result)
