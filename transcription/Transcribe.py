import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os,sys
from segmentation.Segmentation import LengthSegmentationBookkeep, LengthSegmentationBookkeepSegment

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
speech, rate = librosa.load(testfile_resampled, sr=16000)

#load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


input_values = tokenizer(speech, return_tensors = 'pt').input_values
#Store logits (non-normalized predictions)
logits = model(input_values).logits

#Store predicted id's
predicted_ids = torch.argmax(logits, dim =-1)
#decode the audio to generate text
transcriptions = tokenizer.decode(predicted_ids[0])
print(transcriptions)

with open("test_transcripts.txt", 'wt') as out:
    out.write("/data/voshpde/asr_huggingface/transcriptions.txt")


def TranscribeFromBookkeep(segmentationBookkeep):
    ...