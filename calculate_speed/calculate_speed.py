import os
import re
from csv import DictWriter

from dataclasses import dataclass, asdict

@dataclass
class time_info():
    wav_duration:list
    diarization:list
    segmentation:list
    inference:list
    total_processing_time:list

    diarization_relative:list
    segmentation_relative:list
    inference_relative:list
    total_processing_time_relative:list


filelist = os.listdir("timeregistrations")
filelist = [os.path.join("timeregistrations", filename) for filename in filelist]
print(filelist)

for file in filelist:
    with open(file, 'rt') as f:
        timeregistration_text = f.read()
    try:
        wav_duration = float(re.findall(r"Audiofile is ([0-9]+\.[0-9]+) seconds", timeregistration_text)[0])
        diarization_duration = float(re.findall(r"Stage 1 took ([0-9]+\.[0-9]+) seconds", timeregistration_text)[0])
        segmentation_duration = float(re.findall(r"Stage 2 took ([0-9]+\.[0-9]+) seconds", timeregistration_text)[0])
        inference_duration = float(re.findall(r"Stage 3 took ([0-9]+\.[0-9]+) seconds", timeregistration_text)[0])
    except IndexError:
        continue
    time_data = time_info(wav_duration, diarization_duration, segmentation_duration, inference_duration)
    print(asdict(time_data))

    diarization_duration_relative = diarization_duration/wav_duration
    segmentation_duration_relative = segmentation_duration/wav_duration
    inference_duration_relative = inference_duration/wav_duration