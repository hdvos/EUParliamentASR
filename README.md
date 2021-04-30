# Pipeline for Diarization and ASR

Building a pipeline to apply diarization and asr on political wavfiles.

Designed to run on a slurm cluster.

Work in progress ☕

Currently contains 3 steps:

- Diarization: split wavfile according to speaker turns. (every new wavfile contains a single speaker).
- Segmentation: split the diarized wavfiles into wavfiles of length 15-30 seconds for optimized use of Wav2Vec2
- ASR: apply Wav2Vec2