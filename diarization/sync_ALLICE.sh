rsync -vraz --bwlimit=2500 --exclude 'backsync' --exclude 'sync_ALLICE.sh' ./* hpc2:/data/voshpde/pyannote
rsync -vraz --bwlimit=2500 --exclude '*.wav' hpc2:/data/voshpde/pyannote/* backsync/
