from pprint import pprint
import os

SLURM_FOLDER = "all_slurm_voxpopuli"

if not os.path.exists(SLURM_FOLDER):
    os.makedirs(SLURM_FOLDER)

COMMANDS_PER_ROUND = 4
ROUNDS_PER_SLURMSCRIPT = 5

with open('filelist_clean.txt', 'rt') as f:
    filelist= f.readlines()

filelist = [filenm.strip() for filenm in filelist]
# pprint(filelist)
# print(len(filelist))

roundcount = 0
commandcount = 0
filecount = 0

def newslurm():
    return f"""#!/bin/bash


# reset everything
#SBATCH --export=NONE

# number of nodes
#SBATCH --nodes=1# 

# number of tasks per node
#SBATCH --ntasks={COMMANDS_PER_ROUND}

# time you think your job will take (process will end after this time!)
#SBATCH --time=24:00:00

# number of gpus you want to run your job on
#SBATCH --gres=gpu:{COMMANDS_PER_ROUND}

# memory needed
#SBATCH --mem=240G 

# which partition you want to send the job to (view partitions with 'sinfo')
#SBATCH --partition=gpu-medium

# email adress to contact with notifications
#SBATCH --mail-user=h.p.de.vos@fgga.leidenuniv.nl

# when to send notifications <BEGIN|END|FAIL|REQUEUE|ALL>
#SBATCH --mail-type="ALL"

# Load modules

#Pytorch

#Python 3.7+


echo "============LOAD TensorFlow============"
#module load Miniconda3/4.7.10
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
echo "============LOAD PyTorch============"
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4

# cd to wd
cd /data/voshpde/pipeLine

# Install
pip install pyannote.core --user
pip install pyannote.audio==1.1.1 --user
pip install pydub
pip install librosa --user
pip install transformers --user


# run    
    
"""

current_slurmscript = newslurm()

while True:
    try:
        filename = filelist.pop()
        print(filename)
    except IndexError:
        current_slurmscript += "wait\n\n"
        with open(f"{SLURM_FOLDER}/inference_{filecount}.slurm", 'wt') as out:
            out.write(current_slurmscript)
        print("done")
        break
    
    current_slurmscript += f"python ASRpipeline_voxpopuli.py {filename}  &\n"
    commandcount += 1

    if commandcount == COMMANDS_PER_ROUND:
        current_slurmscript += "wait\n\n"
        commandcount = 0
        roundcount += 1
    if roundcount == ROUNDS_PER_SLURMSCRIPT:
        with open(f"{SLURM_FOLDER}/inference_{filecount}.slurm", 'wt') as out:
            out.write(current_slurmscript)
        roundcount = 0
        filecount += 1

        current_slurmscript = newslurm()



