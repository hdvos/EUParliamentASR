#!/bin/bash

# reset everything
#SBATCH --export=NONE

# number of nodes
#SBATCH --nodes=1# 

number of tasks per node
#SBATCH --ntasks=1

# time you think your job will take (process will end after this time!)
#SBATCH --time=1:00:00

# number of gpus you want to run your job on
#SBATCH --gres=gpu:1

# memory needed
#SBATCH --mem=30G 

# which partition you want to send the job to (view partitions with 'sinfo')
#SBATCH --partition=testing

# email adress to contact with notifications
#SBATCH --mail-user=h.p.de.vos@fgga.leidenuniv.nl

# when to send notifications <BEGIN|END|FAIL|REQUEUE|ALL>
#SBATCH --mail-type="ALL"

# Load modules

#Pytorch

#Python 3.7+


# cd to wd
cd /home/voshpde/eerste_test


# run python
python test_pyannote_1.py
