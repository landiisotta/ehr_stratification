#! /bin/zsh

clear
projdir=/home/isotta/data1/ehr-stratification
datadir=$projdir/data

indir=$datadir/ehr-804371-test-2
outdir=/home/riccardo/data1/projects/ehr-stratification/data/dp-test-2-last

test_set=$1

if [ ! -z "$test_set" ]
then
    test_set="--test_set True"
fi

THEANO_FLAGS='device=cpu,floatX=float32' /home/riccardo/software/miniconda3/envs/ehr_stratification/bin/python -u ../src/run_dp.py $indir $outdir $test_set


