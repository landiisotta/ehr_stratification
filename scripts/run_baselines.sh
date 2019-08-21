#! /bin/zsh

clear
projdir=..
datadir=$projdir/data

indir=$datadir/ehr-804371-test-2
outdir=$indir/encodings

gpu=1

test_set=$1

if [ ! -z "$test_set" ]
then
    test_set="--test_set $test_set"
fi

../../stratification_ILRM/myvenv/bin/python -u $projdir/src/baselines.py $indir $outdir $test_set
