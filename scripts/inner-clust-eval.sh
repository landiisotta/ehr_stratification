#! /bin/zsh

projdir=..
datadir=$projdir/data

indir=$datadir/ehr-804371-test-2

encdir=$indir/encodings

# choose between SNOMED and CCS-SINGLE
disease='T2D'
#disease='PD'
#disease='AD'
#disease='MM'
#disease='BC'
#disease='PC'

../../stratification_ILRM/myvenv/bin/python -u $projdir/src/inner-cl-validation.py $datadir $indir $encdir $disease
