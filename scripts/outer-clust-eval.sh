#! /bin/zsh

projdir=..
datadir=$projdir/data

indir=$datadir/ehr-804371-test-2

encdir=$indir/encodings

# choose between SNOMED and CCS-SINGLE
code='snomed'

../../stratification_ILRM/myvenv/bin/python -u $projdir/src/clustering-validation.py $datadir $indir $encdir $code
