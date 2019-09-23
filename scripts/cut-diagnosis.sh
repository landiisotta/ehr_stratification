#! /bin/zsh

projdir=..
datadir=$projdir/data

testdir=$datadir/ehr-804371-test-2

../../stratification_ILRM/myvenv/bin/python -u $projdir/src/cut-diagnosis.py $datadir $testdir
