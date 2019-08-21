#! /bin/zsh

projdir=..
datadir=$projdir/data

testdir=$datadir/ehr-804370-test-1

../../stratification_ILRM/myvenv/bin/python -u $projdir/src/cut-diagnosis.py $datadir $testdir
