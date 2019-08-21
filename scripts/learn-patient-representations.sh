#! /bin/zsh

clear
projdir=..
indir=$projdir/data

outdir=ehr-804371-test-2

gpu=0

emb_file=$indir/embeddings/word2vec-pheno-embedding-100.emb

test_set=$1

if [ ! -z "$test_set" ]
then
    test_set="--test_set $test_set"
fi

sampling=$2

if [ ! -z "$sampling"]
then
    sampling="-s $sampling"
fi

CUDA_VISIBLE_DEVICES=$gpu ../../stratification_ILRM/myvenv/bin/python -u $projdir/src/patient_representations.py $indir $outdir $test_set $sampling -e $emb_file
