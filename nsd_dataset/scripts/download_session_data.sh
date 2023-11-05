#!/bin/bash

for subj in 1 2 5 7; do
    aws --no-sign-request s3 sync --exclude '*' \
        --include 'func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session*.*' \
        s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0"$subj" \
        ./subj0"$subj"

    aws --no-sign-request s3 sync --exclude '*' \
        --include 'func1pt8mm/roi/nsdgeneral.nii.gz' \
        s3://natural-scenes-dataset/nsddata/ppdata/subj0"$subj" \
        ./subj0"$subj"
done
