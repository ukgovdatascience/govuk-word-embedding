#!/bin/bash
ENV=./.env
DATA=$DATA_DIR
OUTPUT=$OUT_DIR

docker run -i --rm \
    --env-file $ENV \
    -v $DATA:/mnt/DATA \
    -v $OUTPUT:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python create-vocabulary.py

docker run -i $ENV \
    --env-file ./.env \
    -v $DATA:/mnt/DATA \
    -v $OUTPUT:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python build-word-embedding.py

docker run -i --rm \
    --env-file $ENV \
    -v $DATA:/mnt/DATA \
    -v $OUTPUT:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python tsne-plot.py

