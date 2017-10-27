#!/bin/bash
ENV=./.env

docker run -i --rm \
    --env-file $ENV \
    -v $DATA_DIR:/mnt/DATA \
    -v $OUT_DIR:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python create-vocabulary.py

docker run -i --rm \
    --env-file $ENV \
    -v $DATA_DIR:/mnt/DATA \
    -v $OUT_DIR:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python build-word-embedding.py

docker run -i --rm \
    --env-file $ENV \
    -v $DATA_DIR:/mnt/DATA \
    -v $OUT_DIR:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python tsne-plot.py

