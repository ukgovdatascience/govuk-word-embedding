#!/bin/bash

docker run -i --rm \
    --env-file ./.env \
    -v /data:/mnt/DATA \
    -v /data/output:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python create-vocabulary.py

docker run -i --rm \
    --env-file ./.env \
    -v /data:/mnt/DATA \
    -v /data/output:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python build-word-embedding.py

docker run -i --rm \
    --env-file ./.env \
    -v /data:/mnt/DATA \
    -v /data/output:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python tsne-plot.py

