#!/bin/bash

docker run -i --rm \
    --env-file ./.env \
    -v /home/ubuntu/DATA:/mnt/DATA \
    -v /home/ubuntu/output:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python build-word-embedding.py

