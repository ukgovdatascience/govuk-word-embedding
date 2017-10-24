#!/bin/bash

docker run -i --rm \
    -v /data:/mnt/DATA \
    -v /output:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python build-word-embedding.py

