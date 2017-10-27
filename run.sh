#!/bin/bash

docker run -i --rm \
    --env-file ./.env \
    -v /Users/matthewupson/Documents/govuk-word-embedding/:/mnt/DATA \
    -v /Users/matthewupson/Documents/govuk-word-embedding/output:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python create-vocabulary.py

docker run -i --rm \
    --env-file ./.env \
    -v /Users/matthewupson/Documents/govuk-word-embedding/DATA:/mnt/DATA \
    -v /Users/matthewupson/Documents/govuk-word-embedding/output:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python build-word-embedding.py

docker run -i --rm \
    --env-file ./.env \
    -v /Users/matthewupson/Documents/govuk-word-embedding/DATA:/mnt/DATA \
    -v /Users/matthewupson/Documents/govuk-word-embedding/output:/mnt/output \
    ukgovdatascience/govuk-word-embedding:latest python tsne-plot.py

/Users/matthewupson/Documents/govuk-word-embedding
