# govuk-word-embedding

Docker image for running word embeddings of a corpus of html documents (GOV.UK).

This container sets up a python environment for running python scripts. These scripts are stored in this repository in the `govuk-word-embedding` subfolder. It is designed for use with a [databox](https://github.com/ukgovdatascience/govuk-word-embedding).

This docker container can be pulled directly from docker hub with `docker pull ukgovdatascience/govuk-word-embedding`. Whilst this is preferable, note that there is a delay in the automated build pipeline on docker hub, meaning that it may be quicker to build the image on a local or remote machine if you wish changes to be reflected more quickly. This can be done with `docker build -t ukgovdatascience/govuk-word-embedding:latest .`.

## Configuring the docker container

There are a large number of parameters that need to be set when launching the docker container which relate to the location of inputs and outputs, and various parameters for the scripts. These parameters are stored in the `.env` file, and are passed to the `docker run` command with the `--env-file .env` flag.

## Python scripts

|Name|Function|input|output|
|----|--------|-----|------|
|create-vocabulary.py|Builds the vocabulary file from a collection of html files stored in folders.|Raw html files arranged in folders.|vocubulary file|
|build-word-embedding.py|Builds the word embedding from the vocabulary file.|vocabulary file|word_embedding, reverse_dictionary, tensorboard log, saved model|
|tsne-plot.py|Produces tSNE plot using word embedding.|word_embedding, reverse_dictionary|tsne plot|
