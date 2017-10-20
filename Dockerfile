FROM python:3.6

MAINTAINER Matthew Upson
LABEL date="2017-10-20"
LABEL version="0.1.0"
LABEL description="Build a word embedding of gov.uk using tensorflow"

# Update server and install git 

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y git

COPY ./govuk-word-embedding govuk-word-embedding

# Set working directory

WORKDIR /govuk-word-embedding

# Install python requirements

RUN pip install -r requirements.txt

#RUN ./run.sh

# ENTRYPOINT ["python"]

# List Arguments for compilation (might be better as a script)

# CMD ["import", "--experiment", "early_years", "input/early-years.csv"]

