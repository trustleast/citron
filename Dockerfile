FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workdir

RUN pip3 install -U pip setuptools wheel fastapi
RUN pip3 install -U 'spacy[cuda11x]'
RUN python -m spacy download en_core_web_trf

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -m nltk.downloader names

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/workdir"

CMD  ["fastapi", "run", "server.py", "--host", "0.0.0.0", "--port", "8080"]