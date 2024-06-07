FROM python:3.10.14-bullseye

LABEL org.opencontainers.image.authors="Marie-Anne André - https://github.com/Nanouna; Théo Rosen - https://github.com/ ?  ;Thomas Manson - https://github.com/dev-mansonthomas"
LABEL org.opencontainers.image.url="https://github.com/Nanouna/recettes-et-sentiments"
LABEL org.opencontainers.image.version=1.0.0
LABEL org.opencontainers.image.vendor="http://solace.com/"
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.title="Recettes Et Sentiments"
LABEL org.opencontainers.image.base.name="https://hub.docker.com/layers/library/python/3.10.14-bullseye/images/sha256-4b3ba5fcafd38a22fdd23e6df33d8a1a4aa2971c686b7bcddad080e784266bed?context=explore"

RUN apt-get update     && \
    apt-get upgrade -y && \
    apt-get install -y wget

RUN adduser --home /rs rs && \
    chown   -R rs:rs /rs && \
    chmod   755 /rs

WORKDIR /rs

COPY ../requirements.txt requirements.txt
RUN pip install --upgrade pip; \
    pip install --prefer-binary --ignore-installed -r requirements.txt

COPY ../recettes_et_sentiments recettes_et_sentiments
COPY ../docker/start.sh start.sh
WORKDIR /rs/recettes_et_sentiments
RUN pip install .

RUN make reset_local_files

USER rs
ENTRYPOINT ["/start.sh"]
