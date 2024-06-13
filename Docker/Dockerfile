FROM python:3.10.14-bullseye

LABEL org.opencontainers.image.authors="Marie-Anne André - https://github.com/Nanouna; Théo Rosen - https://github.com/theorosen12; Thomas Manson - https://github.com/dev-mansonthomas"
LABEL org.opencontainers.image.url="https://github.com/Nanouna/recettes-et-sentiments"
LABEL org.opencontainers.image.version=1.0.0
LABEL org.opencontainers.image.vendor="recettes_et_sentiments inc."
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.title="Recettes Et Sentiments"
LABEL org.opencontainers.image.base.name="https://hub.docker.com/layers/library/python/3.10.14-bullseye/images/sha256-4b3ba5fcafd38a22fdd23e6df33d8a1a4aa2971c686b7bcddad080e784266bed?context=explore"

RUN apt-get update     && \
    apt-get upgrade -y && \
    apt-get install -y wget

# RUN adduser --home /rs rs && \
#     chown   -R rs:rs /rs && \
#     chmod   755 /rs

RUN mkdir /rs
RUN pip install wheel         && \
    pip install --upgrade pip


COPY ../requirements.txt /rs/
COPY ../setup.py /rs/
COPY ../recettes_et_sentiments /rs/recettes_et_sentiments
# RUN chown -R rs:rs /rs
WORKDIR /rs

RUN pip install -r requirements.txt && \
    pip install . || { echo 'pip install failed' ; exit 1; }

# RUN make reset_local_files
COPY ../Docker/start.sh /rs/start.sh


# USER rs
ENTRYPOINT ["/rs/start.sh"]
