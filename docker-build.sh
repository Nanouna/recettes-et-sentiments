#!/bin/sh

ENV_FILE="Docker/.env.yaml"
GAR_IMAGE=$(yq e '.GAR_IMAGE' $ENV_FILE)

# build the docker image for local execution
docker build -t $GAR_IMAGE -f Docker/Dockerfile .
