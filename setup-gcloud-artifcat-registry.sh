#!/bin/sh

GCP_REGION=$(yq e '.GCP_REGION' $ENV_FILE)
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev

DOCKER_REPO_NAME=recettes-et-sentiments
gcloud artifacts repositories create $DOCKER_REPO_NAME \
  --repository-format=docker \
  --location=$GCP_REGION  \
  --description="Docker images in GAR for recettes-et-sentiments"
