#!/bin/sh
#
# configure docker for de
#

ENV_FILE="Docker/.env.yaml"
GCP_REGION=$(yq e '.GCP_REGION' $ENV_FILE)

#configure Docker pour utiliser l'authentification gcloud avec Google Cloud Platform
echo gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev


DOCKER_REPO_NAME=recettes-et-sentiments
# echo the next command
echo gcloud artifacts repositories create $DOCKER_REPO_NAME \
  --repository-format=docker \
  --location=$GCP_REGION  \
  --description="Docker images in GAR for recettes-et-sentiments"

# create recettes-et-sentiments repository in Google Artifacts
gcloud artifacts repositories create $DOCKER_REPO_NAME \
  --repository-format=docker \
  --location=$GCP_REGION  \
  --description="Docker images in GAR for recettes-et-sentiments"
