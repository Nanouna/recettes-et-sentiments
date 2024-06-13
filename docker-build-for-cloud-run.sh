#!/bin/sh

# prerequisite :
# * Cloud Run API is enabled
# * Artifact Registry is enabled
# * install yq (snap install yq for linux, brew install yq) ==> yq (https://github.com/mikefarah/yq/) version v4.44.1
# * create  a Docker/.env.yaml file and fill it wit correct values
# * execute this setup script once : setup-gcloud-artifcat-registry.sh

# Lire les variables d'environnement depuis le fichier YAML
ENV_FILE="Docker/.env.yaml"

GCP_REGION=$(yq e '.GCP_REGION' $ENV_FILE)
GCP_PROJECT_ID=$(yq e '.GCP_PROJECT_ID' $ENV_FILE)
DOCKER_REPO_NAME=$(yq e '.DOCKER_REPO_NAME' $ENV_FILE)
GAR_IMAGE=$(yq e '.GAR_IMAGE' $ENV_FILE)
GAR_IMAGE_VERSION=$(yq e '.GAR_IMAGE_VERSION' $ENV_FILE)

# build the image for Google Cloud Run
echo docker build -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$GAR_IMAGE:$GAR_IMAGE_VERSION -f Docker/Dockerfile .
docker build -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$GAR_IMAGE:$GAR_IMAGE_VERSION -f Docker/Dockerfile .

# push it to Google Artifact registry
docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$GAR_IMAGE:$GAR_IMAGE_VERSION
