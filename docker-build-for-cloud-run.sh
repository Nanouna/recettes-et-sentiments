#!/bin/sh

# Lire les variables d'environnement depuis le fichier YAML
ENV_FILE="Docker/.env.yaml"

GCP_REGION=$(yq e '.GCP_REGION' $ENV_FILE)
GCP_PROJECT_ID=$(yq e '.GCP_PROJECT_ID' $ENV_FILE)
DOCKER_REPO_NAME=$(yq e '.DOCKER_REPO_NAME' $ENV_FILE)
GAR_IMAGE=$(yq e '.GAR_IMAGE' $ENV_FILE)
GAR_IMAGE_VERSION=$(yq e '.GAR_IMAGE_VERSION' $ENV_FILE)

echo docker build -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$GAR_IMAGE:$GAR_IMAGE_VERSION -f Docker/Dockerfile .
docker build -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$GAR_IMAGE:$GAR_IMAGE_VERSION -f Docker/Dockerfile .
