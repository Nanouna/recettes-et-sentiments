#!/bin/sh

# Lire les variables d'environnement depuis le fichier YAML
ENV_FILE="Docker/.env.yaml"

BUCKET_URL=$(yq e '.BUCKET_URL' $ENV_FILE)
FILE_NAMES=$(yq e '.FILE_NAMES' $ENV_FILE)
GAR_IMAGE=$(yq e '.GAR_IMAGE' $ENV_FILE)
GAR_MEMORY=$(yq e '.GAR_MEMORY' $ENV_FILE)
PORT=$(yq e '.PORT' $ENV_FILE)

# Exécuter la commande docker run avec les variables d'environnement
docker run \
    -e BUCKET_URL="$BUCKET_URL" \
    -e FILE_NAMES="$FILE_NAMES" \
    -e PORT="8080" \
    -p 8080:8080 \
    $GAR_IMAGE
