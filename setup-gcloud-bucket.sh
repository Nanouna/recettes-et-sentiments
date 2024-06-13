#!/bin/bash
ENV_FILE="Docker/.env.yaml"
BUCKET_URL=$(yq e '.BUCKET_URL' $ENV_FILE)
GCP_REGION=$(yq e '.GCP_REGION' $ENV_FILE)

BUCKET_NAME=$(echo $BUCKET_URL | awk -F/ '{print $4}')

gcloud storage buckets create gs://$BUCKET_NAME --location=$GCP_REGION --uniform-bucket-level-access

gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME --member="allUsers" --role="roles/storage.objectViewer"


# which is equivalent to :
# - create a google cloud bucket
#   - give it a name (it must be globally unique, so recettes-et-sentiments won't work)
#   - Data location : Region level, Using the same region as Docker/.env.yaml:GCP_REGION
#   - storage class default
#   - permissions :
#     - uncheck "Enforce public access prevention on this bucket'
#     - Uniform permissions
#   - data protection : leave defaults
#   - once created allow public access by :
#     - Go to the permission tab
#     - Click Grant Access
#     - in the "New principals" field, type "allUsers"
#     - in the "Role" field, type "Storage Object Viewer"
#     - Save
