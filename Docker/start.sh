#!/bin/sh

# Vérifier si les variables d'environnement sont définies
if [ -z "$BUCKET_URL" ] || [ -z "$FILE_NAMES" ]; then
    echo "Les variables d'environnement BUCKET_URL et FILE_NAMES doivent être définies."
    exit 1
fi

IFS=';'
BASE_FOLDER=/tmp/data/

mkdir ${BASE_FOLDER}

for FILE in $FILE_NAMES; do
    # URL complète du fichier
    FILE_URL="${BUCKET_URL}/${FILE}"

    # Destination où le fichier sera téléchargé
    DESTINATION="${BASE_FOLDER}/${FILE}"

    # Télécharger le fichier en utilisant wget
    wget -q -O "$DESTINATION" "$FILE_URL"

    # Vérifier si le fichier a été téléchargé avec succès
    if [ -f "$DESTINATION" ]; then
        echo "Fichier $FILE téléchargé avec succès."
    else
        echo "Échec du téléchargement du fichier $FILE."
        exit 1
    fi
done

# echo "uncompressing .tar.bz2 files"
# cd ${BASE_FOLDER}
# cat *.tar.bz2 | tar -ixjv
# cd -
# echo "uncompressing .tar.bz2 files - DONE"

ls -la $BASE_FOLDER

cd /rs/
exec uvicorn recettes_et_sentiments.api.fast:app --host 0.0.0.0 --port $PORT
