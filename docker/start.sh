

# Vérifier si les variables d'environnement sont définies
if [ -z "$BUCKET_URL" ] || [ -z "$FILE_NAMES" ]; then
    echo "Les variables d'environnement BUCKET_URL et FILE_NAMES doivent être définies."
    exit 1
fi

IFS=';'
BASE_FOLDER=/r-s/data/

mkdir ${BASE_FOLDER}

for FILE in $FILE_NAMES; do
    # URL complète du fichier
    FILE_URL="${BUCKET_URL}/${FILE}"

    # Destination où le fichier sera téléchargé
    DESTINATION="/r-s/data/${FILE}"

    # Télécharger le fichier en utilisant wget
    wget -O "$DESTINATION" "$FILE_URL"

    # Vérifier si le fichier a été téléchargé avec succès
    if [ -f "$DESTINATION" ]; then
        echo "Fichier $FILE téléchargé avec succès."
    else
        echo "Échec du téléchargement du fichier $FILE."
        exit 1
    fi
done


cd ${BASE_FOLDER}
cat *.tar.bz2 | tar -ixjv
cd -

exec uvicorn recettes-et-sentiments.api.fast:app --host 0.0.0.0 --port $PORT
