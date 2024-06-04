if ! [ -x "$(command -v gsutil)" ]; then
    echo 'Installing Google cloud utilities' >&2
    apt-get update
    apt-get install apt-transport-https ca-certificates gnupg curl
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    apt-get update && apt-get install google-cloud-cli
    gcloud init
fi

gsutil -m rsync -d -r checkpoints gs://thesis_cloud_files/fct_checkpoints

git config --global user.email "isaacrwasserman@gmail.com"
git config --global user.name "Isaac Wasserman"

git add .
echo "Enter commit message: "
read commitMessage
git commit -m "$commitMessage"
git push origin main
