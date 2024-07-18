pip install -r requirements.txt
mkdir runs
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init

wandb login

# Start new session to make sure that the new PATH is sourced and run "gsutil init"
# This is necessary because the PATH is not updated in the current shell
gsutil -m rsync -r gs://thesis_cloud_files/fct_checkpoints checkpoints