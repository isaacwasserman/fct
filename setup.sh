pip install -r requirements.txt
mkdir runs
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init

wandb login

# source ~/.bashrc

# gsutil cp gs://thesis_cloud_files/Rain13K.hdf5 data/Rain13K.hdf5