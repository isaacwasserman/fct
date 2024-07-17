# Use gsutil to sync the local directory "checkpoints" to the GCS bucket (i.e. upload the folder to GCS)
gsutil -m rsync -r checkpoints gs://thesis_cloud_files/fct_checkpoints

git add .
git config --global user.email "isaacrwasserman@gmail.com"
git config --global user.name "Isaac Wasserman"
git commit -m "Logging out"
git push origin main