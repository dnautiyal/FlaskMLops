# FlaskMLops

- docker build -t aerial-detection-webapp .
- docker run -e AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) -e AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key) -p 8000:8000 aerial-detection-webapp


## to stop the container
docker stop <container_name>