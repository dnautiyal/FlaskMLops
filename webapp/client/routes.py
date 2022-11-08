
# from .prediction import get_prediction, create_output_image
from flask import Flask, request, render_template, flash, request,redirect, url_for
from flask import current_app as app
import uuid
import boto3
import os
import logging
import requests
import json
from urllib.parse import unquote

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_EXTENSIONS = {'jpg'}
logger = logging.getLogger(__name__)
S3_BUCKET = "aerial-detection-mlops4"
INPUT_S3_KEY =  "inferencing/photos/input"
OUTPUT_S3_IMAGES_KEY =  "inferencing/photos/output/images"
OUTPUT_S3_LABELS_KEY =  "inferencing/photos/output/labels"
tmp_file_folder = "./static/tmp_data"

s3_client = boto3.client('s3')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def retina_ai():
    # Write the GET Method to get the index file
    if request.method == 'GET':
        return render_template('index.html')
    # Write the POST Method to post the results file
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('File Not Uploaded')
            return
        # Read file from upload
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Assign an id to the asynchronous task
            task_id = uuid.uuid4().hex
            new_file_name = f'{task_id}-{file.filename}'
            img = request.files['file']
            if img:
                try:
                    os.makedirs(tmp_file_folder, exist_ok=True) 
                    local_file_name = f'{tmp_file_folder}/{new_file_name}'
                    img.save(local_file_name)
                    s3_client.upload_file(Bucket = S3_BUCKET, Filename = local_file_name, Key = f'{INPUT_S3_KEY}/{new_file_name}')
                    data = {"input_image_file_url": f's3://{S3_BUCKET}/{INPUT_S3_KEY}/{new_file_name}',
                            "output_image_file_url": f's3://{S3_BUCKET}/{OUTPUT_S3_IMAGES_KEY}/OUT-{new_file_name}',
                            "output_label_file_url": f's3://{S3_BUCKET}/{OUTPUT_S3_IMAGES_KEY}/OUT-{os.path.splitext(new_file_name)[0]}.txt'
                            }
                    r = requests.get(url = "http://inference-service:8000/detect", params = data)
                    # os.remove(local_file_name)
                    logger.info(f"Successfully handled {new_file_name}")
                    logger.info(r.text)
                    dict = json.loads(r.text)
                    output_image_file_url = dict["output_image_file_url"]
                    
                    logger.info(f'Output file is : {output_image_file_url}')
                    bucket_name, key_name_without_file, file_name = parse_s3_url(unquote(output_image_file_url))
                    # local_output_file_name = f'{tmp_file_folder}{os.sep}{file_name}'
                    local_output_file_name = "static/images/prediction.jpg"
                    if os.path.exists(local_output_file_name)
                        os.remove(local_output_file_name)
                    s3_client.download_file(Bucket = bucket_name, Key = f'{key_name_without_file}/{file_name}', Filename = local_output_file_name)
                except requests.exceptions.HTTPError as errh:
                    logger.info("Http Error:",errh)
                except requests.exceptions.ConnectionError as errc:
                    logger.info("Error Connecting:",errc)
                except requests.exceptions.Timeout as errt:
                    logger.info("Timeout Error:",errt)
                except requests.exceptions.RequestException as err:
                    logger.info("OOps: Something Else",err)
                # except OSError as e:
                #     logger.warn ("Error deleting file: %s - %s." % (e.filename, e.strerror))
                except Exception as e:
                    logger.warn(e)
                
            
            # Get category of prediction
            category_data = "received file"
            # Render the result template
            return render_template('result.html', input_file_name=local_file_name, output_file_name = local_output_file_name)
        return redirect(request.url)
    
def parse_s3_url(s3_path: str):
    s3_path_split = s3_path.split('/')
    bucket_name = s3_path_split[2]
    key_name_without_file = '/'.join(s3_path_split[3:-1])
    file_name = s3_path_split[-1]
    return bucket_name, key_name_without_file, file_name


