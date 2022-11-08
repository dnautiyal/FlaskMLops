import argparse
from fastapi import FastAPI, File, UploadFile
# from triton_client import TritonClient
import logging
from urllib.parse import unquote
import boto3
import os

triton_url = 'triton:8000'
model = 'yolov7-visdrone-finetuned'
logger = logging.getLogger('inference_service')
# triton_client = TritonClient(model, triton_url)

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Aerial Detection Inference Service')
tmp_file_folder_input = "./tmp_data/input"

client = boto3.client('s3')

#The inference-service endpoint receives post requests with the image and returns the transformed image
@app.get("/detect/", tags=["Object Detect"])
async def detect(input_image_file_url: str, output_image_file_url: str, output_label_file_url: str):
    #We read the file and decode it
    # s3://aerial-detection-mlops4/inferencing/photos/input/19d09312c52945f8bcdd283c627d9b44-9999942_00000_d_0000214.jpg
    bucket_name, key_name_without_file, file_name = parse_s3_url(unquote(input_image_file_url))
    os.makedirs(tmp_file_folder_input, exist_ok=True)
    temp_input_filename = f'{tmp_file_folder_input}{os.sep}{file_name}'
    client.download_file(Bucket = bucket_name, Key = f'{key_name_without_file}/{file_name}', Filename = temp_input_filename)
    print(f'created local temp file : {temp_input_filename}')
    
    print(f'bucket_name = {bucket_name}, key_name_without_file = {key_name_without_file}, file_name = {file_name}')
    print("input_image_file_url: " + unquote(input_image_file_url))
    print("output_image_file_url: " + unquote(output_image_file_url))
    print("output_label_file_url: " + unquote(output_label_file_url))
    return {"input_image_file_url": input_image_file_url, "output_image_file_url": output_image_file_url}

def parse_s3_url(s3_path: str):
    s3_path_split = s3_path.split('/')
    bucket_name = s3_path_split[2]
    key_name_without_file = '/'.join(s3_path_split[3:-1])
    file_name = s3_path_split[-1]
    return bucket_name, key_name_without_file, file_name




@app.get("/", tags=["Health Check"])
async def root():
    return {"inference_service_health": "Ok"}
