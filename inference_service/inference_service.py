import argparse
from fastapi import FastAPI, File, UploadFile
from triton_client import TritonClient
import logging
from urllib.parse import unquote
import boto3
import os

_triton_url = 'triton:8002'
_model = 'yolov7-visdrone-finetuned'
logger = logging.getLogger('inference_service')


#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Aerial Detection Inference Service')
tmp_file_folder_input = "./tmp_data/input"
tmp_output_img_folder = "./tmp_data/output/image"
tmp_output_lbl_folder = "./tmp_data/output/label"

client = boto3.client('s3')
global triton_client

#The inference-service endpoint receives post requests with the image and returns the transformed image
@app.get("/detect/", tags=["Object Detect"])
async def detect(input_image_file_url: str, output_image_file_url: str, output_label_file_url: str):
    #We read the file and decode it
    # s3://aerial-detection-mlops4/inferencing/photos/input/19d09312c52945f8bcdd283c627d9b44-9999942_00000_d_0000214.jpg
    bucket_name, key_name_without_file, file_name = parse_s3_url(unquote(input_image_file_url))
    os.makedirs(tmp_file_folder_input, exist_ok=True)
    os.makedirs(tmp_output_img_folder, exist_ok=True)
    os.makedirs(tmp_output_lbl_folder, exist_ok=True)
    temp_input_filename = f'{tmp_file_folder_input}{os.sep}{file_name}'
    temp_output_image_filename = f'{tmp_output_img_folder}{os.sep}OUT-{file_name}'
    temp_output_label_filename = f'{tmp_output_lbl_folder}{os.sep}OUT-{os.path.splitext(file_name)[0]}.txt'
    client.download_file(Bucket = bucket_name, Key = f'{key_name_without_file}/{file_name}', Filename = temp_input_filename)
    logger.info(f'created local temp file : {temp_input_filename}')
    
    logger.info(f'bucket_name = {bucket_name}, key_name_without_file = {key_name_without_file}, file_name = {file_name}')
    logger.info("input_image_file_url: " + unquote(input_image_file_url))
    logger.info("output_image_file_url: " + unquote(output_image_file_url))
    logger.info("output_label_file_url: " + unquote(output_label_file_url))
    try:
        get_triton_client().detectImage(input_image_file=temp_input_filename, output_image_file=temp_output_image_filename, output_label_file=temp_output_label_filename)
        out_bucket_name, out_key_name_without_file, out_file_name = parse_s3_url(unquote(output_image_file_url))
        new_out_image_file_name_only = temp_output_image_filename.split('/')[-1]
        client.upload_file(Bucket = out_bucket_name, Filename = temp_output_image_filename, Key = f'{out_key_name_without_file}/{new_out_image_file_name_only}')
    except Exception as e:
        logger.error(e)
    return {"input_image_file_url": input_image_file_url, "output_image_file_url": temp_output_image_filename}

def parse_s3_url(s3_path: str):
    s3_path_split = s3_path.split('/')
    bucket_name = s3_path_split[2]
    key_name_without_file = '/'.join(s3_path_split[3:-1])
    file_name = s3_path_split[-1]
    return bucket_name, key_name_without_file, file_name

def get_triton_client():
    if triton_client is None:
        triton_client = TritonClient(model = _model, triton_url = _triton_url)
    return triton_client


@app.get("/", tags=["Health Check"])
async def root():
    return {"inference_service_health": "Ok"}
