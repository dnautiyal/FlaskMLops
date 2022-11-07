import argparse
from fastapi import FastAPI, File, UploadFile
from triton_client import TritonClient
import logging
from urllib.parse import unquote 

triton_url = 'triton:8000'
model = 'yolov7-visdrone-finetuned'
logger = logging.getLogger('inference_service')
# triton_client = TritonClient(model, triton_url)

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Aerial Detection Inference Service')

#The inference-service endpoint receives post requests with the image and returns the transformed image
@app.get("/detect/", tags=["Object Detect"])
async def detect(input_image_file_url: str, output_image_file_url: str, output_label_file_url: str):
    #We read the file and decode it
    # s3://aerial-detection-mlops4/inferencing/photos/input/19d09312c52945f8bcdd283c627d9b44-9999942_00000_d_0000214.jpg'
    print("input_image_file_url: " + unquote(input_image_file_url))
    print("output_image_file_url: " + unquote(output_image_file_url))
    print("output_label_file_url: " + unquote(output_label_file_url))
    return {"input_image_file_url": input_image_file_url, "output_image_file_url": output_image_file_url}


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}
