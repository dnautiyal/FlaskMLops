FROM python:3.8.13
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["uvicorn", "inference_service:app","--host", "0.0.0.0", "--port", "8001"]