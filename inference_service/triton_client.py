import numpy as np
import sys
import cv2
import argparse
import logging

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import VisDroneLabels

class TritonClient:
    INPUT_NAMES = ["images"]
    OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]


    # https://docs.python.org/3/library/argparse.html#argumentparser-objects
    def __init__(self, model = 'yolov7-visdrone-finetuned', triton_url='triton:8001'):
        """
        We instantiate the TritonClient class with the triton_url
        Args:
            - triton_url (str): path to the triton server
        """
        # self.FLAGS = FLAGS_PARAM
        self.model = model #Inference model name, default yolov7
        self.logger = logging.getLogger(__name__)
        # Create server context
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url= triton_url, # Inference server URL, default localhost:8001
                verbose= False, #Enable verbose client output
                ssl= False, #Enable SSL encrypted channel to the server
                root_certificates= None, #File holding PEM-encoded root certificates, default none
                private_key= None, #File holding PEM-encoded private key, default is none
                certificate_chain= None) #File holding PEM-encoded certicate chain default is none
        except Exception as e:
            self.logger.error("context creation failed: " + str(e))
            sys.exit()

        # Health check
        if not self.triton_client.is_server_live():
            self.logger.error("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            self.logger.error("FAILED : is_server_ready")
            sys.exit(1)

        if not self.triton_client.is_model_ready(self.FLAGS.model):
            self.logger.error("FAILED : is_model_ready")
            sys.exit(1)

        if self.FLAGS.model_info:
            # Model metadata
            try:
                self.metadata = self.triton_client.get_model_metadata(model)
                self.logger.info(self.metadata)
            except InferenceServerException as ex:
                if "Request for unknown model" not in ex.message():
                    self.logger.error("FAILED : get_model_metadata")
                    self.logger.error("Got: {}".format(ex.message()))
                    sys.exit(1)
                else:
                    self.logger.error("FAILED : get_model_metadata")
                    sys.exit(1)

            # Model configuration
            try:
                self.config = self.triton_client.get_model_config(model)
                if not (self.config.config.name == model):
                    self.logger.error("FAILED: get_model_config")
                    sys.exit(1)
                self.logger.info(self.config)
            except InferenceServerException as ex:
                self.logger.error("FAILED : get_model_config")
                self.logger.error("Got: {}".format(ex.message()))
                sys.exit(1)

    def detectImage(self, input_image_file, output_image_file, output_label_file, image_width = 960, image_height = 544):
        # if self.FLAGS.mode == 'image':
        #     self.logger.info("Running in 'image' mode")
        if not input_image_file:
            self.logger.warn("FAILED: no input image")
            return
        if not output_image_file:
            self.logger.warn("FAILED: no output_image_file specified")
            return

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(self.INPUT_NAMES[0], [1, 3, image_width, image_height], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[3]))

        self.logger.info("Creating buffer from image file...")
        input_image = cv2.imread(str(input_image_file))
        if input_image is None:
            self.logger.warn(f"FAILED: could not load input image {str(input)}")
            return
        input_image_buffer = preprocess(input_image, [image_width, image_height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

        inputs[0].set_data_from_numpy(input_image_buffer)

        self.logger.info("Invoking inference...")
        results = self.triton_client.infer(model_name=self.FLAGS.model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=self.FLAGS.client_timeout)
        if self.FLAGS.model_info:
            statistics = self.triton_client.get_inference_statistics(model_name=self.model)
            if len(statistics.model_stats) != 1:
                self.logger.warn("FAILED: get_inference_statistics")
                # sys.exit(1)
            self.logger.info(statistics)
        self.logger.info("Done")

        for output in self.OUTPUT_NAMES:
            result = results.as_numpy(output)
            self.logger.info(f"Received result buffer \"{output}\" of size {result.shape}")
            self.logger.info(f"Naive buffer sum: {np.sum(result)}")

        num_dets = results.as_numpy(self.OUTPUT_NAMES[0])
        det_boxes = results.as_numpy(self.OUTPUT_NAMES[1])
        det_scores = results.as_numpy(self.OUTPUT_NAMES[2])
        det_classes = results.as_numpy(self.OUTPUT_NAMES[3])
        detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, input_image.shape[1], input_image.shape[0], [image_width, image_height])
        self.logger.info(f"Detected objects: {len(detected_objects)}")

        for box in detected_objects:
            self.logger.info(f"{VisDroneLabels(box.classID).name}: {box.confidence}")
            input_image = render_box(input_image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
            size = get_text_size(input_image, f"{VisDroneLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
            input_image = render_filled_box(input_image, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
            input_image = render_text(input_image, f"{VisDroneLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

        if output_image_file:
            cv2.imwrite(output_image_file, input_image)
            self.logger.info(f"Saved result to {output_image_file}")
        else:
            cv2.imshow('image', input_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
