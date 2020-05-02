"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file (type CAM for webcam)")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client #
    client = mqtt.Client(client_id="", clean_session=True, userdata=None)
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network` #
    infer_network.load_model(args.model, args.device, args.cpu_extension)

    # Handle the input stream #
    if args.input == "CAM":
        args.input = 0
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # Grab the shape of the input
    net_input_shape = infer_network.get_input_shape()
    cap_width = int(cap.get(3))
    cap_height = int(cap.get(4))
    cap_fps = cap.get(5)

    # Loop until stream is over #
    previous_detections = 0
    total_detections = 0
    time_shift_count = 0
    time_shift = cap_fps * 1
    frame_counter = 0
    while cap.isOpened():

        # Read from the video capture #
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the image as needed #
        processed_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        processed_frame = processed_frame.transpose((2, 0, 1))
        processed_frame = processed_frame.reshape(1, *processed_frame.shape)

        # Start asynchronous inference for specified request #
        infer_network.exec_net(processed_frame)

        # Wait for the result #
        if infer_network.wait() == 0:

            # Get the results of the inference request #
            result = infer_network.get_output()[0][0]

            # Extract any desired stats from the results #
            current_detections = 0
            for detection in result:
                if detection[2] > prob_threshold and detection[1] == 1:
                    x_min = int(detection[3] * cap_width)
                    y_min = int(detection[4] * cap_height)
                    x_max = int(detection[5] * cap_width)
                    y_max = int(detection[6] * cap_height)
                    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                    current_detections += 1

            # Calculate and send relevant information on #
            frame_counter += 1
            if current_detections != previous_detections:
                time_shift_count += 1
                if time_shift_count > time_shift:
                    time_shift_count = 0
                    new_detections = max(current_detections - previous_detections, 0)
                    total_detections += new_detections
                    if current_detections >= previous_detections:
                        frame_counter = 0
                    else:
                        detection_duration = int(frame_counter / cap_fps)
                        frame_counter = 0
                        client.publish('person/duration', payload='{"duration":' + str(detection_duration) + '}', qos=0, retain=False)
                    previous_detections = current_detections
            client.publish('person', payload='{"count":' + str(current_detections) + ',"total":' + str(total_detections) + '}', qos=0, retain=False)

        # Send the frame to the FFMPEG server #
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
