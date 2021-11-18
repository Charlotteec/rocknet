import jetson.inference
import jetson.utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")

opt = parser.parse_args()

img = jetson.utils.loadImage(opt.filename)

net = jetson.inference.imageNet(argv=['--model=model/rocknet.onnx', '--input_blob=input_0', '--output_blob=output_0', '--labels=labels.txt'])

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
