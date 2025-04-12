#!/bin/bash
# Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Enable error checking and verbose execution
# -e: Exit immediately if a command exits with a non-zero status
# -x: Print each command before executing it (helpful for debugging)
set -ex

# Convert Tensorflow inception V3 module to ONNX
# Pre-requisite: Python3, venv, and Pip3 are installed on the system
# Create the directory structure for the Inception model in the model repository
mkdir -p model_repository/inception_onnx/1

# Download the pre-trained Inception V3 TensorFlow model
# -O: Specify the output file location
wget -O /tmp/inception_v3_2016_08_28_frozen.pb.tar.gz \
     https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz

# Extract the downloaded tarball in the /tmp directory
# The parentheses create a subshell so the cd command doesn't affect the parent shell
# Change to /tmp directory and extract the tarball
# tar options:
# x - extract
# z - decompress using gzip
# f - use archive file
(cd /tmp && tar xzf inception_v3_2016_08_28_frozen.pb.tar.gz)

# Create a Python virtual environment named 'tf2onnx' to isolate dependencies
python3 -m venv tf2onnx

# Activate the virtual environment to use its Python interpreter and packages
source ./tf2onnx/bin/activate

# Install required Python packages in the virtual environment
# - numpy<2: Compatible version of NumPy
# - tensorflow: Required to load the TensorFlow model
# - tf2onnx: Tool to convert TensorFlow models to ONNX format
pip3 install "numpy<2" tensorflow tf2onnx

# Convert the TensorFlow model to ONNX format
# --graphdef: Path to the input TensorFlow frozen graph
# --output: Path for the output ONNX model
# --inputs: Name of the input tensor in the TensorFlow model
# --outputs: Name of the output tensor in the TensorFlow model
# Convert the TensorFlow model (.pb file) to ONNX format
# .pb (Protocol Buffer) files are TensorFlow's serialized model format that contains
# the model's graph definition and weights in a binary format
# Convert the Inception V3 model to ONNX format
# Inception V3 is a pre-trained image classification model that can classify images into 1000 categories
# It's commonly used for image recognition tasks and transfer learning
# The frozen.pb file contains the pre-trained model weights and architecture in TensorFlow's format
python3 -m tf2onnx.convert --graphdef /tmp/inception_v3_2016_08_28_frozen.pb --output inception_v3_onnx.model.onnx --inputs input:0 --outputs InceptionV3/Predictions/Softmax:0

# Exit the virtual environment after conversion is complete
deactivate

# Move the converted ONNX model to the appropriate location in the model repository
mv inception_v3_onnx.model.onnx model_repository/inception_onnx/1/model.onnx


# Download a pre-trained DenseNet ONNX model
# First, create the directory structure for the DenseNet model in the model repository
mkdir -p model_repository/densenet_onnx/1

# Download the pre-trained DenseNet ONNX model directly to the model repository
# -O: Specify the output file location
     # DenseNet-7 is a version of DenseNet-121 (a convolutional neural network with 121 layers)
     # that has been optimized and converted to ONNX format.
     # DenseNet is known for its dense connectivity pattern where each layer connects to every other layer,
     # making it efficient for image classification tasks.
wget -O model_repository/densenet_onnx/1/model.onnx \
     https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-7.onnx
