#!/usr/bin/env bash
set -e

trap "exit" SIGINT

rm -rf ./data/fashion/
wget --directory-prefix=./data/fashion/ http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget --directory-prefix=./data/fashion/ http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget --directory-prefix=./data/fashion/ http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget --directory-prefix=./data/fashion/ http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
