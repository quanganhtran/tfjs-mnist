/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import assert from 'assert';

// MNIST data constants:
const BASE_URL = 'http://localhost:8081/';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte';
const TEST_IMAGES_FILE = 't10k-images-idx3-ubyte';
const TEST_LABELS_FILE = 't10k-labels-idx1-ubyte';
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
export const IMAGE_HEIGHT = 28;
export const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

const cache = {};

// Downloads a test file only once and returns the buffer for the file.
async function fetchOnceAndSaveToDiskWithBuffer(filename) {

    const url = `${BASE_URL}${filename}`;
    if (cache[filename]) {
        return cache[filename];
    }
    console.log(`  * Downloading from: ${url}`);
    return fetch(url).then((response) => {
        cache[filename] = response.arrayBuffer();
        return cache[filename];
    });
}

function loadHeaderValues(buffer, headerLength) {
    const dv = new DataView(buffer);
    const headerValues = [];
    for (let i = 0; i < headerLength / 4; i++) {
        // Header data is stored in-order (aka big-endian)
        headerValues[i] = dv.getUint32(i * 4);
    }
    return headerValues;
}

async function loadImages(filename) {
    const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);
    const dv = new DataView(buffer);

    const headerBytes = IMAGE_HEADER_BYTES;
    const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

    const headerValues = loadHeaderValues(buffer, headerBytes);
    assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
    assert.equal(headerValues[2], IMAGE_HEIGHT);
    assert.equal(headerValues[3], IMAGE_WIDTH);

    const images = [];
    let index = headerBytes;
    while (index < dv.byteLength) {
        const array = new Float32Array(recordBytes);
        for (let i = 0; i < recordBytes; i++) {
            // Normalize the pixel values into the 0-1 interval, from
            // the original 0-255 interval.
            array[i] = dv.getUint8(index++) / 255;
        }
        images.push(array);
    }

    assert.equal(images.length, headerValues[1]);
    return images;
}

async function loadLabels(filename) {
    const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);
    const dv = new DataView(buffer);

    const headerBytes = LABEL_HEADER_BYTES;
    const recordBytes = LABEL_RECORD_BYTE;

    const headerValues = loadHeaderValues(buffer, headerBytes);
    assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);

    const labels = [];
    let index = headerBytes;
    while (index < dv.byteLength) {
        const array = new Int32Array(recordBytes);
        for (let i = 0; i < recordBytes; i++) {
            array[i] = dv.getUint8(index++);
        }
        labels.push(array);
    }

    assert.equal(labels.length, headerValues[1]);
    return labels;
}

/** Helper class to handle loading training and test data. */
export class MnistDataset {
    constructor() {
        this.dataset = null;
        this.trainSize = 0;
        this.testSize = 0;
        this.trainBatchIndex = 0;
        this.testBatchIndex = 0;
    }

    /** Loads training and test data. */
    async loadData(dataType) {
        this.dataset = await Promise.all([
            loadImages(`${dataType}/${TRAIN_IMAGES_FILE}`),
            loadLabels(`${dataType}/${TRAIN_LABELS_FILE}`),
            loadImages(`${dataType}/${TEST_IMAGES_FILE}`),
            loadLabels(`${dataType}/${TEST_LABELS_FILE}`)
        ]);
        this.trainSize = this.dataset[0].length;
        this.testSize = this.dataset[2].length;
    }

    getTrainData() {
        return this.getData_(true);
    }

    getTestData(length) {
        const data = this.getData_(false);
        return {
            xs: length ? data.xs.slice(0, length) : data.xs,
            labels: data.labels
        }
    }

    getData_(isTrainingData) {
        let imagesIndex;
        let labelsIndex;
        if (isTrainingData) {
            imagesIndex = 0;
            labelsIndex = 1;
        } else {
            imagesIndex = 2;
            labelsIndex = 3;
        }
        const size = this.dataset[imagesIndex].length;
        tf.util.assert(
            this.dataset[labelsIndex].length === size,
            `Mismatch in the number of images (${size}) and ` +
            `the number of labels (${this.dataset[labelsIndex].length})`);

        // Only create one big array to hold batch of images.
        const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
        const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
        const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

        let imageOffset = 0;
        let labelOffset = 0;
        for (let i = 0; i < size; ++i) {
            images.set(this.dataset[imagesIndex][i], imageOffset);
            labels.set(this.dataset[labelsIndex][i], labelOffset);
            imageOffset += IMAGE_FLAT_SIZE;
            labelOffset += 1;
        }

        return {
            xs: tf.tensor4d(images, imagesShape),
            labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
        };
    }
}
