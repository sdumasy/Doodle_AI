# -*- coding: utf-8 -*-
import struct
import numpy as np
from tqdm import tqdm
import ndjson
import tensorflow as tf
from PIL import Image, ImageDraw
from Constants import Constants
import time

class ImgLoader(object):
    """ This class downloads the images and converts them to Tensorflow records"""

    def __init__(self, output_file):
        self.images_per_class_per_tf_record = 1000 # The amount of images of 1 class written to a single TF record
        self.output_file = output_file
        self.C = Constants()

    def _int64_feature(self, value):
        """ Convert an int to a tfrecord train feature"""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        """ Convert a byte to a tfrecord train feature"""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def count_images_per_class(self):
        classes_dict = {}
        amount_images = 0
        for filename in tqdm(self.C.file_list): #For all the files names
            with open(filename) as f:
                data = ndjson.load(f)
                for line in enumerate(data): #Go through the strokes and construct image from it
                    amount_images += 1
                    if not filename in classes_dict:
                        classes_dict[filename] = 1
                    else:
                        classes_dict[filename] += 1
        print("Amount of classes:", len(self.C.file_list))
        print("Total amount images", amount_images)
        print(classes_dict)

    def create_image_records(self):
        """Loop through the image files and write to tfrecords"""
        data_left = True
        passes_through_file = 5
        amount = 0
        while data_left:
            amount += 1
            writer = tf.python_io.TFRecordWriter(self.output_file + '/record_' + str(passes_through_file))

            for x, filename in enumerate(tqdm(self.C.file_list)): #For all the files names
                x += 1
                if x > 3:
                    break
                with open(filename) as f:
                    data = ndjson.load(f)

                    start_index = passes_through_file * self.images_per_class_per_tf_record
                    end_index = start_index + self.images_per_class_per_tf_record

                    if end_index > len(data):
                        end_index = len(data)

                    for j in range(start_index, end_index): #Go through the strokes and construct image from it
                        line = data[j]
                        if j < (passes_through_file * self.images_per_class_per_tf_record) + self.images_per_class_per_tf_record:
                            img = self.draw_it(line).reshape(1, 255, 255)
                            img_raw = img.tostring()

                            feature={
                                'class': self._int64_feature(self.C.class_list.index(line['word'])),
                                'key': self._bytes_feature(str.encode(line['key_id'])),
                                'image_raw': self._bytes_feature(img_raw)
                            }

                            example = tf.train.Example(features=tf.train.Features(feature = feature))
                            writer.write(example.SerializeToString())
                        else:
                            break
            passes_through_file += 1
            print("closing writer")
            writer.close()
            break #only do one file for now

    def draw_it(self, sample):
        """Draw the image to a numpy array from the stroke array"""
        image = Image.new("P", (255,255), color=255)
        image_draw = ImageDraw.Draw(image)

        strokes = sample["drawing"]
        for stroke in strokes:
            for i in range(len(stroke[0])-1):

                image_draw.line([stroke[0][i],
                                 stroke[1][i],
                                 stroke[0][i+1],
                                 stroke[1][i+1]],
                                fill=0, width=6)
        return np.array(image)

    def rec_img(self, record_file):
            """Reconstruct the image and show the image"""

            record_iterator = tf.python_io.tf_record_iterator(path=record_file)
            sum = 0

            for string_record in record_iterator:

                sum +=1
                if sum>10:
                    break

                example = tf.train.Example()
                example.ParseFromString(string_record)

                img_string = (example.features.feature['image_raw']
                    .bytes_list
                    .value[0])

                label = (example.features.feature['class']
                    .int64_list
                    .value[0])

                img_1d = np.fromstring(img_string, dtype=np.uint8)
                # key = key_string.decode()
                reconstructed_img = img_1d.reshape((255, 255))
                img = Image.fromarray(reconstructed_img, 'L')
                img.show()
                print(label)

            print(sum)
            time.sleep(10)

