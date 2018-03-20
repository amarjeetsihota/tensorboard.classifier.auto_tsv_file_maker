# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This is the complete code for the following blogpost:
# https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
#   (https://goo.gl/Ujm2Ep)

import tensorflow as tf


class CSVDecoder():
  def __init__(self, file_name, feature_names, defaults, ignore_names):
    self.file_name = file_name
    self.feature_names = feature_names
    self.defaults = defaults
    self.ignore_names = ignore_names


  def decode_csv(self, line):
    # parsed_line = tf.decode_csv(line, [ [0.], [""],[""],[""],[""],[""],[""],[""],[""],[""], [0]])
    parsed_line = tf.decode_csv(line, record_defaults=self.defaults)
    label = parsed_line[-1:]  # Last element is the label
    del parsed_line[-1]  # Delete last element
    features = parsed_line  # Everything but last elements are the features

    new_feature_names = []
    new_features = []
    for i in range(0, len(self.feature_names)):
      if self.feature_names[i] not in self.ignore_names:
        new_feature_names.append(self.feature_names[i])
        new_features.append(features[i])
    d = dict(zip(new_feature_names, new_features)), label
    return d
