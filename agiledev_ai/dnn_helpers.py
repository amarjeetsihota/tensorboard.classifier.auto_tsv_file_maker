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

import os

import six.moves.urllib.request as request
import tensorflow as tf
import numpy as np
import inspect as inspect
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training import training_util
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.feature_column import feature_column as tfc
from enum import Enum



class FeatureColumnLookup():
  def __init__(self):
    self.feature_data_dict = {}


  def add(self, key, obj):
    self.feature_data_dict[key] = obj

  def addDeepColumns(self, dense_columns):
    for col in dense_columns:
      self.add(col.name, col)

  def get_label(self, feature_key, i):
    # three types of dense columns numeric, indicator or embedding
    dense_column = self.feature_data_dict[feature_key]
    if isinstance(dense_column, tfc._EmbeddingColumn):
      return 'E' + str(i)
    elif isinstance(dense_column, tfc._IndicatorColumn):
      categorical_column = dense_column.categorical_column
      if isinstance(categorical_column, tfc._VocabularyListCategoricalColumn):
        vocab_list = categorical_column.vocabulary_list
        return vocab_list[i]
      elif isinstance(categorical_column, tfc._BucketizedColumn):
        boundaries = categorical_column.boundaries
        try:
          return boundaries[i]
        except:
          return ">" + str(boundaries[i - 1])

    # this should only be numeric col
    return i


class TBHookClass(session_run_hook.SessionRunHook):
  """Hook to extend calls to MonitoredSession.run()."""

  def __init__(self, feature_column_helper, log):
    self.batch_no = -1
    self.feature_column_helper = feature_column_helper
    self.log = log

  def begin(self):
    with open(self.log + "metadata.tsv", "w") as f:
      f.write("Index\tLabel\n")
      concat = tf.get_default_graph().get_tensor_by_name("dnn/input_from_feature_columns/input_layer/concat:0")
      row_no = 0
      for i in concat.op.node_def.input:
        tensor = tf.get_default_graph().get_tensor_by_name(i + ":0")
        shape = tensor.shape.as_list()
        tensor_name = tensor.name.split("/")[3]
        if len(shape) == 2:
          if shape[0] == None:
            if tensor.op.op_def.name == "Reshape":
              for j in range(0, shape[1]):
                lookup_label = self.feature_column_helper.get_label(tensor_name, j)
                f.write("{}\t{}:{}\n".format(row_no, tensor_name[:10], lookup_label))
                row_no = row_no + 1
    f.close()

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    pass

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self.batch_no = self.batch_no + 1

    concat = tf.get_default_graph().get_tensor_by_name("dnn/input_from_feature_columns/input_layer/concat:0")
    tensor_name1 = concat.name
    return session_run_hook.SessionRunArgs([tensor_name1])

  def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):  # pylint: disable=unused-argument
    if self.batch_no < 1:
      results = np.array(run_values[0][0])  # results is index 0
      np.savetxt(self.log + "concat.dump" + str(self.batch_no) + ".csv", results, delimiter=",")

  def end(self, session):  # pylint: disable=unused-argument
    pass

