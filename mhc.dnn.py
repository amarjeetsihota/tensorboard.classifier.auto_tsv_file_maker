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
from agiledev_ai import csv_decoder
from agiledev_ai import dnn_helpers


# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

# Windows users: You only need to change PATH, rest is platform independent
LOG = "./log/"
BATCH_SIZE = 32
COUNTRY_BUCKET_SIZE = 40

FILE_TRAIN = "./data/survey.train.csv"
FILE_TEST = "./data/survey.test.csv"





def build_model_columns(feature_helper):
  """Builds a set of wide and deep feature columns."""
  # Continuous columns

  age = tf.feature_column.numeric_column('age')

  agebuckets = tf.feature_column.bucketized_column(source_column=age, boundaries=[20,30,40,50,60])


  gender = tf.feature_column.categorical_column_with_vocabulary_list(
    'gender', [
      'Male', 'Female', 'None'])

  country = tf.feature_column.categorical_column_with_vocabulary_list(
    'country',
    ['United States', 'Canada', 'United Kingdom', 'Bulgaria', 'France', 'Portugal', 'Netherlands',
     'Switzerland',
     'Poland', 'Australia', 'Germany', 'Russia', 'Mexico', 'Brazil', 'Slovenia', 'Costa Rica',
     'Austria', 'Ireland',
     'India', 'South Africa', 'Italy', 'Sweden', 'Colombia', 'Latvia', 'Romania', 'Belgium',
     'New Zealand', 'Spain',
     'Finland', 'Uruguay', 'Israel', 'Bosnia and Herzegovina', 'Hungary', 'Singapore', 'Japan',
     'Nigeria', 'Croatia',
     'Norway', 'Thailand',
     'Denmark'])

  state = tf.feature_column.categorical_column_with_vocabulary_list(
    'state',
    ['IL', 'IN', 'NA', 'TX', 'TN', 'MI', 'OH', 'CA', 'CT', 'MD', 'NY', 'NC', 'MA', 'IA', 'PA', 'WA', 'WI', 'UT', 'NM',
     'OR', 'FL', 'MN', 'MO', 'AZ', 'CO', 'GA', 'DC', 'NE', 'WV', 'OK', 'KS', 'VA', 'NH', 'KY', 'AL', 'NV', 'NJ', 'SC',
     'VT', 'SD', 'ID', 'MS', 'RI',
     'WY', 'LA', 'ME'])

  self_employed = tf.feature_column.categorical_column_with_vocabulary_list(
    'self_employed', [
      'Yes', 'No', 'NA'])

  family_history = tf.feature_column.categorical_column_with_vocabulary_list(
    'family_history', [
      'Yes', 'No'])

  work_interference = tf.feature_column.categorical_column_with_vocabulary_list(
    'work_interference', [
      'NA', 'Never', 'Often', 'Rarely', 'Sometimes'])

  no_employees = tf.feature_column.categorical_column_with_vocabulary_list(
    'no_employees', [
      '1to5', '6to26', '26to100', '100to500', '500to1000', 'MoreThan1000'])

  remote_work = tf.feature_column.categorical_column_with_vocabulary_list(
    'remote_work', [
      'Yes', 'No'])

  tech_company = tf.feature_column.categorical_column_with_vocabulary_list(
    'tech_company', [
      'Yes', 'No'])

  benefits = tf.feature_column.categorical_column_with_vocabulary_list(
    'benefits', [
      'Yes', 'No', 'NA'])

  care_options = tf.feature_column.categorical_column_with_vocabulary_list(
    'care_options', [
      'Yes', 'No', 'Not Sure'])

  wellness_program = tf.feature_column.categorical_column_with_vocabulary_list(
    'wellness_program', [
      'Yes', 'No', 'NA'])

  seek_help = tf.feature_column.categorical_column_with_vocabulary_list(
    'seek_help', [
      'Yes', 'No', 'NA'])

  anonymity = tf.feature_column.categorical_column_with_vocabulary_list(
    'anonymity', [
      'Yes', 'No', 'NA'])

  leave = tf.feature_column.categorical_column_with_vocabulary_list(
    'leave', [
      'NA', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', 'Very easy'])

  mental_health_consequence = tf.feature_column.categorical_column_with_vocabulary_list(
    'mental_health_consequence', [
      'Yes', 'No', 'Maybe'])

  phys_health_consequence = tf.feature_column.categorical_column_with_vocabulary_list(
    'phys_health_consequence', [
      'Yes', 'No', 'Maybe'])

  coworkers = tf.feature_column.categorical_column_with_vocabulary_list(
    'coworkers', [
      'Yes', 'No', 'Some of them'])

  supervisor = tf.feature_column.categorical_column_with_vocabulary_list(
    'supervisor', [
      'Yes', 'No', 'Some of them'])

  mental_health_interview = tf.feature_column.categorical_column_with_vocabulary_list(
    'mental_health_interview', [
      'Yes', 'No', 'Maybe'])

  phys_health_interview = tf.feature_column.categorical_column_with_vocabulary_list(
    'phys_health_interview', [
      'Yes', 'No', 'Maybe'])

  mental_vs_physical = tf.feature_column.categorical_column_with_vocabulary_list(
    'mental_vs_physical', [
      'Yes', 'No', 'NA'])

  obs_consequence = tf.feature_column.categorical_column_with_vocabulary_list(
    'obs_consequence', [
      'Yes', 'No'])

  deep_columns = [
    tf.feature_column.indicator_column(agebuckets),
    tf.feature_column.indicator_column(gender),
    tf.feature_column.indicator_column(country),
    tf.feature_column.indicator_column(state),
    tf.feature_column.indicator_column(self_employed),
    tf.feature_column.indicator_column(family_history),
    tf.feature_column.indicator_column(work_interference),
    tf.feature_column.indicator_column(no_employees),
    tf.feature_column.indicator_column(remote_work),
    tf.feature_column.indicator_column(tech_company),
    tf.feature_column.indicator_column(benefits),
    tf.feature_column.indicator_column(care_options),
    tf.feature_column.indicator_column(wellness_program),
    tf.feature_column.indicator_column(seek_help),
    tf.feature_column.indicator_column(anonymity),
    tf.feature_column.indicator_column(leave),
    tf.feature_column.indicator_column(mental_health_consequence),
    tf.feature_column.indicator_column(phys_health_consequence),
    tf.feature_column.indicator_column(coworkers),
    tf.feature_column.indicator_column(supervisor),
    tf.feature_column.indicator_column(mental_health_interview),
    tf.feature_column.indicator_column(phys_health_interview),
    tf.feature_column.indicator_column(mental_vs_physical),
    tf.feature_column.indicator_column(obs_consequence)

  ]

  feature_column_helper.addDeepColumns(deep_columns)

  return deep_columns






def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
  # Create an input function reading a file using the Dataset API
  # Then provide the results to the Estimator API
  csv_defaults = [tf.constant([], dtype=tf.int32),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.string),
                  tf.constant([], dtype=tf.int32)

                  ]
  feature_names = ['age', 'gender', 'country', 'state', 'self_employed', 'family_history', 'work_interference',
                   'no_employees',
                   'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help',
                   'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers',
                   'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical',
                   'obs_consequence'
                   ]

  ignore_names = []

  decoder = csv_decoder.CSVDecoder(file_path, feature_names, csv_defaults, ignore_names)
  dataset = (tf.data.TextLineDataset(file_path)  # Read text file
    .skip(1)  # Skip header row
    .map(decoder.decode_csv))  # Transform each elem by applying decode_csv fn
  if perform_shuffle:
    # Randomizes input using a window of 256 elements (read into memory)
    dataset = dataset.shuffle(buffer_size=256)

  dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
  dataset = dataset.batch(BATCH_SIZE)  # Batch size to use
  iterator = dataset.make_one_shot_iterator()
  batch_features, batch_labels = iterator.get_next()

  return batch_features, batch_labels


def tbexport(model, tsv_file,  dims, start_row_no, tensor_name):
  ## outputs the tsv file + tensor
  tensor_file = open( LOG + tensor_name +  "_tensor.tsv", "w")


  placeholder = np.zeros((len(model.index2word), dims ))

  for row_no, word in enumerate(model.index2word):
    try:
      placeholder[row_no] = model[word]
      tsv_file.write("{}\t{}\n".format(start_row_no + row_no,word))
      tensor_string = ""
      for f in model[word]:
        tensor_string  = tensor_string + "%f\t" % f

      tensor_file.write( tensor_string +  '\n'  )
    except:
      pass

  tensor_file.close()


  return placeholder
# --------
# start
# ----------
tf.logging.set_verbosity(tf.logging.INFO)
np.set_printoptions(threshold=np.nan)


config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "dnn/hiddenlayer_0/kernel"
embedding.metadata_path = "metadata.tsv"
projector.visualize_embeddings(tf.summary.FileWriter(LOG), config)


feature_column_helper = dnn_helpers.FeatureColumnLookup()
tbhook = dnn_helpers.TBHookClass(feature_column_helper, LOG)



# Create a deep neural network regression classifier
# Use the DNNClassifier pre-made estimator
classifier = tf.estimator.DNNClassifier(
  feature_columns=build_model_columns(feature_column_helper),  # The input features to our model
  hidden_units=[100, 10],  # Two layers, each with 10 neurons
  n_classes=2,  # binary i.e. 0 or 1
  model_dir=LOG,
  config=tf.contrib.learn.RunConfig(
    save_checkpoints_steps=200,  # determines how many steps per checkpoint
    save_checkpoints_secs=None,
    save_summary_steps=50
  ))  # Path to where checkpoints etc are stored

print("finished building classifer")




# Train our model, use the previously function my_input_fn
# Input to training is a file with training example
# Stop training after 8 iterations of train data (epochs)
print("started training")

classifier.train(
  input_fn=lambda: my_input_fn(FILE_TRAIN, True, 300), hooks=[tbhook])

print("finished training")

evaluate_result = classifier.evaluate(
  input_fn=lambda: my_input_fn(FILE_TEST, False, 1))

print("create tensor file")

weights = classifier.get_variable_value('dnn/hiddenlayer_0/kernel')

tensor_file = open(LOG + "classifier_tensor.tsv", "w")

for row_no, val in enumerate(weights):
  try:
    tensor_string = ""
    for f in val:
      tensor_string = tensor_string + "%f\t" % f

    tensor_file.write(tensor_string + "\n")
  except:
    pass

tensor_file.close()

print("Evaluation results")
for key in evaluate_result:
  print("   {}, was: {}".format(key, evaluate_result[key]))

