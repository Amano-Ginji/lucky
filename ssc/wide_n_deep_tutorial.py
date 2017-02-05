# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "model", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "train_dev_data",
    "",
    "Path to the train-dev data.")
flags.DEFINE_string(
    "dev_data",
    "",
    "Path to the dev data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

COLUMNS = [
    'label',
    'month',
    'day',
    'sid',
    'month_day',
    'month_sid',
    'day_sid',
    'month_day_sid',
    'last_code',
    'codes_0days_0',
    'codes_0days_1',
    'codes_0days_2',
    'codes_0days_3',
    'codes_0days_4',
    'codes_0days_5',
    'codes_0days_6',
    'codes_0days_7',
    'codes_0days_8',
    'codes_0days_9',
    'codes_1days_0',
    'codes_1days_1',
    'codes_1days_2',
    'codes_1days_3',
    'codes_1days_4',
    'codes_1days_5',
    'codes_1days_6',
    'codes_1days_7',
    'codes_1days_8',
    'codes_1days_9',
    'codes_7days_0',
    'codes_7days_1',
    'codes_7days_2',
    'codes_7days_3',
    'codes_7days_4',
    'codes_7days_5',
    'codes_7days_6',
    'codes_7days_7',
    'codes_7days_8',
    'codes_7days_9',
    'codes_30days_0',
    'codes_30days_1',
    'codes_30days_2',
    'codes_30days_3',
    'codes_30days_4',
    'codes_30days_5',
    'codes_30days_6',
    'codes_30days_7',
    'codes_30days_8',
    'codes_30days_9',
    'codes_365days_0',
    'codes_365days_1',
    'codes_365days_2',
    'codes_365days_3',
    'codes_365days_4',
    'codes_365days_5',
    'codes_365days_6',
    'codes_365days_7',
    'codes_365days_8',
    'codes_365days_9'
]
LABEL_COLUMN = COLUMNS[0]
CATEGORICAL_COLUMNS = [
    'month',
    'day',
    'sid',
    'month_day',
    'month_sid',
    'day_sid',
    'month_day_sid',
    'last_code'
]
CONTINUOUS_COLUMNS = [
    'codes_0days_0',
    'codes_0days_1',
    'codes_0days_2',
    'codes_0days_3',
    'codes_0days_4',
    'codes_0days_5',
    'codes_0days_6',
    'codes_0days_7',
    'codes_0days_8',
    'codes_0days_9',
    'codes_1days_0',
    'codes_1days_1',
    'codes_1days_2',
    'codes_1days_3',
    'codes_1days_4',
    'codes_1days_5',
    'codes_1days_6',
    'codes_1days_7',
    'codes_1days_8',
    'codes_1days_9',
    'codes_7days_0',
    'codes_7days_1',
    'codes_7days_2',
    'codes_7days_3',
    'codes_7days_4',
    'codes_7days_5',
    'codes_7days_6',
    'codes_7days_7',
    'codes_7days_8',
    'codes_7days_9',
    'codes_30days_0',
    'codes_30days_1',
    'codes_30days_2',
    'codes_30days_3',
    'codes_30days_4',
    'codes_30days_5',
    'codes_30days_6',
    'codes_30days_7',
    'codes_30days_8',
    'codes_30days_9',
    'codes_365days_0',
    'codes_365days_1',
    'codes_365days_2',
    'codes_365days_3',
    'codes_365days_4',
    'codes_365days_5',
    'codes_365days_6',
    'codes_365days_7',
    'codes_365days_8',
    'codes_365days_9'
]


def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.
  month = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='month', hash_bucket_size=100)
  day = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='day',  hash_bucket_size=300)
  sid = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='sid',  hash_bucket_size=1000)
  month_day = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='month_day', hash_bucket_size=1000)
  month_sid = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='month_sid', hash_bucket_size=3000)
  day_sid = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='day_sid', hash_bucket_size=10000)
  month_day_sid = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='month_day_sid', hash_bucket_size=100000)
  last_code = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='last_code', hash_bucket_size=30)

  # Continuous base columns.
  codes_0days_0 = tf.contrib.layers.real_valued_column('codes_0days_0')
  codes_0days_1 = tf.contrib.layers.real_valued_column('codes_0days_1')
  codes_0days_2 = tf.contrib.layers.real_valued_column('codes_0days_2')
  codes_0days_3 = tf.contrib.layers.real_valued_column('codes_0days_3')
  codes_0days_4 = tf.contrib.layers.real_valued_column('codes_0days_4')
  codes_0days_5 = tf.contrib.layers.real_valued_column('codes_0days_5')
  codes_0days_6 = tf.contrib.layers.real_valued_column('codes_0days_6')
  codes_0days_7 = tf.contrib.layers.real_valued_column('codes_0days_7')
  codes_0days_8 = tf.contrib.layers.real_valued_column('codes_0days_8')
  codes_0days_9 = tf.contrib.layers.real_valued_column('codes_0days_9')
  codes_1days_0 = tf.contrib.layers.real_valued_column('codes_1days_0')
  codes_1days_1 = tf.contrib.layers.real_valued_column('codes_1days_1')
  codes_1days_2 = tf.contrib.layers.real_valued_column('codes_1days_2')
  codes_1days_3 = tf.contrib.layers.real_valued_column('codes_1days_3')
  codes_1days_4 = tf.contrib.layers.real_valued_column('codes_1days_4')
  codes_1days_5 = tf.contrib.layers.real_valued_column('codes_1days_5')
  codes_1days_6 = tf.contrib.layers.real_valued_column('codes_1days_6')
  codes_1days_7 = tf.contrib.layers.real_valued_column('codes_1days_7')
  codes_1days_8 = tf.contrib.layers.real_valued_column('codes_1days_8')
  codes_1days_9 = tf.contrib.layers.real_valued_column('codes_1days_9')
  codes_7days_0 = tf.contrib.layers.real_valued_column('codes_7days_0')
  codes_7days_1 = tf.contrib.layers.real_valued_column('codes_7days_1')
  codes_7days_2 = tf.contrib.layers.real_valued_column('codes_7days_2')
  codes_7days_3 = tf.contrib.layers.real_valued_column('codes_7days_3')
  codes_7days_4 = tf.contrib.layers.real_valued_column('codes_7days_4')
  codes_7days_5 = tf.contrib.layers.real_valued_column('codes_7days_5')
  codes_7days_6 = tf.contrib.layers.real_valued_column('codes_7days_6')
  codes_7days_7 = tf.contrib.layers.real_valued_column('codes_7days_7')
  codes_7days_8 = tf.contrib.layers.real_valued_column('codes_7days_8')
  codes_7days_9 = tf.contrib.layers.real_valued_column('codes_7days_9')
  codes_30days_0 = tf.contrib.layers.real_valued_column('codes_30days_0')
  codes_30days_1 = tf.contrib.layers.real_valued_column('codes_30days_1')
  codes_30days_2 = tf.contrib.layers.real_valued_column('codes_30days_2')
  codes_30days_3 = tf.contrib.layers.real_valued_column('codes_30days_3')
  codes_30days_4 = tf.contrib.layers.real_valued_column('codes_30days_4')
  codes_30days_5 = tf.contrib.layers.real_valued_column('codes_30days_5')
  codes_30days_6 = tf.contrib.layers.real_valued_column('codes_30days_6')
  codes_30days_7 = tf.contrib.layers.real_valued_column('codes_30days_7')
  codes_30days_8 = tf.contrib.layers.real_valued_column('codes_30days_8')
  codes_30days_9 = tf.contrib.layers.real_valued_column('codes_30days_9')
  codes_365days_0 = tf.contrib.layers.real_valued_column('codes_365days_0')
  codes_365days_1 = tf.contrib.layers.real_valued_column('codes_365days_1')
  codes_365days_2 = tf.contrib.layers.real_valued_column('codes_365days_2')
  codes_365days_3 = tf.contrib.layers.real_valued_column('codes_365days_3')
  codes_365days_4 = tf.contrib.layers.real_valued_column('codes_365days_4')
  codes_365days_5 = tf.contrib.layers.real_valued_column('codes_365days_5')
  codes_365days_6 = tf.contrib.layers.real_valued_column('codes_365days_6')
  codes_365days_7 = tf.contrib.layers.real_valued_column('codes_365days_7')
  codes_365days_8 = tf.contrib.layers.real_valued_column('codes_365days_8')
  codes_365days_9 = tf.contrib.layers.real_valued_column('codes_365days_9')

  # Wide columns and deep columns.
  wide_columns = [
      month,
      day,
      sid,
      month_day,
      month_sid,
      day_sid,
      month_day_sid,
      last_code
  ]
  deep_columns = [
      tf.contrib.layers.embedding_column(month, dimension=8),
      tf.contrib.layers.embedding_column(day, dimension=8),
      tf.contrib.layers.embedding_column(sid, dimension=8),
      tf.contrib.layers.embedding_column(last_code, dimension=8),
      codes_0days_0,
      codes_0days_1,
      codes_0days_2,
      codes_0days_3,
      codes_0days_4,
      codes_0days_5,
      codes_0days_6,
      codes_0days_7,
      codes_0days_8,
      codes_0days_9,
      codes_1days_0,
      codes_1days_1,
      codes_1days_2,
      codes_1days_3,
      codes_1days_4,
      codes_1days_5,
      codes_1days_6,
      codes_1days_7,
      codes_1days_8,
      codes_1days_9,
      codes_7days_0,
      codes_7days_1,
      codes_7days_2,
      codes_7days_3,
      codes_7days_4,
      codes_7days_5,
      codes_7days_6,
      codes_7days_7,
      codes_7days_8,
      codes_7days_9,
      codes_30days_0,
      codes_30days_1,
      codes_30days_2,
      codes_30days_3,
      codes_30days_4,
      codes_30days_5,
      codes_30days_6,
      codes_30days_7,
      codes_30days_8,
      codes_30days_9,
      codes_365days_0,
      codes_365days_1,
      codes_365days_2,
      codes_365days_3,
      codes_365days_4,
      codes_365days_5,
      codes_365days_6,
      codes_365days_7,
      codes_365days_8,
      codes_365days_9
  ]

  linear_opt = tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.1,
    l2_regularization_strength=0.001
  )

  '''
  dnn_opt = tf.train.AdadeltaOptimizer(
    learning_rate=0.1
  )
  '''

  dnn_opt = tf.train.ProximalAdagradOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.001
  )

  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns,
                                          n_classes=10,
                                          optimizer=linear_opt)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50],
                                       n_classes=10,
                                       optimizer=dnn_opt)
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        n_classes=10,
        linear_optimizer=linear_opt,
        dnn_optimizer=dnn_opt)
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values.astype(str),
      shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values.astype(int))
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval():
  """Train and evaluate the model."""
  train_file_name, train_dev_file_name, dev_file_name, test_file_name = FLAGS.train_data, FLAGS.train_dev_data, FLAGS.dev_data, FLAGS.test_data
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python")
  df_train_dev = pd.read_csv(
      tf.gfile.Open(train_dev_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python")
  df_dev = pd.read_csv(
      tf.gfile.Open(dev_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      engine="python")
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      engine="python")

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_train_dev = df_train_dev.dropna(how='any', axis=0)
  df_dev = df_dev.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print("model directory = %s" % model_dir)

  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=lambda: input_fn(df_test),
    eval_steps=FLAGS.train_steps,
    every_n_steps=10)

  m = build_estimator(model_dir)
  #m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps, monitors=[validation_monitor])
  #for i in range(0, 100):
  for i in range(0, 1):
    print("------------------------------ {}-th iter ------------------------------------".format(i))
    m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
    '''
    results = m.evaluate(input_fn=lambda: input_fn(df_train), steps=1)
    for key in sorted(results):
      print(">>> evaluate train file. %s: %s" % (key, results[key]))
    '''
    results = m.evaluate(input_fn=lambda: input_fn(df_train_dev), steps=1)
    for key in sorted(results):
      print(">>> evaluate train-dev file. %s: %s" % (key, results[key]))
    results = m.evaluate(input_fn=lambda: input_fn(df_dev), steps=1)
    for key in sorted(results):
      print(">>> evaluate dev file. %s: %s" % (key, results[key]))
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
      print(">>> evaluate test file. %s: %s" % (key, results[key]))


def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
