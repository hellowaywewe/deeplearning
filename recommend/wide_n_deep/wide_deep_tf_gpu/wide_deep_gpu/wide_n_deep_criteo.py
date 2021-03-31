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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import argparse
import pandas as pd

import tensorflow as tf


CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]
LABEL_COLUMN = ["clicked"]

TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS


def column_to_dtype(column):
    if column in CATEGORICAL_COLUMNS:
        return tf.string
    else:
        return tf.float32


def serving_input_fn():
    feature_placeholders = {
        column: tf.placeholder(column_to_dtype(column), [None])
        for column in FEATURE_COLUMNS
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def generate_input_fn(data_file, num_epochs, shuffle, batch_size):
    df_data = pd.read_csv(tf.io.gfile.GFile(data_file), names=TRAIN_DATA_COLUMNS, skipinitialspace=True,
                          engine="python", skiprows=1)
    # remove NaN elements
    df_data = df_data.dropna(how="any", axis=0)
    labels = df_data["clicked"].astype(int)
    return tf.estimator.inputs.pandas_input_fn(x=df_data, y=labels, batch_size=batch_size,
                                               num_epochs=num_epochs, shuffle=shuffle, num_threads=5)


def build_model(model_type, model_dir, wide_columns, deep_columns):
    runconfig = tf.estimator.RunConfig(
        save_checkpoints_secs=120,
        save_checkpoints_steps=None
    )
    m = None
    if model_type == 'wide':
        m = tf.estimator.LinearClassifier(
            config=runconfig,
            model_dir=model_dir,
            feature_columns=wide_columns)

    elif model_type == 'deep':
        m = tf.estimator.DNNClassifier(
            config=runconfig,
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 70, 50, 25])

    elif model_type == 'wide_and_deep':
        m = tf.estimator.DNNLinearCombinedClassifier(
            config=runconfig,
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 70, 50, 25])
    else:
        raise ValueError("The model_type should be in {'wide', 'deep', 'wide_and_deep'}.")

    return m


def build_feature_cols():
    # Sparse base columns.
    wide_columns = []
    for name in CATEGORICAL_COLUMNS:
        wide_columns.append(tf.feature_column.categorical_column_with_hash_bucket(name, hash_bucket_size=1000))

    # Continuous base columns.
    deep_columns = []
    for name in CONTINUOUS_COLUMNS:
        deep_columns.append(tf.feature_column.numeric_column(name))

    # Embed wide columns into deep columns
    for col in wide_columns:
        deep_columns.append(tf.feature_column.embedding_column(col, dimension=8))

    return wide_columns, deep_columns


def build_estimator(model_type='wide_and_deep', model_dir=None):
    if model_dir is None:
        model_dir = 'models/model_' + model_type + '_' + str(int(time.time()))
        print("Model directory = %s" % model_dir)

    wide_columns, deep_columns = build_feature_cols()
    m = build_model(model_type, model_dir, wide_columns, deep_columns)
    print('Estimator built')
    return m


def train_and_eval(args_opt):
    print("Begin training and evaluation")
    output_dir = args_opt.output_dir
    if output_dir is None:
        raise ValueError("The output_dir is None.")

    if output_dir[-1] != '/':
        output_dir += '/'

    train_file = args_opt.train_file
    test_file = args_opt.test_file
    model_type = args_opt.model_type
    batch_size = args_opt.batch_size
    train_line_count = args_opt.train_line_count
    num_epochs = args_opt.num_epochs
    eval_line_count = args_opt.eval_line_count
    eval_batch_size = args_opt.eval_batch_size

    # train_steps = (train_line_count/batch_size) * num_epochs
    train_steps = (num_epochs * train_line_count) / batch_size
    test_steps = eval_line_count / eval_batch_size

    model_dir = output_dir + model_type + '_' + str(int(time.time()))
    print("Save model checkpoints to " + model_dir)
    export_dir = model_dir + '/exports'

    m = build_estimator(model_type, model_dir)

    m.train(
        input_fn=generate_input_fn(train_file, num_epochs=num_epochs, shuffle=True, batch_size=batch_size),
        steps=train_steps)
    print('Train done')

    results = m.evaluate(
        input_fn=generate_input_fn(test_file, num_epochs=1, shuffle=False, batch_size=eval_batch_size),
        steps=test_steps)
    print('Evaluate done')
    print("AUC: %s" % results['auc'])

    # for key in sorted(results):
    #     print("%s: %s" % (key, results[key]))

    m.export_saved_model(export_dir_base=export_dir, serving_input_receiver_fn=serving_input_fn)
    print('Export model to ' + export_dir)


def shuffle_extract(args_opt):
    lines = []
    with open(args_opt.train_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    f.close()
    random.shuffle(lines)

    sample_lines = []
    i = 0
    for line in lines:
        if i < args_opt.eval_line_count:
            sample_lines.append(line)
            i += 1

    out = open(args_opt.test_file, 'w')
    for line in sample_lines:
        out.write(line)
    out.close()


def get_args_opt():
    parser = argparse.ArgumentParser(description='WIND_AND_DEEP')
    parser.add_argument('--output_dir', type=str, required=False, default='models/',
                        help='The location to write checkpoints and export models')
    parser.add_argument('--train_file', type=str, required=False, default='/dataset/criteo/csv_data/large.csv',
                        help='The path of the train data file')
    parser.add_argument('--test_file', type=str, required=False, default='/dataset/criteo/csv_data/eval.csv',
                        help='The path of the test data file')
    parser.add_argument("--model_type", type=str, default="wide_and_deep",
                        help="Valid model types: {'wide', 'deep', 'wide_and_deep'}.")
    parser.add_argument("--batch_size", type=int, default=80000, help="Train data batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16000, help="Eval data batch size")
    parser.add_argument("--train_line_count", type=int, default=8000000,
                        help="Train dataset total lines")
    parser.add_argument("--num_epochs", type=int, default=15, help="Train epoch numbers")
    parser.add_argument("--eval_line_count", type=int, default=800000,
                        help="Eval dataset lines extracted from train file, which are written to test file.")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    print("TensorFlow version {}".format(tf.__version__))
    args_opt = get_args_opt()
    # Prepare eval dataset
    shuffle_extract(args_opt)
    start_time = time.time()
    print('Start train and eval time: ', start_time)
    train_and_eval(args_opt)
    end_time = time.time()
    cost_train_eval_time = end_time - start_time
    print('Train and eval total cost time: ', cost_train_eval_time/60, ' minutes.')
    print('Done all the jobs: data preprocess, train and eval!!!')
