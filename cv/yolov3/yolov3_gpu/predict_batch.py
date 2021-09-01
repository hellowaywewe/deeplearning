# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""YoloV3 eval."""
import os
import argparse
import datetime

import cv2

import mindspore as ms
import mindspore.context as context
import pandas as pd
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.yolo import YOLOV3DarkNet53
from src.logger import get_logger
from src.config import ConfigYOLOV3DarkNet53
from src.transforms import _reshape_data
from src.detection import DetectionEngine


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser('mindspore coco testing')
    # device related
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: GPU)')
    # dataset related
    parser.add_argument('--image_path', required=True, type=str, default='', help='image file path')
    parser.add_argument('--output_dir', type=str, default='./', help='image file output folder')
    parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')
    # network related
    parser.add_argument('--pretrained', required=True, default='', type=str,
                        help='model_path, local pretrained model to load')
    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='checkpoint save location')
    # detect_related
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
    parser.add_argument('--ignore_threshold', type=float, default=0.7,
                        help='threshold to throw low quality boxes')

    args, _ = parser.parse_known_args()
    return args


def data_preprocess(ori_img, config):
    img, ori_image_shape = _reshape_data(ori_img, config.test_img_shape)
    img = img.transpose(2, 0, 1)
    return img, ori_image_shape


def write_results(results):
    excel_path = './predict_results.xlsx'
    writer = pd.ExcelWriter(excel_path)
    df1 = pd.DataFrame(data=results, columns=['img_name', 'species', 'score'])
    df1.to_excel(writer, 'predict_results', index=False)
    writer.save()


def predict():
    """The function of predict."""
    args = parse_args()

    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=False, device_id=devid)

    # logger
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    args.logger = get_logger(args.outputs_dir, rank_id)

    args.logger.info('Creating Network....')
    network = YOLOV3DarkNet53(is_training=False)

    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained))
    else:
        args.logger.info('{} not exists or not a pre-trained file'.format(args.pretrained))
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(args.pretrained))
        exit(1)

    config = ConfigYOLOV3DarkNet53()
    args.logger.info('testing shape: {}'.format(config.test_img_shape))

    network.set_train(False)
    detection = DetectionEngine(args)

    args.logger.info('Start inference....')

    image_path = args.image_path
    results = []
    for img_dir, dirnames, img_names in os.walk(image_path):
        print("image_path")
        for img_name in img_names:
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # init detection engine
                input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
                img_file = os.path.join(img_dir, img_name)
                # data preprocess operation
                ori_image = cv2.imread(img_file, 1)
                image, image_shape = data_preprocess(ori_image, config)

                prediction = network(Tensor(image.reshape(1, 3, 416, 416), ms.float32), input_shape)
                output_big, output_me, output_small = prediction
                output_big = output_big.asnumpy()
                output_me = output_me.asnumpy()
                output_small = output_small.asnumpy()

                detection.detect([output_small, output_me, output_big], args.per_batch_size,
                                 image_shape, config)
                detection.do_nms_for_results()
                img, res_list = detection.draw_boxes_in_image(ori_image, img_name)

                output_img = 'output_' + os.path.basename(img_file).lower()
                cv2.imwrite(os.path.join(args.output_dir, output_img), img)
                results.extend(res_list)
    write_results(results)


if __name__ == "__main__":
    predict()
