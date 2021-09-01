import os
from flask import Flask, request, jsonify
import json
from easydict import EasyDict as edict
import numpy as np


from src.transforms import _reshape_data
from src.yolo import YOLOV3DarkNet53
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.config import ConfigYOLOV3DarkNet53
from mindspore import Tensor
import mindspore as ms
from src.detection import DetectionEngine


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20*1024*1024


@app.route('/')
def hello_world():
    return 'hello world'


def data_preprocess(img, config):
    img, ori_image_shape = _reshape_data(img, config.test_img_shape)
    img = img.transpose(2, 0, 1)
    return img, ori_image_shape


def yolov3_predict(instance, strategy):
    network = YOLOV3DarkNet53(is_training=False)
    pretrained_ckpt = '/dataset/ckpt-files/shanshui_full/yolov3.ckpt'
    if not os.path.exists(pretrained_ckpt):
        err_msg = "The yolov3.ckpt file does not exist!"
        return {"status": 1, "err_msg": err_msg}
    param_dict = load_checkpoint(pretrained_ckpt)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    load_param_into_net(network, param_dict_new)

    config = ConfigYOLOV3DarkNet53()

    # init detection engine
    args = edict()
    args.ignore_threshold = 0.01
    args.nms_thresh = 0.5
    detection = DetectionEngine(args)

    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    print('Start inference....')
    network.set_train(False)
    ori_image = np.array(json.loads(instance['data']), dtype=instance['dtype'])
    image, image_shape = data_preprocess(ori_image, config)
    prediction = network(Tensor(image.reshape(1, 3, 416, 416), ms.float32), input_shape)
    output_big, output_me, output_small = prediction
    output_big = output_big.asnumpy()
    output_me = output_me.asnumpy()
    output_small = output_small.asnumpy()

    per_batch_size = 1
    detection.detect([output_small, output_me, output_big], per_batch_size,
                     image_shape, config)
    detection.do_nms_for_results()
    out_img = detection.draw_boxes_in_image(ori_image)

    for i in range(len(detection.det_boxes)):
        print("x: ", detection.det_boxes[i]['bbox'][0])
        print("y: ", detection.det_boxes[i]['bbox'][1])
        print("h: ", detection.det_boxes[i]['bbox'][2])
        print("w: ", detection.det_boxes[i]['bbox'][3])
        print("score: ", round(detection.det_boxes[i]['score'], 3))
        print("category: ", detection.det_boxes[i]['category_id'])

    return {
        "status": 0,
        "instance": {
            "shape": list(out_img.shape),
            "dtype": out_img.dtype.name,
            "data": json.dumps(out_img.tolist())
        }
    }


@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    instance = json_data['instance']
    strategy = json_data['strategy']
    res = yolov3_predict(instance, strategy)
    return jsonify(res)


if __name__ == '__main__':
    app.run(host="172.17.0.3", port=8080, debug=True)
