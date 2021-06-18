# Copyright 2021 Huawei Technologies Co., Ltd
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

import sys
import os
import cv2
import requests
import json
import numpy as np
import socket
from src.transforms import _reshape_data
from src.config import ConfigYOLOV3DarkNet53


# 支持的文件格式
IMG_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']  # 集合类型


# 判断文件名是否是我们支持的格式
def is_image(img_path):
    if '.' in img_path and img_path.rsplit('.', 1)[1].lower() in IMG_EXTENSIONS:
        return True
    return False


class Client:
    def __init__(self, host='119.3.225.187', port=80):
        self.host = host
        self.port = port

    def _server_started(self):
        """
        Detect whether the serving server is started or not.
        A bool value of True will be returned if the server is started, else False.
        Returns:
            A bool value of True(if server started) or False(if server not started).
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((self.host, self.port))
            s.shutdown(2)
            return True
        except:
            return False

    def data_preprocess(self, img_path, config):
        img = cv2.imread(img_path, 1)
        img, ori_image_shape = _reshape_data(img, config.test_img_shape)
        img = img.transpose(2, 0, 1)
        return img, ori_image_shape

    def predict(self, img_path, strategy="TOP1_CLASS"):
        if not is_image(img_path):
            raise Exception("The image format does not match `png`, `jpg`, `jpeg` or `gif`.")

        # data preprocess operation
        config = ConfigYOLOV3DarkNet53()
        image, image_shape = self.data_preprocess(img_path, config)
        print(image.shape)

        if not self._server_started() is True:
            print('Server not started at host %s, port %d' % (self.host, self.port))
            sys.exit(0)
        else:
            # Construct the request payload
            payload = {
                'instance': {
                    'shape': image_shape.tolist(),
                    'dtype': image.dtype.name,
                    'data': json.dumps(image.tolist())
                },
                'strategy': strategy
            }
            headers = {'Content-Type': 'application/json'}
            url = 'http://'+self.host+':'+str(self.port)+'/predict'
            res = requests.post(url=url, headers=headers, data=json.dumps(payload))
            res.content.decode("utf-8")
            res_body = res.json()

            if res.status_code != requests.codes.ok:
                print("Request error! Status code: ", res.status_code)
                sys.exit(0)
            elif res_body['status'] != 0:
                print(res_body['err_msg'])
                sys.exit(0)
            else:
                detection = np.array(res_body['detection'])
                img = detection.draw_boxes_in_image(img_path)
                output_img = 'output_' + os.path.basename(img_path).lower()
                output_dir = './'
                cv2.imwrite(os.path.join(output_dir, output_img), img)


if __name__ == '__main__':
    client = Client()
    img_path = "/Users/wewe/Desktop/249.jpeg"
    client.predict(img_path=img_path)
