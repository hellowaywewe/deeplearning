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
import base64
import sys
import os
from io import BytesIO

import cv2
import requests
import json
import numpy as np
import socket


# 支持的文件格式
from PIL import Image

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

    def data_preprocess(self, img_path):
        img = cv2.imread(img_path, 1)
        return np.array(img)

    def predict(self, img_path, strategy="TOP1_CLASS"):
        if not is_image(img_path):
            raise Exception("The image format does not match `png`, `jpg`, `jpeg` or `gif`.")

        if not self._server_started() is True:
            print('Server not started at host %s, port %d' % (self.host, self.port))
            sys.exit(0)
        else:
            image = self.data_preprocess(img_path)
            # Construct the request payload
            payload = {
                'instance': {
                    'shape': list(image.shape),
                    'dtype': image.dtype.name,
                    'data': json.dumps(image.tolist())
                },
                'strategy': strategy
            }
            headers = {'Content-Type': 'application/json'}
            url = 'http://'+self.host+':'+str(self.port)+'/predict'
            res = requests.post(url=url, headers=headers, data=json.dumps(payload))
            res.content.decode("utf-8")
            print(res.status_code)
            res_body = res.json()

            if res.status_code != requests.codes.ok:
                print("Request error! Status code: ", res.status_code)
                sys.exit(0)
            elif res_body['status'] != 0:
                print(res_body['err_msg'])
                sys.exit(0)
            else:
                instance = res_body['instance']
                res_data = np.array(json.loads(instance['data']))
                output_img = 'output_' + os.path.basename(img_path).lower()
                output_dir = './'
                cv2.imwrite(os.path.join(output_dir, output_img), res_data)

    def getv_pic(self, filepath):
        f = open(filepath, 'rb')  # 第一个参数图像路径
        img_b64encode = base64.b64encode(f.read())
        img_b64decode = base64.b64decode(img_b64encode)
        f.close()
        return img_b64decode

    def base64tonumpy(self, img_data):
        np_arr = np.fromstring(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
        return np.array(img)


if __name__ == '__main__':
    client = Client()
    # img_path = "/Users/wewe/Downloads/shanshui/data/pic/510.JPG"
    img_path = "/Users/wewe/Desktop/0.jpeg"
    # client.predict(img_path=img_path)
    img_data = client.getv_pic(img_path)
    image = client.base64tonumpy(img_data)
    print(image.shape)

    instance = {
        'shape': list(image.shape),
        'dtype': image.dtype.name,
        'data': json.dumps(image.tolist())
    }

    data = client.data_preprocess(img_path)

    img_np = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output_buffer = BytesIO()
    img_np.save(output_buffer, format="JPEG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    print(str(base64_str, encoding="utf-8"))



