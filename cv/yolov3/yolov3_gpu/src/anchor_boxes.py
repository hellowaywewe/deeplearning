import glob
import xml.etree.ElementTree as ET
import numpy as np
from anchor_kmeans import AnchorKmeans, kmeans, avg_iou


def load_dataset(xml_path):
    dataset_boxes = []
    for xml_file in glob.glob("{}/*xml".format(xml_path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext("size/height"))
        width = int(tree.findtext("size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin"))
            ymin = int(obj.findtext("bndbox/ymin"))
            xmax = int(obj.findtext("bndbox/xmax"))
            ymax = int(obj.findtext("bndbox/ymax"))
            # print("(xmin, ymin): ({}, {}), (xmax, ymax): ({}, {}).".format(xmin, ymin, xmax, ymax))

            if xmin == xmax or ymin == ymax:
                print(xml_file)
                continue

            box_width = xmax - xmin
            box_height = ymax - ymin
            width_norm = np.float64(box_width / width)
            height_norm = np.float64(box_height / height)
            dataset_boxes.append([width_norm, height_norm])
            # print("(width_norm, height_norm): ({}, {}).".format(width_norm, height_norm))
    return np.array(dataset_boxes)


if __name__ == '__main__':
    annotations_xml_path = "/Users/wewe/Downloads/shanshui/data/txt"
    # 9个anchor box,分别有3个small, 3个medium, 3个large
    CLUSTERS = 9
    dataset_boxes = load_dataset(annotations_xml_path)
    # print("anchor boxes shape: {}".format(dataset_boxes.shape))

    out = kmeans(dataset_boxes, k=CLUSTERS)
    print("Accuracy: {:.2f}%".format(avg_iou(dataset_boxes, out) * 100))
    print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))
    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    # print("Ratios:\n {}".format(sorted(ratios)))


# model = AnchorKmeans(k=CLUSTERS)
    # model.fit(dataset_boxes)
    # avg_iou = model.avg_iou()
    # print("K = {}, AVG_IOU = {:.4f}".format(CLUSTERS, avg_iou))

    # for k in range(2, 11):
    #     model = AnchorKmeans(k, random_seed=333)
    #     model.fit(dataset_boxes)
    #     avg_iou = model.avg_iou()
    #     print("K = {}, AVG_IOU = {:.4f}".format(k, avg_iou))
