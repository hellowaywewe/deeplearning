import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        print(clusters)
        last_clusters = nearest_clusters

    return clusters


class AnchorKmeans(object):
    """
    K-means clustering on bounding boxes to generate anchors
    """
    def __init__(self, k, max_iter=300, random_seed=None):
        self.k = k
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.n_iter = 0
        self.anchors_ = None
        self.labels_ = None
        self.ious_ = None

    def fit(self, boxes):
        """
        Run K-means cluster on input boxes.
        :param boxes: 2-d array, shape(n, 2), form as (w, h)
        :return: None
        """
        assert self.k < len(boxes), "K must be less than the number of data."

        # If the current number of iterations is greater than 0, then reset
        if self.n_iter > 0:
            self.n_iter = 0

        np.random.seed(self.random_seed)
        n = boxes.shape[0]

        # Initialize K cluster centers (i.e., K anchors)
        self.anchors_ = boxes[np.random.choice(n, self.k, replace=True)]

        self.labels_ = np.zeros((n,))

        while True:
            self.n_iter += 1

            # If the current number of iterations is greater than max number of iterations , then break
            if self.n_iter > self.max_iter:
                break

            self.ious_ = self.iou(boxes, self.anchors_)
            distances = 1 - self.ious_
            cur_labels = np.argmin(distances, axis=1)

            # If anchors not change any more, then break
            if (cur_labels == self.labels_).all():
                break

            # Update K anchors
            for i in range(self.k):
                self.anchors_[i] = np.mean(boxes[cur_labels == i], axis=0)

            self.labels_ = cur_labels

    @staticmethod
    def iou(boxes, anchors):
        """
        Calculate the IOU between boxes and anchors.
        :param boxes: 2-d array, shape(n, 2)
        :param anchors: 2-d array, shape(k, 2)
        :return: 2-d array, shape(n, k)
        """
        # Calculate the intersection,
        # the new dimension are added to construct shape (n, 1) and shape (1, k),
        # so we can get (n, k) shape result by numpy broadcast
        w_min = np.minimum(boxes[:, 0, np.newaxis], anchors[np.newaxis, :, 0])
        h_min = np.minimum(boxes[:, 1, np.newaxis], anchors[np.newaxis, :, 1])
        inter = w_min * h_min

        # Calculate the union
        box_area = boxes[:, 0] * boxes[:, 1]
        anchor_area = anchors[:, 0] * anchors[:, 1]
        union = box_area[:, np.newaxis] + anchor_area[np.newaxis]

        return inter / (union - inter)

    def avg_iou(self):
        """
        Calculate the average IOU with closest anchor.
        :return: None
        """
        return np.mean(self.ious_[np.arange(len(self.labels_)), self.labels_])