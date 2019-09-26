from typing import Tuple
import numpy as np


def nms_blend(
        boxes: np.ndarray,
        scores: np.ndarray,
        overlap_threshold: float,
        n_blended: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ Based on nms from
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    which is based on Malisiewicz et al.
    Custom bit is at the end, this allows to calculate score of the box which
    is a blend of the scores.
    """
    pick = []  # picked indexes
    if len(boxes) == 0:
        return boxes[pick], scores[pick]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    out_scores = np.zeros_like(scores)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        matched = np.concatenate([
            [last],
            np.where(overlap > overlap_threshold)[0]])
        # The custom bit: calculate score as the sum of all matched boxes
        out_scores[i] = scores[idxs[matched]].sum() / n_blended
        idxs = np.delete(idxs, matched)

    return boxes[pick], out_scores[pick]
