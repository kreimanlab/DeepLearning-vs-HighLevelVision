# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""corresponds to the functions im_detect_all and box_results_with_nms_and_limit
   from detectron/core/test.py except differences that are indicated by #ADDED  """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import logging
import numpy as np

from caffe2.python import core
from caffe2.python import workspace
import pycocotools.mask as mask_util

from detectron.core.config import cfg
from detectron.utils.timer import Timer
import detectron.core.test_retinanet as test_retinanet
import detectron.modeling.FPN as fpn
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.image as image_utils
import detectron.utils.keypoints as keypoint_utils
#ADDED
import detectron.core.test as im_detect

logger = logging.getLogger(__name__)


def im_detect_w_features_func(model, im, box_proposals, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    # Handle RetinaNet testing separately for now
    if cfg.RETINANET.RETINANET_ON:
        cls_boxes = test_retinanet.im_detect_bbox(model, im, timers)
        return cls_boxes, None, None

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores, boxes, im_scale = im_detect.im_detect_bbox_aug(model, im, box_proposals)
    else:
        scores, boxes, im_scale = im_detect.im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
        )
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    #ADDED cls_feats from box_results_with_nms_and_limit
    scores, boxes, cls_boxes, cls_feats = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            masks = im_detect.im_detect_mask_aug(model, im, boxes)
        else:
            masks = im_detect.im_detect_mask(model, im_scale, boxes)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = im_detect.segm_results(
            cls_boxes, masks, boxes, im.shape[0], im.shape[1]
        )
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            heatmaps = im_detect.im_detect_keypoints_aug(model, im, boxes)
        else:
            heatmaps = im_detect.im_detect_keypoints(model, im_scale, boxes)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = im_detect.keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None

    #ADDED return cls_feats
    return cls_boxes, cls_segms, cls_keyps, cls_feats


def box_results_with_nms_and_limit(scores, boxes):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    #ADDED fetch ROI features
    roi_features = workspace.FetchBlob(core.ScopedName('roi_feat'))
    #ADDED initialize list cls_feats to store ROI features w respect to their most likely classes
    cls_feats = [[] for _ in range(num_classes)]
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        #ADDED keep only features whose scores > cfg.TEST.SCORE_THRESH)[0]
        feats_j = roi_features[inds]
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
            #ADDED non-maximum suppression
            cls_feats[j] = feats_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets 

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
                #ADDED
                cls_feats[j] = cls_feats[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    #ADDED return cls_feats
    return scores, boxes, cls_boxes, cls_feats

