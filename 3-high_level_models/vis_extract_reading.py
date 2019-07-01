#!/usr/bin/env python

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

"""This is a modified version of function vis_one_image from detectron/utils/vis.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import numpy as np 

import detectron.utils.env as envu 
envu.set_up_matplotlib() 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon 

import pycocotools.mask as mask_util

import detectron.utils.keypoints as keypoint_utils
from detectron.utils.colormap import colormap 
from detectron.utils.vis import convert_from_cls_format, kp_connections, get_class_string

#ADAPTED from vis.py
def vis_extract_func(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, cls_feats=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf', out_when_no_box=False):
    """Visual debugging of detections."""
    #ADDED declare variables to return
    textbox_assigned = 0
    textbox_feats = None
    one_human_assigned = 0
    human_feats = None
    textbox = None
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    #ADDED similar to convert_from_cls_format, but for feats_list
    feats_list = [b for b in cls_feats if len(b) > 0]
    if len(feats_list) > 0:
        feats = np.concatenate(feats_list)
    else:
        feats = None

    if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh) and not out_when_no_box:
        return None, None, 0

    dataset_keypoints, _ = keypoint_utils.get_keypoints()

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    if boxes is None:
        sorted_inds = [] # avoid crash when 'boxes' is None
    else:
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        #ADDED the text material can be either: tv (i=63), laptop (64), cell phone (68), book (74)
        is_txtmtrl = classes[i]==63 or classes[i]==64 or classes[i]==68 or classes[i]==74
        if is_txtmtrl and not textbox_assigned:
            textbox_feats = feats[i]
            normbbox = bbox/256
            textbox_xmid = (normbbox[2] + normbbox[0]) / 2
            textbox_ymid = (normbbox[3] + normbbox[1]) / 2
            textbox = np.concatenate((normbbox,textbox_xmid,textbox_ymid), axis=None)

            # show box (off by default)
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False, edgecolor='g',
                              linewidth=0.5, alpha=box_alpha))

            if show_class:
                ax.text(
                    bbox[0], bbox[1] - 2,
                    get_class_string(classes[i], score, dataset),
                    fontsize=3,
                    family='serif',
                    bbox=dict(
                        facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                    color='white')

            # show mask
            if segms is not None and len(segms) > i:
                img = np.ones(im.shape)
                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                mask_color_id += 1

                w_ratio = .4
                for c in range(3):
                    color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
                for c in range(3):
                    img[:, :, c] = color_mask[c]
                e = masks[:, :, i]

                _, contour, hier = cv2.findContours(
                    e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

                for c in contour:
                    polygon = Polygon(
                        c.reshape((-1, 2)),
                        fill=True, facecolor=color_mask,
                        edgecolor='w', linewidth=1.2,
                        alpha=0.5)
                    ax.add_patch(polygon)

            textbox_assigned = 1

        #ADDED human features are extracted, although we prefer to use "infer_simple_extract_human" for that
        if classes[i]==1 and not one_human_assigned:
            human_feats = feats[i]
            one_human_assigned = 1

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

                if kps[2, i2] > kp_thresh:
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)

    if textbox_assigned:
        output_name = os.path.basename(im_name) + '.' + ext
        fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
        plt.close('all')

    return textbox, textbox_feats, textbox_assigned, human_feats, one_human_assigned