Follow [Detectron/INSTALL.md](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md) to set up the environment. 
The scripts are run similarly to `tools/infer_simple.py` as described in [Detectron/GETTING\_STARTED.md ](https://github.com/facebookresearch/Detectron/blob/master/GETTING_STARTED.md).

- `im_detect_features.py` returns the features of the region proposals from any category, like `detectron/core/test.py`. It is used in all three activities, <em> drinking</em>, <em> reading</em> or <em> sitting</em>.

- `vis_extract_X.py` returns features, bounding box and keypoints from the category of interest. It works in pair with `infer_simple_extract_X.py`, where X replaces `human`, `reading` or `drinking` respectively. 

- The folder `classify` contains code for classifying the features extracted using Detectron.

#### Extracting the person in the picture

We use the model giving best performance on the COCO Keypoint Detection Task, according to detectron's [model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md). Here, the selected model is `e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml`.

```
python tools/infer_simple_extract_human.py \
	--cfg configs/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml \
	--output-dir demo/output/reading_gray/train/yes/ \
	--image-ext jpg \
	--wts https://dl.fbaipublicfiles.com/detectron/37732318/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_55_09.Lx8H5JVu/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	--thresh 0.4 \
	--kp-thresh 1.3 \
	demo/reading_gray/train/yes/ 
```

The model is applied to all <em> .jpg</em> images in the `demo/reading_gray/train/yes/` directory. The keypoints, bounding box and features of the main person on the image are saved into <em> .pkl</em> files. The main person corresponds to the largest bounding box, among the boxes with a score in the person category that is higher than some threshold (set by --thresh).

<div align="center">
  <img src="example_read_person.png" width="250px" />
  <p>Example output showing the person's keypoints.</p>
</div>

#### Extracting the reading material in the picture

We use the model giving best performance on the COCO Object Detection Task, according to detectron's [model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md). Here, the selected model is `e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml`.

```
python tools/infer_simple_extract_reading.py 
	--cfg configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml \
	--output-dir demo/output/reading_rgb/train/yes/ \
	--image-ext jpg \
	--wts https://dl.fbaipublicfiles.com/detectron/36494496/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml.07_50_11.fkwVtEvg/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	--thresh 0.2 \
	demo/reading_rgb/train/yes/ 
```

<div align="center">
  <img src="example_read_txtbx.png" width="250px" />
  <p>Example output showing segmentation of the book in the picture.</p>
</div>


#### Extracting the beverage in the picture

Same as extracting the reading material, except that we retain the "cup", "wine glass", or "bottle" categories, instead of the "book", "cell phone", "laptop" or "TV" categories in the case of <em> reading</em>.

Use `tools/infer_simple_extract_drinking.py` instead of `tools/infer_simple_extract_reading.py` and adjust the image folder. The model configuration and weights are the same as for <em> reading</em>.
