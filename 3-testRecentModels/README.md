These scripts are to be run in the [Detectron](https://github.com/facebookresearch/Detectron) environment. Place the scripts in the folder "tools" along with the other inference scripts.

## detecting the person in the picture

- \textbf{infer_simple_extract_human.py}    modified script of tools/infer_simple.py. The keypoints and features of the "main" person on the image are saved into pkl files. The "main" person corresponds to the largest bounding box, among the boxes with a score in the person category that is higher than some threshold (set by --thresh).
- \textbf{vis_extract_human.py}    modified script of detectron/utils/vis.py, returning keypoints and features of the "main" person in the image.

<div align="center">
  <img src="example_read_person.pdf" width="250px" />
  <p>Example output showing keypoints on the person.</p>
</div>

## detecting the reading material in the picture

- \textbf{infer_simple_extract_reading.py}   modified script of tools/infer_simple.py. The bounding box and features of the "main" text material on the image are saved into pkl files. The "main" text material corresponds to the largest bounding box, among the boxes with a score that is higher than some threshold (set by --thresh)in one of the following COCO categories: tv, laptop, cell phone or book.
- \textbf{vis_extract_reading.py}   modified script of detectron/utils/vis.py, returning keypoints and features of the "main" text material in the image.

<div align="center">
  <img src="example_read_txtbx.pdf" width="250px" />
  <p>Example output showing segmentation of the book in the picture.</p>
</div>

## extracting features

- \textbf{im_detect_w_features.py}   modified script of detectron/core/test.py such, returning the features of the region proposals. It is called by both infer_simple_extract_human.py and infer_simple_extract_reading.py.
