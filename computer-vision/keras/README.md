### Prerequisites

TensorFlow v2.1.0

### Misclassification Rate

The misclassification rate allows to remove biases of our dataset, as described in section 3.3 of our publication.
1. `python misclassification_rate_iterate.py` applies many cross-validations on the dataset. After each iteration, the predictions on a random validation set are written in a csv file.
2. `misclassification_rate_list.py` reads the csv files and sums up the number of misclassifications for each image.

### Fine-tuning

1. `python ft_presplit_numpy_datagen.py` to fine-tune on the training set and validate on the validation set.
2. `python load_inf_numpy_datagen.py` to load the fine-tuned model and apply on the test set.

### Grad-CAM

`python gradcam_loop.py` is adapted from the implementation of [jacobgil](https://github.com/jacobgil/keras-grad-cam), based on [Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391).
