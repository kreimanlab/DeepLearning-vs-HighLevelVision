# Deep-Learning-vs-High-Level-Vision

This project can be summarized in three steps.

#### 1 - regular_deep_learning

The regular models here refer to AlexNet and VGG16. These two models remain the foundation of the most recent Deep Learning models and still reach among the best accuracies on classification tasks such as the ImageNet challenge.

The first step of the project consists in building datasets that are difficult to classify for AlexNet and VGG16. We start from a large dataset, with images from the web and taken by ourselves. We run many cross-validations over the dataset such that images can be ordered by how easily they are classified. The "easy" images are then removed to obtain a dataset with accuracy close to 50% with VGG16.

#### 2 - psiturk

Secondly, images are shown to human participants during a psychophysics experiment using the psiTurk interface. This step shows that images are easily classified by human subjects.

#### 3 - high_level_models

High-Level models here refer in particular to [Detectron](https://github.com/facebookresearch/Detectron) and Densereg. These models are the closest to an $\textit{interpretation}$ of the image since they extract the mask and coordinates of elements like a person, a book, a beverage.
