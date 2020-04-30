The neural nets here were pre-trained on [ImageNet](http://www.image-net.org/).

- `alexnet_finetune.m` was fine-tuned with a small learning rate 10<sup>-4</sup>). As offered by MatLab, the learning rate is multiplied by 20 on fc7.

- in `alexnet_svm.m`, the classifier is an SVM instead of SoftMax trained with SGD above.

- `alexnet_misclass_rate.m` applies many cross-validations on the dataset, every time attributing randomly images to either training or validation set. At each iteration, the number of iterations (tabCountIter) and misclassifications (tabCountMisclass) for each image of the validation set is stored.

- `vgg16` scripts implement fine-tuning of the VGG16 network, with either SVM or SoftMax classifier.
