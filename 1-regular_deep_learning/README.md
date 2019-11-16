For all of these Neural Nets, we use weights pretrained on ImageNet.

# alexnet_finetune.m

Finetuning means that the weight update is small (learning rate 10<sup>-4</sup>). As offered by MatLab, the learning rate is multiplied by 20 on fc7.

# alexnet_svm.m

Classifier is an SVM instead of SoftMax trained with SGD above.

# alexnet_misclass_rate.m

Cross-validation, with many iterations, is run on the dataset, every time attributing randomly images to either training or validation set. 
At each iteration, we store the iteration number and prediction accuracy of each image in the validation set.
The tables tabCountIter and tabCountMisclass, storing these values, are saved.
