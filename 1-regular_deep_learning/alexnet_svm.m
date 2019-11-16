%% Initialization
clear ; close all; clc

imds = imageDatastore('../../images/grayscale3channels/imgs_rmvd/hm_sit/sitting_rmv15_adjusted/',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

n_iter = 5;
meanTab = zeros(n_iter,1);
stdTab =  zeros(n_iter,1);
for idx = 1:n_iter

    imdsrand = shuffle(imds);
    [imdsTrain,imdsValidation] = splitEachLabel(imdsrand,0.9,'randomized');

    % create neural network
    net = alexnet; %vgg16 vgg19
    layersTransfer = net.Layers(2:end-3);
    numClasses = 2;
    layers = [
        imageInputLayer([256 256 3])
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];

    % convert from 256x256 to 227x227
    inputSize = net.Layers(1).InputSize;
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain); % 'DataAugmentation',imageAugmenter
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

    % training
    layer = 'fc7';
    featuresTrain = activations(net,augimdsTrain,layer,'MiniBatchSize',32,'OutputAs','rows');
    featuresTest = activations(net,augimdsValidation,layer,'MiniBatchSize',32,'OutputAs','rows');
    % train classifier
    classifier = fitcecoc(featuresTrain,imdsTrain.Labels);

    % classify
    YPred = predict(classifier,featuresTest);
    % mean
    meanTab(idx) = mean(YPred == imdsValidation.Labels);
    % std
    stdTab(idx) = std(YPred == imdsValidation.Labels);

    disp('loooooop')
    disp(idx)
    close all hidden;
end

table(meanTab, stdTab)
