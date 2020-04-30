%% Initialization
clear ; close all; clc

imds = imageDatastore('../../images/grayscale3channels/google/reading',...
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
    inputSize = layers(1).InputSize;
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain); % 'DataAugmentation',imageAugmenter
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

    % training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',10, ...
        'MaxEpochs',5, ...
        'InitialLearnRate',1e-4, ...
        'Verbose',false);

    % training
    netTransfer = trainNetwork(augimdsTrain,layers,options);

    % classify
    [YPred,scores] = classify(netTransfer,augimdsValidation);
    % mean
    meanTab(idx) = mean(YPred == imdsValidation.Labels);
    % std
    stdTab(idx) = std(YPred == imdsValidation.Labels);

    disp('loooooop')
    disp(idx)
    close all hidden;
end

table(meanTab, stdTab)
