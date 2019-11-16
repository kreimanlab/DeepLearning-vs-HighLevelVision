%% Initialization
clear ; close all hidden; clc
% images_from_google/sitting
imds = imageDatastore('../../images/grayscale3channels/new/reading-wFullMerged/',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

n_iter = 1;
stats = zeros(n_iter,2);
for idx = 1:n_iter

    imdsrand = shuffle(imds);
    [imdsTrain,imdsValidation] = splitEachLabel(imdsrand,0.9,'randomized');

    % create neural network
    net = vgg16; %vgg16 vgg19 alexnet
    layersTransfer = net.Layers(1:end-3); %2:end-3
    numClasses = 2;
    layers = [
        %imageInputLayer([256 256 3])
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];

    % settings for data augmentation
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);

    % convert from 256x256 to 227x227
    inputSize = net.Layers(1).InputSize(1:2); % [256 256]
    augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain); % 'DataAugmentation',imageAugmenter
    augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);

    % training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ...
        'MaxEpochs',1, ...
        'InitialLearnRate',1e-4, ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',20, ...
        'ValidationPatience',Inf, ...
        'Verbose',false, ...
        'Plots','training-progress');

    % training
    netTransfer = trainNetwork(augimdsTrain,layers,options);

    % classify % CAUSE CRASH 
    [YPred,scores] = classify(netTransfer,augimdsValidation);
    % classify
    %[YPred,scores] = classify(netTransfer,augimdsValidation);

    % mean
    %stats(idx,1) = mean(YPred == imdsValidation.Labels);
    % std
    %stats(idx,2) = std(YPred == augimdsValidation.Labels);

    disp('loooooop')
    disp(idx)
    %close all hidden;
    
end