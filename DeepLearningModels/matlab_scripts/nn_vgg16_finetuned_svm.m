%% Initialization
clear ; close all hidden; clc
% images_from_google/sitting
imds = imageDatastore('/home/klab/vincent/official_backup/images/grayscale3channels/final/drinking_rm40/',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

n_iter = 15;
meanTab = zeros(n_iter,1);
valSize = 31; % number of images from each class in validation set (x2 = total val set) max was 144 for now
valFiles = cell(valSize*2,n_iter);  
for idx = 1:n_iter
    imdsrand = shuffle(imds);
    [imdsValidation,imdsTrain] = splitEachLabel(imdsrand,valSize,'randomized');
    valFiles(1:valSize*2,idx) = imdsValidation.Files;

    % create neural network
    net = vgg16; %vgg16 vgg19 alexnet
    layersTransfer = net.Layers(1:end-3); %2:end-3
    numClasses = 2;
    layers = [
        %imageInputLayer([256 256 3])
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        %fullyConnectedLayer(numClasses)
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
    
    options = trainingOptions('adam', ...
        'MiniBatchSize',32, ...
        'MaxEpochs',12, ...
        'InitialLearnRate',1e-4, ...
        'Verbose',false);
    %{
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',32, ...
        'MaxEpochs',10, ...
        'InitialLearnRate',1e-4, ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',20, ...
        'ValidationPatience',Inf, ...
        'Verbose',false, ...
        'Plots','training-progress');
    %}
    % training
    disp('start training')
    netTransfer = trainNetwork(augimdsTrain,layers,options);
    disp('updated weights')

    % training
    layer = 'fc7';
    featuresTrain = activations(netTransfer,augimdsTrain,layer,'MiniBatchSize',70,'OutputAs','rows');
    featuresTest = activations(netTransfer,augimdsValidation,layer,'MiniBatchSize',70,'OutputAs','rows');

    % train classifier
    classifier = fitcecoc(featuresTrain,imdsTrain.Labels);
    disp('trained classifier, start inference')

    % classify
    YPred = predict(classifier,featuresTest);

    % mean
    meanTab(idx) = mean(YPred == imdsValidation.Labels);
    disp('done inference')
    disp(idx)
end