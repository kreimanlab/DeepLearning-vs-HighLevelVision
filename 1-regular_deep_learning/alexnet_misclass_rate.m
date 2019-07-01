%% Initialization
clear ; close all; clc

imds = imageDatastore('../../../images/grayscale3channels/imgs_rmvd/merged_drink/hm70_ggl40/',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

n_data = sum(imds.countEachLabel{:,2});
tabCountIter = zeros(n_data,1);
tabCountMisclass = zeros(n_data,1);

n_iter = 100;
stats = zeros(n_iter,2);
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

    % settings for data augmentation
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);

    % convert from 256x256 to 227x227
    inputSize = net.Layers(1).InputSize;
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain); % 'DataAugmentation',imageAugmenter
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

    % training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',10, ...
        'MaxEpochs',3, ...
        'InitialLearnRate',1e-4, ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',3, ...
        'ValidationPatience',Inf, ...
        'Verbose',false, ...
        'Plots','training-progress');

    % training
    %netTransfer = trainNetwork(augimdsTrain,layers,options);
    
    % training
    layer = 'fc7';
    featuresTrain = activations(net,augimdsTrain,layer,'MiniBatchSize',32,'OutputAs','rows');
    featuresTest = activations(net,augimdsValidation,layer,'MiniBatchSize',32,'OutputAs','rows');
    % train classifier
    classifier = fitcecoc(featuresTrain,imdsTrain.Labels);

    % classify
    YPred = predict(classifier,featuresTest);
    % mean
    stats(idx,1) = mean(YPred == imdsValidation.Labels);
    % std
    stats(idx,2) = std(YPred == imdsValidation.Labels);
    
    %% count misclassifications
    n_dataVal = sum(imdsValidation.countEachLabel{:,2});
    for j = 1:n_data
        for k = 1:n_dataVal
            if strcmp(imdsValidation.Files{k},imds.Files{j})
                tabCountIter(j) = tabCountIter(j) + 1;
                if YPred(k) ~= imdsValidation.Labels(k)
                    disp('im val iteration')
                    disp(k)
                    tabCountMisclass(j) = tabCountMisclass(j) + 1;
                end
            end
        end
    end
    
    disp('loop')
    disp(idx)
    close all hidden;
end

save('tabCountIter.mat','tabCountIter')
save('tabCountMisclass.mat','tabCountMisclass')