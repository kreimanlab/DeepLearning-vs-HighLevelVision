import os
import cv2
import numpy as np
import psutil

def getLabeledData(dir_, label):
	files = [f for f in os.listdir(
		dir_) if os.path.isfile(os.path.join(dir_, f))]
	im_h,im_w,chanels = 256,256,3
	dataset = np.ndarray(shape=(len(files), im_h, im_w, chanels),
	                     dtype=np.float32)

	for i,file in enumerate(files):
		im = cv2.imread(dir_ + file)
		dataset[i] = im

	labelVec = np.ones([len(files),1], dtype="float32") * label
	return dataset, labelVec, files

for iter_ in range(700):
    from tensorflow.keras.backend import clear_session
    clear_session()

    ################# GET DATA ##################
    input_path = "rgb/sitting/full/"
    print(input_path)
    imgsNo, labelsNo, nofiles = getLabeledData(input_path + 'no/', 0) #no/
    imgsYes, labelsYes, yesfiles = getLabeledData(input_path + 'yes/', 1)

    nofiles = ['no/' + f_ for f_ in nofiles]
    yesfiles = ['yes/' + f_ for f_ in yesfiles]
    X = np.concatenate((imgsNo,imgsYes),axis=0)
    y = np.concatenate((labelsNo,labelsYes),axis=0)
    files = nofiles + yesfiles

    # record memory usage at each iteration
    with open('var.txt', 'a') as f2:
        process = psutil.Process(os.getpid())
        f2.write(str(process.memory_info().rss) + " in bytes \n")
        
    #np.random.seed(0)
    random_index = np.random.permutation(len(X))
    X, y = X[random_index], y[random_index]
    files = np.asarray(files)[random_index]

    percent = 0.70
    split_tr_te = int(len(X) * percent)
    X_train, y_train = X[:split_tr_te], y[:split_tr_te]
    X_test, y_test = X[split_tr_te:], y[split_tr_te:]
    test_files = np.asarray(files)[split_tr_te:]

    ################# Feature Normalization ##################
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    # fits the model on batches with real-time data augmentation
    BATCH_SIZE = 32
    train_flow = datagen.flow(X_train, y_train, BATCH_SIZE, shuffle=False)

    test_gen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
    test_gen.fit(X_test)
    test_flow = test_gen.flow(X_test, y_test, batch_size=16, shuffle=False)


    ############ MODEL ARCHITECTURE #############
    #.inception_resnet_v2 import InceptionResNetV2  .resnet_v2 import ResNet50V2   .xception import Xception
    from tensorflow.python.keras.applications.xception import Xception
    from tensorflow.python.keras.models import Sequential, Model
    from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D, AveragePooling2D, Input

    input_tensor = Input(shape=(256, 256, 3))
    #InceptionResNetV2   ResNet50V2   Xception
    base_model = Xception(weights= 'imagenet', include_top=False, pooling = 'avg', input_tensor=input_tensor)
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1, activation= 'sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    # Say not to train ResNet layers
    for layer in base_model.layers:
        layer.trainable = True

    ################# COMPILE MODEL #################
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop

    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(lr=1e-4),
                  metrics=['binary_accuracy']) 

    ################# TRAINING #################
    from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    log_folder = 'misclass'
    NUM_EPOCHS = 2

    cb_checkpointer = ModelCheckpoint(filepath = log_folder + '/best.hdf5', 
                                    monitor = 'val_binary_accuracy',  
                                    save_best_only = True,
                                    mode = 'auto')
    csv_record_train = CSVLogger(log_folder + '/training.log')
    history = model.fit(train_flow,
                        epochs=NUM_EPOCHS,
                        validation_data=test_flow,
                        callbacks=[cb_checkpointer]) 

    ################# PREDICTIONS #################
    val_summary = model.evaluate(test_flow)
    preds = model.predict(test_flow) 

    ############## CSV ##############
    def export_csv(pred, filenames, log_folder, metrics_names, val_summary, iter_):
        pred_class = np.where(pred < 0.5, 0, 1)
        
        filenames = np.asarray(filenames) 
        filenames = filenames.reshape((len(filenames), 1))

        conc = np.concatenate((filenames, pred, pred_class), axis=1)

        with open(log_folder+'/pred' + str(iter_) + '.csv', 'a') as csvfile:
            np.savetxt(csvfile, conc, delimiter=",", fmt='%s')

    def export_history(log_folder, history_):
        with open(log_folder  + '/history.csv', 'a') as csvfile:
            for key in history_.history.keys():
                csvfile.write("%s : %s \n\n"%(key, history_.history[key]))

    export_csv(preds, test_files, log_folder, 
               model.metrics_names, val_summary, iter_)

    export_history(log_folder, history)