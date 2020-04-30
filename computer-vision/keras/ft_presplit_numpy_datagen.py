import os
import cv2
import numpy as np

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

################# GET TRAINING DATA ##################
input_path = "images/cvpr_datasets_final/sitting_gray/train/" #"cvpr_datasets_final/sitting_rgb/train/" "rgb/sitting/presplit1/train/"
print(input_path)
imgsNo, labelsNo, nofiles = getLabeledData(input_path + 'no/', 0) #no/
imgsYes, labelsYes, yesfiles = getLabeledData(input_path + 'yes/', 1)

nofiles = ['no/' + f_ for f_ in nofiles]
yesfiles = ['yes/' + f_ for f_ in yesfiles]

X_train = np.concatenate((imgsNo,imgsYes),axis=0)
y_train = np.concatenate((labelsNo,labelsYes),axis=0)
tr_files = nofiles + yesfiles

np.random.seed(0)
random_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[random_index], y_train[random_index]
train_files = np.asarray(tr_files)[random_index]

################# GET TESTNG DATA ##################
input_path = "images/cvpr_datasets_final/sitting_gray/val/" #"cvpr_datasets_final/sitting_rgb/val/"
imgsNo, labelsNo, nofiles = getLabeledData(input_path + 'no/', 0) #no/
imgsYes, labelsYes, yesfiles = getLabeledData(input_path + 'yes/', 1)

nofiles = ['no/' + f_ for f_ in nofiles]
yesfiles = ['yes/' + f_ for f_ in yesfiles]

X_test = np.concatenate((imgsNo,imgsYes),axis=0)
y_test = np.concatenate((labelsNo,labelsYes),axis=0)
te_files = nofiles + yesfiles

np.random.seed(0)
random_index = np.random.permutation(len(X_test))
X_test, y_test = X_test[random_index], y_test[random_index]
test_files = np.asarray(te_files)[random_index]

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
# fits the model on batches with real-time data augmentation:
BATCH_SIZE = 32
train_flow = datagen.flow(X_train, y_train, BATCH_SIZE, shuffle=False)

test_gen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
test_gen.fit(X_test)
test_flow = test_gen.flow(X_test, y_test, batch_size=30, shuffle=False)


############ MODEL ARCHITECTURE #############
#.inception_resnet_v2 import InceptionResNetV2 || .resnet_v2 import ResNet50V2 || .xception import Xception || .vgg16 import VGG16 || .inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D, AveragePooling2D, Input

input_tensor = Input(shape=(256, 256, 3))
#InceptionResNetV2 || ResNet50V2 || Xception || VGG16 || InceptionV3
base_model = ResNet50V2(weights= 'imagenet', include_top=False, pooling = 'avg', input_tensor=input_tensor)
x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
#x = Dropout(0.5)(x)
predictions = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = base_model.input, outputs = predictions)
# Say not to train ResNet layers
for layer in base_model.layers:
    layer.trainable = True

################# COMPILE MODEL #################
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

model.compile(loss='binary_crossentropy', #categorical_crossentropy  keras.losses.SparseCategoricalCrossentropy()
              optimizer=Adam(lr=1e-4),
              metrics=['binary_accuracy']) #binary_accuracy   keras.metrics.SparseCategoricalAccuracy()

################# TRAINING #################
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
log_folder = 'resnet-log'
NUM_EPOCHS = 6
#STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
cb_checkpointer = ModelCheckpoint(filepath = log_folder + '/best.hdf5', 
                                monitor = 'val_binary_accuracy',  
                                save_best_only = True,
                                mode = 'auto') #
csv_record_train = CSVLogger(log_folder + '/training.log')
history = model.fit(train_flow,
                    epochs=NUM_EPOCHS,
                    validation_data=test_flow,
                    callbacks=[cb_checkpointer,csv_record_train])

################# PREDICTIONS #################
val_summary = model.evaluate(test_flow)
preds = model.predict(test_flow)

############## CSV ##############
def export_csv(pred, filenames, log_folder, metrics_names, val_summary):
    pred_class = np.where(pred < 0.5, 0, 1)
    filenames = np.asarray(filenames) 
    filenames = filenames.reshape((len(filenames), 1))
    conc = np.concatenate((filenames, pred, pred_class), axis=1)

    with open(log_folder+'/pred.csv', 'a') as csvfile:
        csvfile.write("\nMETRICS\n%s:%s "%(metrics_names[0], val_summary[0]))
        csvfile.write("%s:%s\n"%(metrics_names[1], val_summary[1]))
        csvfile.write("FILENAMES\n")
        np.savetxt(csvfile, conc, delimiter=",", fmt='%s')

def export_history(log_folder, history_):
    with open(log_folder  + '/history.csv', 'a') as csvfile:
        for key in history_.history.keys():
            csvfile.write("%s : %s \n\n"%(key, history_.history[key]))

export_csv(preds, test_files, log_folder, 
           model.metrics_names, val_summary)

export_history(log_folder, history)