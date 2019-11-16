# Fine-tuning all layers of ResNet

from tensorflow.python.keras.applications import ResNet50V2
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D, AveragePooling2D, Input
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pickle 
import numpy as np

NUM_CLASSES = 2

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
#patience: number of epochs with no improvement after which training will be stopped
NUM_EPOCHS = 9
EARLY_STOP_PATIENCE = 6

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_VALIDATION = 1 # because predict_generator

train_data_dir = '/train_dir/'
log_folder = 'resnetV2-scratch'

#################################
###### MODEL ARCHITECTURE #######

input_tensor = Input(shape=(256, 256, 3))
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

#################################
######### COMPILE MODEL #########

model.compile(loss='binary_crossentropy', #binary_crossentropycategorical_crossentropy
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])

# preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)
# Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the inputs to nonlinear activation functions
# Batch Normalization helps in faster convergence
data_generator = ImageDataGenerator(rescale=1./255,
                                    preprocessing_function=preprocess_input,
                                    validation_split=0.1,
                                    horizontal_flip=True,
                                    width_shift_range=[-30,30])
IMAGE_RESIZE = 256
train_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE_TRAINING,
    class_mode="binary", #binary
    shuffle=True,
    seed=42,
    subset='training'
)
# confirm the iterator works
batchX, batchy = train_generator.next()

valid_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE_TRAINING,
    class_mode="binary",
    shuffle=True,
    seed=42,
    subset='validation'
)

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = log_folder + '/best.hdf5', 
                                monitor = 'val_loss', 
                                save_best_only = True, 
                                mode = 'auto')
csv_record_train = CSVLogger(log_folder + '/training.log')
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#validation_steps or steps=STEP_SIZE_VALID -> the last n samples (<batch_size) of generator would be ignored
#also it's optional for Sequence 
fit_history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    epochs=NUM_EPOCHS,
                    callbacks=[cb_checkpointer, cb_early_stopper, csv_record_train]
)

print(fit_history.history.keys())

val_summary = model.evaluate_generator(generator=valid_generator)
valid_generator.reset()
preds = model.predict_generator(valid_generator, 
                            verbose=1)

#################################
############## CSV ##############

def export_csv(pred, filenames, log_folder, class_indices, metrics_names, val_summary):
    pred_class = np.argmax(pred,axis=1)
    pred_class = pred_class.reshape((len(pred_class), 1))
    filenames = np.asarray(filenames)
    filenames = filenames.reshape((len(pred_class), 1))
    conc = np.concatenate((filenames, pred, pred_class), axis=1)

    with open(log_folder+'/pred.csv', 'a') as csvfile:
        csvfile.write("CLASS INDICES\n")
        for key in class_indices.keys():
            csvfile.write("%s:%s "%(key,class_indices[key]))
        
        csvfile.write("\nMETRICS\n%s:%s "%(metrics_names[0], val_summary[0]))
        csvfile.write("%s:%s\n"%(metrics_names[1], val_summary[1]))
        csvfile.write("FILENAMES\n")
        np.savetxt(csvfile, conc, delimiter=",", fmt='%s')

def export_history(log_folder, history_):
    with open(log_folder  + '/history.csv', 'a') as csvfile:
        for key in history_.history.keys():
            csvfile.write("%s : %s \n\n"%(key, history_.history[key]))

export_csv(preds, valid_generator.filenames, log_folder, 
           valid_generator.class_indices, model.metrics_names, val_summary)

export_history(log_folder, fit_history)

#################################
############ PICKLE #############

with open(log_folder + '/trainHistoryDict', 'wb') as f:
    pickle.dump([val_summary,preds], f) 
