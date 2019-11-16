# same as inference but with flattened images

from PIL import Image
import os
import numpy as np

def getLabeledData(dir_, label):
    imgs = []
    files = [f for f in os.listdir(
        dir_) if os.path.isfile(os.path.join(dir_, f))]
    for index,file in enumerate(files):
        im = Image.open(dir_+file)
        im.load()
        data = np.asarray(im, dtype="float32")
        imgs.append(data)
        
    labelVec = np.ones([len(imgs),1], dtype="float32") * label
    
    return np.asarray(imgs), labelVec

input_path = "/train_dir/"
imgsNo, labelsNo = getLabeledData(input_path + 'no/', 0)
imgsYes, labelsYes = getLabeledData(input_path + 'yes/', 1)

X = np.concatenate((imgsNo,imgsYes),axis=0)
Y = np.concatenate((labelsNo,labelsYes),axis=0)

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load model_head
image_size = 256
conv = ResNet50(weights='imagenet', 
                include_top=False,
                input_shape=(image_size, image_size, 3))
model_head = Sequential()
model_head.add(conv)
model_head.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['acc'])


from sklearn.model_selection import KFold
from tensorflow.python.keras.utils import to_categorical
from sklearn.utils import shuffle

kf = KFold(n_splits=3, shuffle=True)
epochs = 20############75
hists = []

for train_index, test_index in kf.split(X):
    print("getting data")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)
    
    # shuffle
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # get vectors
    print("getting vectors")
    train_vectors = model_head.predict(X_train)
    test_vectors = model_head.predict(X_test)
    print(train_vectors.shape, test_vectors.shape)

    # build model

    # Create the model_tail
    model_tail = Sequential()
    # model.add(layers.Flatten())
    model_tail.add(GlobalAveragePooling2D(input_shape=(8,8,2048)))    
    model_tail.add(Dense(512, activation='relu'))
    #model_tail.add(Dropout(0.5))
    #model_tail.add(Dense(512, activation='relu'))
    #model_tail.add(Dropout(0.5))
    model_tail.add(Dense(2, activation='softmax'))

    # generator
    train_datagen = ImageDataGenerator(rescale=1/256)
    validation_datagen = ImageDataGenerator(rescale=1/256)
    # flow generator
    print('flowing generator')
    train_flow = train_datagen.flow(train_vectors, y_train, batch_size=32)
    validation_flow = validation_datagen.flow(test_vectors, y_test, batch_size=32)


    # compile model
    print("compiling")
    model_tail.compile(loss='categorical_crossentropy',
	              optimizer=Adam(lr=1e-3),   # optimizers.Adadelta() optimizers.SGD(lr=1e-4)
	              metrics=['acc'])

    # fits the model
    history = model_tail.fit_generator(
	    train_flow,
	    steps_per_epoch=len(X_train) / 32,
	    shuffle=True,
	    epochs=epochs,
	    validation_data=validation_flow,
	    validation_steps=len(X_test) / 32,
	    verbose=1)

    # store histories
    hists.append(history)

#################################
############## CSV ##############

def export_history(log_folder, history_):
    with open(log_folder, 'a') as csvfile:
        for key in history_.history.keys():
            csvfile.write("%s : %s \n\n"%(key, history_.history[key]))

for i,history in enumerate(hists):
    log_folder = 'HTresnet/history' + str(i) + '.csv'
    export_history(log_folder, history)
