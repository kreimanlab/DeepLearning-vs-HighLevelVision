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

################# GET DATA ##################
input_path = "cvpr_datasets_final/reading_gray/val/" 
print(input_path)
imgsNo, labelsNo, nofiles = getLabeledData(input_path + 'no/', 0) 
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_gen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
test_gen.fit(X_test)
test_flow = test_gen.flow(X_test, y_test, batch_size=60, shuffle=False)

################# LOAD MODEL #################
from tensorflow.keras.models import load_model
weights_folder = 'resnet-log'
model = load_model(weights_folder + '/reading_gray/best.hdf5')

################# COMPILE MODEL #################
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import Accuracy

model.compile(loss='binary_crossentropy', 
              optimizer=Adam(lr=1e-4),
              metrics=['binary_accuracy']) 

################# PREDICTIONS #################
val_summary = model.evaluate(test_flow)
print('VAL SUM')
print(val_summary)
preds = model.predict(test_flow) 

############## CSV ##############

log_folder = 'resnet-log'
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

export_csv(preds, test_files, log_folder, 
           model.metrics_names, val_summary)
