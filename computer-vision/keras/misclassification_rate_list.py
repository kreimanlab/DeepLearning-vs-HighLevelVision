# list all the images into list_files.csv 
import os

def list_imgs(dir_, clss):
	all_files = os.listdir(dir_ + clss) 
	with open(dir_ + 'list_files.csv', 'a') as f1:
		for line in all_files:
			f1.write(clss + line + ',0,0\n')

im_dir = "images/rgb/sitting/full/"
list_imgs(im_dir, "yes/")
list_imgs(im_dir, "no/")

# read every predictions file pred(i).csv,
# each image in pred(i).csv gets total+=1
# and misclass+=1 if it was misclassified
pred_folder = "misclass/"
pred_files = os.listdir(pred_folder) 

for i in range(len(pred_files)):
	print("pred file", i)
	with open(pred_folder + 'pred' + str(i) + '.csv','r') as f2:
	    for line in f2:
	    	filename = line.split(',')[0]
	    	if line.split('/')[0] == 'no':
	    		clss = str(0)
	    	else:
	    		clss = str(1)
	    	pred = line.split(',')[2].split('\n')[0]
	    	imgs = []
	    	with open(im_dir + 'list_files.csv', 'r') as f3:
	    		imgs = f3.readlines()
	    		for j,im_ in enumerate(imgs):
	    			if filename == im_.split(',')[0]:
	    				if clss != pred:
	    					misclass = int(im_.split(',')[1]) + 1
	    				else:
	    					misclass = int(im_.split(',')[1])
	    				
	    				total = int(im_.split(',')[2]) + 1
	    				imgs[j] = filename + ',' + str(misclass) + ',' + str(total) + '\n'

	    	with open(im_dir + 'list_files.csv', 'w') as f4:
	    		for j,im_ in enumerate(imgs):
	    			f4.write(im_)
