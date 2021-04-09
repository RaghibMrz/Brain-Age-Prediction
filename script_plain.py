import tkinter as tk
from tkinter import filedialog
import os, sys, csv, numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
import requests
import json

root = tk.Tk()
home = os.path.dirname(os.path.abspath(sys.argv[0]))

def getAge():
	root.withdraw()

	# prompt user to load volume file generated using SynthSeg
	path = filedialog.askopenfilename()

	# vol_path = os.path.join(home, './outputs/'+str(filename)+'_vol.csv')

	# get volume file generated
	with open(path, newline='') as f:
	    reader = csv.reader(f)
	    features = np.array(list(reader))[2,1:]
	    features = np.array(list(map(lambda x: float(x), features)))

	# import trained neural network
	ann = tf.keras.models.load_model(os.path.join(home, './models/model.h5'))

	# get prediction, output to user
	features = features.reshape(1, 33)
	sc = pickle.load(open('models/scaler.pkl','rb'))
	features = sc.transform(features)

	pred = ann.predict(features)

	print("\n\n\n")
	print("Prediction: ")
	print(pred[0][0])
	exit()


canvas = tk.Canvas(root, width=400, height=300)
canvas.pack()
promptLabel = tk.Label(root, text="Select SynthSeg Segmented Volume File")
canvas.create_window(200, 80, window=promptLabel)
button = tk.Button(text='Enter', command=getAge)
canvas.create_window(200, 160, window=button)
label = tk.Label(root, text="")
genButton = tk.Button()
canvas.mainloop() 