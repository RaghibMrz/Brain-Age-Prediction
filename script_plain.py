import tkinter as tk
from tkinter import filedialog
import os, sys, csv, numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
import requests
import json

class BrainAge:
	def __init__(self, master):
		self.home = os.path.dirname(os.path.abspath(sys.argv[0]))

		self.master = master
		master.title("Brain Age Prediction Tool")

		self.canvas = tk.Canvas(self.master, width=400, height=300)
		self.canvas.pack()
		self.promptLabel = tk.Label(self.master, text="Select SynthSeg Segmented Volume File\n Please make sure this is the CSV file returned by SynthSeg")
		self.canvas.create_window(200, 80, window=self.promptLabel)
		self.button = tk.Button(text='Choose file', command=self.getAge)
		self.canvas.create_window(200, 120, window=self.button)
		self.label = tk.Label(self.master, text="")
		self.genButton = tk.Button()
	
		self.age = 0

	def getAge(self):
		self.master.withdraw()
		ext = ''

		# prompt user to load volume file generated using SynthSeg
		path = filedialog.askopenfilename(title = "Select volume file",filetypes = (("CSV Files","*.csv"),))
		fn = os.path.basename(path)
		self.master.deiconify()
		# vol_path = os.path.join(home, './outputs/'+str(filename)+'_vol.csv')

		# get volume file generated
		with open(path, newline='') as f:
			reader = csv.reader(f)
			features = np.array(list(reader))[2,1:]
			features = np.array(list(map(lambda x: float(x), features)))

		# import trained neural network
		ann = tf.keras.models.load_model(os.path.join(self.home, './models/model.h5'))

		# get prediction, output to user
		features = features.reshape(1, 33)
		sc = pickle.load(open('models/scaler.pkl','rb'))
		features = sc.transform(features)

		pred = ann.predict(features)
		self.age = pred[0][0]

		# self.canvas = tk.Canvas(self.master, width=400, height=300)
		# self.canvas.pack()
		self.ageLabel = tk.Label(self.master, text="Brain volume: "+str(fn)+"\nPredicted age: "+str(self.age))
		self.canvas.create_window(200, 180, window=self.ageLabel)


root = tk.Tk()
my_gui = BrainAge(root)
root.mainloop()