# Brain-Age-Prediction
Uses SynthSeg segmented data to predict brain age from T1 weighted MRI scans.

It is recommended that you use a virtual environment to run this script.

Make sure you use Python 3.6+ and have PIP version 21.0.1 (or above)- if you create a python virtual environment you will have to manually upgrade your version of pip to get the correct version of tensorflow. Use the command `pip install pip==21.0.1`

Once you have done that and activated the virtual environment, run the following commands to install the required dependencies and run the script to produce your prediction:

- `pip install -r requirements.txt`
- `python script_plain.py`

The second command should launch an interactive tkinter window which allows you to choose a file, please choose an appropriate volume file returned from a SynthSeg segmentation.
