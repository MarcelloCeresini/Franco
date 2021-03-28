import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import numpy as np

# Specify the model and input file
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model_edgetpu.tflite')
input_file = os.path.join(script_dir, "input.npy")
label_file = os.path.join(script_dir, "labels.txt")  # semplicemente txt con elenco di labels

# transform input_file into correct input for the model (stft padding ecc)
input_data = np.load(input_file)

# Load the TFLite model and allocate tensors.
interpreter = edgetpu.make_interpreter("model_edgetpu.tflite")
interpreter.allocate_tensors()

# Run the inference
common.set_input(interpreter, input_data)
interpreter.invoke()  # inference is finished here
result = classify.get_classes(interpreter, top_k=1)
# all_acceptable_results = classify.get_classes(interpreter, score_threshold=0.7)

# Get the results
labels = dataset.read_label_file(label_file)
result_label = labels.get(result.id, result.id)  # get returns the value associated with the key OR the second param
final = (result_label, result.score)
# final = []
# for r in all_acceptable_results:
#     final = final.append(labels.get(r.id, r.id), r.score)

# Do something with the results
print(final)
# for f in final:
#     print(f)
