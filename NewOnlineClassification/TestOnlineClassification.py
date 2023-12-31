from OnlineClassification import RunPythonModel
import scipy.io
import numpy as np

def english_label(label):
    if label == 1:
        return "rock"
    elif label == 2:
        return "paper"
    else:
        return "scissors"

test_mat = scipy.io.loadmat("DataFile/allSubjFiles.mat")
#this is supposed to be isl data
#test data should be filtered data from the preprocessing, datachtimetr
#test_mat1 = preprocesspt2()
test_data = test_mat["dataChTimeTr"]
test_labels = test_mat["labels"].flatten()

# (1400,4) trial
#choosing the first entry in test data
test_trial_raw = test_data[:, :, 0].T

test_trial = np.zeros((4, test_trial_raw.shape[0]))
test_trial = test_trial_raw.T
test_trial_label = test_labels[0]

model_filename = "/Users/yumi/Documents/Neurotech/Owen Aaron Project/Online_Classification/TrainedKNNMODEL.pkl"

rpm = RunPythonModel(model_filename)
result = rpm.get_rps(test_trial)

print(f"Real label: {english_label(test_trial_label)}, result from your model: {english_label(result)}")