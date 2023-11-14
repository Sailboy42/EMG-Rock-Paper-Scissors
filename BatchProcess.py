import os
import numpy as np

# Assuming you have a function preprocess_data defined elsewhere
# that performs the same data preprocessing as in your MATLAB code

# Initialize variables
dataChTimeTr = np.empty((0, num_channels, num_timepoints))  # Assuming num_channels and num_timepoints are known
labels = np.empty(0, dtype=int)

# Iterate over selected files
for f in selected_files:
    # Load the data
    file_path = os.path.join(data_file_names[f].folder, data_file_names[f].name)
    loaded_data = np.load(file_path, allow_pickle=True).item()  # Assuming the file contains a dictionary
    
    # Extract lsl_data and marker_data
    lsl_data = loaded_data['lsl_data']
    marker_data = loaded_data['marker_data']

    # Preprocess the data
    epoch_data, gesture_list = preprocess_data(lsl_data, marker_data)

    # Check lengths
    if epoch_data.shape[2] != len(gesture_list):
        raise ValueError("Labels don't match the trials")

    # Concatenate data along the 3rd dimension
    dataChTimeTr = np.concatenate((dataChTimeTr, epoch_data), axis=2)
    labels = np.concatenate((labels, gesture_list), axis=0)

# Now dataChTimeTr contains the concatenated data and labels contains the concatenated labels
