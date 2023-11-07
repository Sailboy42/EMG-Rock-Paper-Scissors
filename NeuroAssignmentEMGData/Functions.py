import pickle 



def save(file, data):
    '''
    Just saves the data

    Parameters:
    file - path where data is saveed
    data - data that will be saved to file

    Returns:
    a saved data file

   '''

    if not file.endswith(".pkl"): 
        file += ".pkl"
    with open(file, "wb") as f: #ensures that file is properly closed
        pickle.dump(data,f)


