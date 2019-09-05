import os
import csv
import random
import pickle
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from keras.utils import multi_gpu_model
from keras.models import Model

# Auxiliar functions to process data

# Function to load pickle file into object
def load_obj( name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Saves object into pickle file
def save_obj( obj, name, obj_folder ):
    with open(obj_folder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Loads csv single column sequence of cases into a python list
def load_cases_from_csv(csv_file_name):

    base_list = []

    # opens specified .csv file
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # reads rows from csv file and appends entries to list
        for row in csv_reader:
            base_list.append(row[0])

    return base_list


# Gets immediate subdirectories for a specified main folder
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# Class which allows to conveniently train a model on multiple GPUs and also to save checkpoints for that model
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


# function which allows loading and pre-processing of sequence data
def data_extraction_from_files(dataset_path, img_folder, training_seq=['pd', 'flair', 't1_pre', 't2'],
                                image_res=(256, 256, 50), file_format=".nii", normalize=False):

    # specifies array axis for which slices are located at
    z_axis = 2

    # sorts input sequence array in alphabetic order (not essential)
    training_seq = sorted(training_seq, key=str.lower)

    # creates array of zeros with dimensions set similar to those in data. z_axis is set to a higher value as all images
    # do not have the same size and some slices might be removed
    final_img = np.zeros((image_res[0], image_res[1], image_res[2], len(training_seq)))


    # attempts to load case data, if images are unavailable or some sequences not present it would not interrumpt overall program execution
    try:
        # iterates through interest sequences and loads them to final_img object
        for sequence_idx, sequence_name in enumerate(sorted(training_seq)):

            current_img = nib.load(dataset_path + "/" + img_folder + "/" + sequence_name + file_format)
            current_img_data = current_img.get_data()
            current_img_data = current_img_data[:, :, :]

            final_img[:, :, 0:current_img_data.shape[z_axis], sequence_idx] = current_img_data

        final_img = final_img[:, :, 0:current_img_data.shape[z_axis], :]

    # returns a None object if case couldn't be loaded properly
    except:
        final_img = None

    # normalizes case if it was loaded properly
    if final_img is not None:
        if normalize == True:
            final_img /= np.amax(final_img)

    return final_img



# function which allows loading and pre-processing of sequence data
def data_extraction_from_files_rev(dataset_path, img_folder, training_seq=['pd', 'flair', 't1_pre', 't2'],
                                image_res=(256, 256, 50), file_format=".nii", normalize=False):

    # specifies array axis for which slices are located at
    z_axis = 2

    # sorts input sequence array in alphabetic order (not essential)
    training_seq = sorted(training_seq, key=str.lower)

    # creates array of zeros with dimensions set similar to those in data. z_axis is set to a higher value as all images
    # do not have the same size and some slices might be removed
    final_img = np.zeros((image_res[0], image_res[1], image_res[2], len(training_seq)))


    # attempts to load case data, if images are unavailable or some sequences not present it would not interrumpt overall program execution

        # iterates through interest sequences and loads them to final_img object
    for sequence_idx, sequence_name in enumerate(sorted(training_seq)):

        current_img = nib.load(dataset_path + "/" + img_folder + "/" + sequence_name + file_format)
        current_img_data = current_img.get_data()
        current_img_data = current_img_data[:, :, :]

        final_img[:, :, 0:current_img_data.shape[z_axis], sequence_idx] = current_img_data

    final_img = final_img[:, :, 0:current_img_data.shape[z_axis], :]

# returns a None object if case couldn't be loaded properly



    # normalizes case if it was loaded properly
    if final_img is not None:
        if normalize == True:
            final_img /= np.amax(final_img)

    return final_img





# extract case ground truth labels to train supervised model
def label_extraction_from_files(dataset_path, img_folder, validation_seq=["validated"], values=[1, 2, 3, 4], file_format=".nii"):

    # attempts to load case without interrupting whole program if there is an error
    try:
        current_lbl_data, current_gad_data, final_lbl = None, None, None
        for label_name in sorted(validation_seq):

            if label_name == 'validated':
                current_lbl = nib.load(dataset_path + "/" + img_folder + "/" + label_name + file_format)
                current_lbl_data = current_lbl.get_data()

                current_lbl_data = current_lbl_data[:, :, :]
                val_mask = np.isin(current_lbl_data, values)
                current_lbl_data = current_lbl_data * val_mask

                final_lbl = current_lbl_data

        # cleans ground truth image from undesired labels
        labels_mask = np.isin(final_lbl, values)
        final_lbl *= labels_mask

    except:
        final_lbl = None

    return final_lbl



def main():
    pass



if __name__ == '__main__':
    main()

