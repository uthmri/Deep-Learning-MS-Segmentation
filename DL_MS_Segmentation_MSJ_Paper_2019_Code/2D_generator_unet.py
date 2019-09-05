# library for operative system modification, useful for setting computing devices such as GPU and others
import os
import sys
# library for handling medical images
import nibabel as nib

from math import floor
from random import shuffle
from keras import backend as K

# Sets up the number of GPUs available in the system
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Library containing optimizers
import keras.optimizers as opt

# Functions to get learning rate and to create a 2D Unet
from alt_model import get_lr_metric, create_unet_model2D

# Function to load a model from HDF5 file
from keras.models import load_model

# Imports functions to calculate metrics related to network performance [tissue_dice & weighted_dice_coefficient_loss are the most important]
from aux_metrics import dice_coefficient, recall, precision, weighted_dice_coefficient_loss, tissue_dice

# Function to count GPU devices in system
from tensorflow.python.client import device_lib

# Keras callback functions which support network analysis
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint

# DataGenerator class which eases network training using large datasets (due to memory issues)
from class_generator import DataGenerator

# Helpful functions to deal with data and multi-gpu training
from mri_functions import get_immediate_subdirectories, load_cases_from_csv, ModelMGPU

# Function to evaluate network on data
from volume_construction import visualization


# Class which implements Unet configuration and training
class myUnet(object):

    # Function to train Unet given some parameters and functions
    def train(self, data_folder, csv_logger,  mode, imgs_train
              ,imgs_validation, img_resolution, unet_folder
              , num_of_gpus, train_seq, val_seq, file_format):


        # Custom Data Generator parameters for training set specifically
        # data folder corresponds to folder where data resides, training_seq corresponds to a list of sequence names used as input to the network
        # valid_seq contains the names of files containing ground truth for supervised training
        params = {'dim':(256, 256),
                  'batch_size':2,
                  'n_classes':5,
                  'n_channels':4,
                  'shuffle':True,
                  'augment':True,
                  'data_folder':data_folder,
                  'training_seq': train_seq,
                  'valid_seq':val_seq,
                  'file_format':file_format}

        # Custom Data Generator parameters for validation set specifically, validation data is not augmented
        params_val = {'dim':(256, 256),
                      'batch_size':2,
                      'n_classes':5,
                      'n_channels':4,
                      'shuffle':True,
                      'augment':False,
                      'data_folder':data_folder,
                      'training_seq': train_seq,
                      'valid_seq': val_seq,
                      'file_format':file_format}


        # Define set of images in training set and validation set
        train_set, val_set =[], []
        for idx, img_num in enumerate(imgs_train):
            train_set.append(str(img_num))

        for idx, img_num in enumerate(imgs_validation):
            val_set.append(str(img_num))

        # Partition definition for both sets of data. Labels is included as a convenience for extension to a different problem.
        partition = {'train':train_set, 'validation':val_set}
        labels = {'mask':['0']}

        # Definition of train and validation data generators.
        train_image_generator = DataGenerator(partition['train'], labels, **params)
        val_image_generator = DataGenerator(partition['validation'], labels, **params_val)


        # Learning rate, optimizer, loss function definitions, and number of epochs.
        learn_rate = 1e-4
        optimizer_A = opt.Adam(lr=learn_rate)
        custom_loss = weighted_dice_coefficient_loss
        epochs = 50

        # Learning rate metric for convenience
        lr_metric = get_lr_metric(optimizer_A)

        # Mode for training network from scratch
        if mode == 'create':

            # Creates 2D Unet inspired network model.
            model_to_save = create_unet_model2D((img_resolution[0], img_resolution[1], 4), n_labels=5, layers=5
                                                       , lowest_resolution=64, convolution_kernel_size=(3, 3)
                                                       , deconvolution_kernel_size=(3, 3), pool_size=(2, 2),
                                                       init_lr=learn_rate, gpu_num=num_of_gpus)

            # Creates model which can be trained on multiple GPUs
            parallel_model = ModelMGPU(model_to_save, num_of_gpus)
            # Displays summary of model layers
            model_to_save.summary()
            # Convenient function for saving model achieving lowest validation loss
            model_checkpoint = ModelCheckpoint(unet_folder, monitor='val_loss', mode='min',
                                               verbose=1, save_best_only=True)


            # Compiles model with selection of loss function and optimizer, additional metrics are included for tissue specific dice
            parallel_model.compile(loss=custom_loss, optimizer=optimizer_A, metrics=[dice_coefficient
                , recall, precision, tissue_dice(0, "background_dice"), tissue_dice(1, "grey_dice")
                , tissue_dice(2, "white_dice"), tissue_dice(3, "csf_dice"), tissue_dice(4, "t2_lesion_dice"), lr_metric])

            # Trains models using custom Data Generators, number of workers can be modified as needed. Verbose value modifies command line output
            parallel_model.fit_generator(generator=train_image_generator, validation_data=val_image_generator,
                                         use_multiprocessing=False, workers=16, epochs=epochs,
                                         callbacks=[csv_logger, model_checkpoint] , verbose=2)

        # Mode for training a network for which training was done previously and a model file is available
        if mode == 'train':

            # Learning rate metric
            lr_metric = get_lr_metric(optimizer_A)

            # Loads previosly trained model for which a file is specified, custom objects must be set to those used in the network original training.
            model_to_save = load_model(unet_folder, custom_objects={'loss':custom_loss, 'dice_coefficient': dice_coefficient,
                'recall': recall, 'precision': precision, "background_dice": tissue_dice(0, "background_dice")
                , "grey_dice": tissue_dice(1, "grey_dice"), "white_dice":tissue_dice(2, "white_dice"), "csf_dice":tissue_dice(3, "csf_dice")
                , "t2_lesion_dice":tissue_dice(4, "t2_lesion_dice"), 'lr':lr_metric})


            # Checkpoints model to save the one for which validation loss was lower
            model_checkpoint = ModelCheckpoint(unet_folder, monitor='val_loss',
                                               verbose=1, save_best_only=True)

            # Creates model which can be trained on multiple GPUs
            gpu_model = ModelMGPU(model_to_save, num_of_gpus)

            # Compiles model with custom objects equal to those originally used
            gpu_model.compile(loss=custom_loss, optimizer=optimizer_A, metrics=['categorical_accuracy'
                , 'categorical_crossentropy', dice_coefficient, recall, precision
                , tissue_dice(0, "background_dice"), tissue_dice(1, "grey_dice"), tissue_dice(2, "white_dice")
                , tissue_dice(3, "csf_dice"), tissue_dice(4, "t2_lesion_dice"), lr_metric])

            # Begins model retraining
            gpu_model.fit_generator(train_image_generator, validation_data=val_image_generator, epochs=epochs, verbose=2,
                                 use_multiprocessing=False, workers= 16, callbacks=[csv_logger, model_checkpoint])



if __name__ == '__main__':

    # Clears GPU from any current or previous jobs
    K.clear_session()
    # Prints computing devices available in system
    print(device_lib.list_local_devices())

    # Specifies folder which contains interest volumes
    data_folder = "../Whole_Set"

    # Specifies available MRI sequences file names
    training_seq = ['flair',
                    't1_pre',
                    't2',
                    'pd'
                    ]

    # Specifies ground truth file names
    validation_seq = ['validated']

    # Number of GPUs available in system
    gpu_num = 2

    # dataset partition used
    tr_val_test_size = [0.7, 0.1, 0.2]
    # Tissue labels to be used in ground truth. In our case 1-> grey matter  2-> white matter 3-> csf 4-> T2-lesion
    label_values = [1, 2, 3, 4]

    # Number of slices to remove from top and bottom of brain for convenience
    top_cut, bottom_cut = 6, 6
    # 44 is the median number of slices in the volumes available
    objective_slice_num = 44 - top_cut - bottom_cut

    image_resolution = [256, 256, objective_slice_num]
    image_resolution_normal = [256, 256, 44]

    patch_resolution = (256, 256)

    file_format = ".nii.gz"

    # Folder to save neural network variables per epoch
    logger_folder = "out_files/log.csv"

    # Folder which contains model file
    unet_file_folder = "out_files/model_unet.hdf5"

    # definition of logger to CSV file
    csv_logger = CSVLogger(logger_folder, append=True, separator=';')

    # volumes baseline and test previously defined randomly
    baseline = load_cases_from_csv("cases_files/baseline.csv")

    # specifies case and file for which affine array would be obtained from
    affine_case, affine_file = "23004", "validated{0}".format(file_format)

    # specifies folder in which test cases would be saved
    output_folder = "outputs"



    # Verification that all volumes listed in csv are present in files folder
    data_list = get_immediate_subdirectories(data_folder)
    working_baseline = baseline.copy()
    for idx_case, case in enumerate(baseline):
        if case not in data_list:
            working_baseline.remove(case)
    baseline = working_baseline
    shuffle(baseline)

    # dataset partition ratios
    train_part, val_part, test_part = 0.6, 0.2, 0.2

    # Partition of baseline minus test set into training and validation sets.
    l_baseline = len(baseline)
    train_set, val_set, test_set = baseline[0:floor(l_baseline * train_part)]\
        , baseline[floor(l_baseline * train_part) + 1: floor(l_baseline * (train_part + val_part) )]\
        ,  baseline[floor(l_baseline * (train_part + val_part)) + 1: floor(floor(l_baseline * (train_part + val_part + test_part)))]

    print("Sets sizes: Train Set: {0} | Val Set: {1} | Test Set: {2}".format(len(train_set), len(val_set), len(test_set)))

    # class object to define functions applicable to network
    myunet = myUnet()

    # "create" mode for which network is trained from scratch
    # "train" previously trained model is retrained with weights saved in model file
    mode_op = 'train'

    # network definition with file and system specific parameters
    myunet.train(data_folder=data_folder, mode=mode_op
                                        , csv_logger=csv_logger, imgs_train=train_set
                                        , imgs_validation=val_set, img_resolution=image_resolution
                                        , unet_folder=unet_file_folder, num_of_gpus=gpu_num
                                        , train_seq=training_seq, val_seq=validation_seq
                                        , file_format=file_format)

    # definition of affine set
    ref_img_path = '{0}/{1}/{2}'.format(data_folder, affine_case, affine_file)
    ref_img = nib.load(ref_img_path)
    affine_set = ref_img.affine


    # Generation of test set volumes' segmentation with trained network
    visualization(tuple(image_resolution_normal), data_folder
                  , model_path=unet_file_folder, test_images=test_set
                  , patch_res=patch_resolution, affine_set=affine_set, output_folder=output_folder, file_format=file_format)
