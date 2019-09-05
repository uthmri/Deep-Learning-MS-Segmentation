import keras
import random
import pickle
import numpy as np
from collections import Counter
from skimage.measure import label, regionprops
from scipy import ndimage
from mri_functions import data_extraction_from_files, label_extraction_from_files

def load_obj( name ):
    with open(name, 'rb') as f:
        return pickle.load(f)



# Function which picks slices from volumes to train network
def find_slice(patch_size, echo_image, val_image, lesion_class = 4):

    # Defines function's internal patch size to compute
    patch_size = (patch_size[0], patch_size[1], 1)

    # Constant list to compute
    values = [-1, 1, 0]

    # Declaration of function to pick random slices from volume
    random_selector = random.SystemRandom()

    # Buffer volume to process ground truth array
    buff_image = val_image.reshape((val_image.shape[0], val_image.shape[1], val_image.shape[2]))

    # Generates and array of only lesions
    lesion_locations = np.isin(buff_image, lesion_class)


    # Skimage label function which returns lesion labels and regions depending on label connectivity
    vol_lesions = label(lesion_locations, return_num=True, connectivity=2)



    # Function to fill up small tissues observed in data [likely to be noise]
    if lesion_class not in val_image:
        # Iterates trough lesion regions determined by label function
       for lesion_region in regionprops(vol_lesions[0]):
           # Considers lesions for which the total area [number of voxels] is 3 or less in order to replace them by surrounding labels
           if lesion_region.area <= 3:
                # This section is to determine which is the majority label surrounding small lesions
               for voxel_location in lesion_region.coords:
                   surrounding_pixels = []
                   # Iteration on 3 axis
                   for value_x in values:
                       for value_y in values:
                           for value_z in values:
                               # Excludes current voxel from calculations
                               if value_x == 0 and value_y  == 0 and value_z == 0:
                                   continue

                                # Moves through surrounding voxels on different axes
                               x_val, y_val, z_val = voxel_location[0] + value_x, voxel_location[1] + value_y, voxel_location[2] + value_z

                                # Prevents algorithm from axis values beyond the volume boundary
                               if x_val < 0 or x_val > val_image.shape[0] - 1 or y_val < 0 or y_val > val_image.shape[1] - 1 or z_val < 0 or z_val > val_image.shape[2] -1:
                                   continue

                                # Creates a list with values surrounding the lesion
                               value_at = val_image[x_val, y_val, z_val, 0]
                               surrounding_pixels.append(value_at)

                    # Does majority vote and replaces lesion voxel by surrounding tissue voxels' majority value
                   majority_voxel = Counter(surrounding_pixels)
                   majority_val = majority_voxel.most_common()[0][0]
                   val_image[voxel_location[0], voxel_location[1], voxel_location[2], 0] = majority_val

    #
    #
    # crops slices depending on presence of lesions
    # if network is trained on several slices not containing enhancements, network performance could bias
    # custom resolution can be specified

    if lesion_class in val_image:
        coord_set = []

        for dim_idx, dimension in enumerate(patch_size):

            # selects a random lesion present in the volume
            # , and picks one of its coordinates randomly to select the whole slice.

            random_lesion = random_selector.choice(regionprops(vol_lesions[0]))
            random_coord = random_selector.choice(random_lesion.coords)

            # extends coordinate to pick either patch or whole slice
            range_max = random_coord[dim_idx] + (dimension // 2) #+ range_var
            range_min = random_coord[dim_idx] - (dimension // 2) #+ range_var

            # clips within max range of network resolution
            if range_max > buff_image.shape[dim_idx]:
                val_op = range_max - buff_image.shape[dim_idx]
                range_max -= val_op
                range_min -= val_op

            # clips to 0
            elif range_min < 0:
                val_op = 0 - range_min
                range_max += val_op
                range_min += val_op



            # sets crop to single slice
            if dim_idx == 2:
                range_max, range_min = random_coord[dim_idx], random_coord[dim_idx]

            coord_set.append([range_min, range_max])


        # Crops corresponding patches from sequences and ground truth to conduct training
        X = echo_image[coord_set[0][0]:coord_set[0][1], coord_set[1][0]:coord_set[1][1],
                coord_set[2][0], :]

        y = val_image[coord_set[0][0]:coord_set[0][1], coord_set[1][0]:coord_set[1][1], coord_set[2][0]]
        y = y.reshape((patch_size[0], patch_size[1]))


    # if lesions not present in volume, then a random slice is cropped.
    else:
        coord_set = []

        for dim_idx, dimension in enumerate(patch_size):

            # Selects random coordinate to obtain a patch
            random_coord = random_selector.choice(list(range(0, buff_image.shape[dim_idx])))

            range_max = random_coord + (dimension // 2)
            range_min = random_coord - (dimension // 2)

            if range_max > buff_image.shape[dim_idx]:
                val_op = range_max - buff_image.shape[dim_idx]
                range_max -= val_op
                range_min -= val_op

            elif range_min < 0:
                val_op = 0 - range_min
                range_max += val_op
                range_min += val_op

            if dim_idx == 2:
                range_max, range_min = random_coord, random_coord

            coord_set.append([range_min, range_max])


        # Crops corresponding patches from sequences and ground truth to conduct training
        X = echo_image[coord_set[0][0]:coord_set[0][1], coord_set[1][0]:coord_set[1][1],
            coord_set[2][0], :]

        y = val_image[coord_set[0][0]:coord_set[0][1], coord_set[1][0]:coord_set[1][1], coord_set[2][0]]

        y = y.reshape((patch_size[0], patch_size[1]))


    return X, y





# DataGenerator that deals with the stream of data from CPU to GPU.
# This class helps processing data and putting it in batches that go to the GPU
class DataGenerator(keras.utils.Sequence):
    "Defaults for DataGenerator variables"
    def __init__(self, list_IDs, labels, batch_size=2, dim=(256, 256)
                 , n_channels=4, n_classes=2, shuffle=True
                 , augment=False, data_folder=None
                 , training_seq=['pd', 'flair', 't1_pre', 't2']
                 , valid_seq=["validated"], file_format = ".nii.gz"):


        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.data_folder = data_folder
        self.training_seq = training_seq
        self.valid_seq = valid_seq
        self.file_format = file_format
        self.on_epoch_end()


    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generates batches of data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        # Batch initialization for data and label
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))


        # Gets data on
        for i, ID in enumerate(list_IDs_temp):

            # process and generates data passed through network
            X_buff = data_extraction_from_files(dataset_path=self.data_folder, img_folder=ID
                                                , training_seq=self.training_seq, file_format=self.file_format, normalize=True)
            y_ref = label_extraction_from_files(dataset_path=self.data_folder, img_folder=ID
                                                , validation_seq=self.valid_seq, file_format=self.file_format)


            # Looks for sample slice in current volume
            X[i,], y[i,] = find_slice(self.dim, X_buff, y_ref)


            # Implements random data augmentation on selected slices
            if self.augment == True:
                rand_mirror_x = random.randint(0,1)
                rand_mirror_y = random.randint(0,1)
                rand_rot_by_90 = random.randint(0,1)
                rand_ref_rot_angle = random.randint(0,3)

                # Random mirror on X axis
                if rand_mirror_x == 0:
                    X[i,], y[i,] = np.fliplr(X[i,]), np.fliplr(y[i,])

                # Random mirror on Y axis
                if rand_mirror_y == 0:
                    X[i,], y[i,] = np.flipud(X[i,]), np.flipud(y[i,])

                # Random 90 | 180 | 270 degree rotation
                if rand_rot_by_90 == 0:
                    X[i,], y[i,] = np.rot90(X[i,], rand_ref_rot_angle), np.rot90(y[i,], rand_ref_rot_angle)


        # Returns data and label batch
        # To categorical turns a class vector into a binary class matrix
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
