import os
import math
import numpy as np
import shutil
import nibabel as nib
from keras.models import load_model
from mri_functions import load_obj, load_cases_from_csv, data_extraction_from_files


# function to obtain network segmentation for test cases
def visualization(img_resolution, images_folder, model_path, test_images, patch_res, affine_set, output_folder, file_format):


    # Loads trained model for making predictions
    model = load_model(model_path)

    # loops through available cases
    for img_idx, img_name in enumerate(test_images):

        # source folder where training files reside
        dir_to_move = "{0}/{1}/".format(images_folder, img_name)

        # destination folder where to save cases plus network predicted segmentation
        dir_destination = '{0}/{1}/'.format(output_folder, img_name)

        # keeps track of progress
        print("Image: {0} //  {1} out of {2}".format(img_name, img_idx, len(test_images)))

        # loads data information from sequence files
        data_array = data_extraction_from_files(images_folder, img_name, normalize=True, file_format=file_format)

        # array to store resulting segmentation
        buff_array = np.ndarray((data_array.shape[0], data_array.shape[1], data_array.shape[2]))

        # array to score network predicted tissue scores
        scores_array = np.ndarray((data_array.shape[0], data_array.shape[1], data_array.shape[2], 5))

        # variables to move through image depending on patch size
        iter_x, iter_y, iter_z = math.ceil( data_array.shape[0]/patch_res[0] ), math.ceil( data_array.shape[1]/patch_res[1] ), math.ceil( data_array.shape[2])

        # loops to iterate through volume
        for x_step in range(iter_x):
            for y_step in range(iter_y):
                for z_step in range(iter_z):

                    x_coord_init, x_coord_end = x_step*patch_res[0], x_step*patch_res[0] + patch_res[0]
                    y_coord_init, y_coord_end = y_step*patch_res[1], y_step*patch_res[1] + patch_res[1]
                    z_coord_init, z_coord_end = z_step, z_step

                    if x_coord_end > img_resolution[0]:
                        x_coord_init -= ( x_coord_end - img_resolution[0] )
                        x_coord_end = img_resolution[0] 
                    if y_coord_end > img_resolution[1]:
                        y_coord_init -= ( y_coord_end - img_resolution[1] )
                        y_coord_end = img_resolution[1]
                    if z_coord_end > img_resolution[2]:
                        z_coord_init -= ( z_coord_end - img_resolution[2] )
                        z_coord_end = img_resolution[2]


                    # creates array of image to be predicted
                    image_to_predict = data_array[x_coord_init:x_coord_end, y_coord_init:y_coord_end, z_coord_init]
                    # reshapes sequence array to be compatible with network expected format
                    image_to_predict = image_to_predict.reshape(1, patch_res[0], patch_res[1], 4)
                    # does prediction of sequence image
                    raw_array = model.predict(image_to_predict)
                    # finds maximum of network score to determine which tissue belongs to each voxel
                    predicted_array = np.argmax(raw_array, -1)

                    # saves predicted classes and scores to array for easy handling
                    buff_array[x_coord_init:x_coord_end, y_coord_init:y_coord_end, z_coord_init] = predicted_array[0, :, :]
                    scores_array[x_coord_init:x_coord_end, y_coord_init:y_coord_end, z_coord_init, :] = raw_array[0, :, :, :]


        # creates network prediction file in Nifti format
        network_image = nib.Nifti1Image(buff_array, affine=affine_set)

        nifti_network_name = "{0}/{1}/network.nii.gz".format(output_folder, img_name)

        # copies case directory into output folder
        if os.path.exists(dir_destination) == False:
            shutil.copytree(dir_to_move, dir_destination)


        # saves network file to corresponding folders
        nib.save(network_image, nifti_network_name)





def main():
    data_folder = "../Whole_Set"
    training_seq = ['flair',
                    't1_pre',
                    't2',
                    'pd'
                    ]
    validation_seq = ['validated']


    folders_path = load_cases_from_csv("cases_files/test_set_baseline.csv")

    model_path = "out_files/model_unet.hdf5"

    file_format = ".nii.gz"

    affine_case, affine_file = "23004", "validated{0}".format(file_format)

    # definition of affine set
    ref_img_path = '{0}/{1}/{2}'.format(data_folder, affine_case, affine_file)
    ref_img = nib.load(ref_img_path)
    affine_set = ref_img.affine

    output_folder = "outputs"


    visualization((256, 256, 44), data_folder, model_path, folders_path, (256, 256), affine_set, output_folder=output_folder, file_format=file_format)


if __name__ == "__main__":
    main()
