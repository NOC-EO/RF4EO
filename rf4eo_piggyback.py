#!/usr/bin/env python
import os
import sys
import copy
import pickle
import numpy as np

from src.utils.naming import images_dir_path, classified_file_path, classifier_file_path
from src.utils.filing import get_image_paths, read_geotiff, write_geotiff
from src import print_debug, Config


class RF4EO_PB(object):

    def __init__(self):
        self.Config = Config()

    def run(self):

        PB_LOCATION = self.PATHS['piggy back location']
        if PB_LOCATION == 'None':
            print_debug('ERROR: need to set the piggy back location', force_exit=True)
        PB_config = copy.copydeep(self.Config)
        PB_config.PATHS['study location'] = PB_LOCATION

        # find the images to be classified as a list of filepaths
        images_directory = images_dir_path(configuration=self.Config)
        images_to_classify = get_image_paths(images_directory=images_directory)
        number_of_images = len(images_to_classify)
        if not number_of_images:
            print_debug(msg=f'ERROR: no images found here "{images_directory}"', force_exit=True)
        print_debug()

        for image_index, image_path in enumerate(images_to_classify):

            # if this image has already been classified skip to next image
            image_name = os.path.split(image_path)[1]
            classified_filepath = classified_file_path(self.Config, image_name)
            if os.path.exists(classified_filepath):
                print_debug(msg=f'WARNING: already classified {image_name}')
                continue

            # import the image geotiff to a numpy array using GDAL
            # since a numpy array does not know where it is located
            # also need to keep the geometrical information for later
            print_debug(msg=f'classifying: {image_name}')
            try:
                image_np, geometry = read_geotiff(image_file_path=image_path)
            except AttributeError:
                print_debug(f'AttributeError: image file could not be read: {image_path}')
                continue
            (y_size, x_size, number_bands) = image_np.shape

            classifier_file = open(classifier_file_path(PB_config, image_name), 'rb')
            classifier = pickle.load(classifier_file)
            classifier_file.close()

            # then classify the full image using the trained classifier
            image_np = image_np.reshape((y_size * x_size, number_bands))
            try:
                classified_image_np = classifier.predict(X=np.nan_to_num(image_np))
            except MemoryError:
                print_debug(msg=f'MemoryError: acquire a bigger machine...')
                continue

            # save classified image to disc as a geotiff again using GDAL
            # with the same geometrical information that came from the source image
            write_geotiff(output_array=classified_image_np.reshape((y_size, x_size)),
                          output_file_path=classified_filepath,
                          geometry=geometry)


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance and implement the run function
    obj = RF4EO_PB()
    obj.run()
