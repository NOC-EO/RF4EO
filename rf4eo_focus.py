#!/usr/bin/env python
import os
import sys
import numpy as np
from skimage.morphology import remove_small_holes

from src.utils.naming import classified_dir_path, seagrass_file_path
from src.utils.filing import get_image_paths, read_geotiff, write_geotiff
from src import print_debug, Config


class RF4EO_FOCUS(object):

    def __init__(self):
        self.Config = Config()
        print_debug()

    def run(self):

        VERSION = self.Config.SETTINGS["version"]
        MEC = int(self.Config.SETTINGS["MEC"])
        ATTRIBUTES = [int(attribute) for attribute in self.Config.SETTINGS["classes to map"].split(',')]
        PATCH_THRESHOLD = int(self.Config.SETTINGS['patch threshold'])

        classified_images_dir_path = classified_dir_path(configuration=self.Config)
        MEC_image_paths = get_image_paths(images_directory=classified_images_dir_path, identifier=f'v{VERSION}')
        number_images = len(MEC_image_paths)

        if not number_images:
            print_debug(f'WARNING: no images found here "{classified_images_dir_path}"', force_exit=True)

        image_np, geometry = read_geotiff(MEC_image_paths[0])
        image_np = np.squeeze(image_np)
        (y_size, x_size) = image_np.shape

        # if no classes of interest were specified aggregate all classes found in the images
        classes_in_images = list(np.unique(image_np).astype(np.int8))
        if ATTRIBUTES == [0]:
            ATTRIBUTES = classes_in_images
        print_debug(msg=f'classes in images: {classes_in_images}')
        print_debug(msg=f'classes to aggregate: {ATTRIBUTES}')
        print_debug(msg=f'image shape: ({y_size}, {x_size})')
        print_debug()

        # import the classified images
        images_np = np.zeros((number_images, y_size, x_size))
        good_images = 0
        for image_index, image_path in enumerate(MEC_image_paths):
            image_name = os.path.split(image_path)[1]
            try:
                image_np = read_geotiff(image_path)[0].squeeze()
                images_np[image_index] = image_np
                print_debug(msg=f'good: {image_name}')
                good_images += 1
            except AttributeError:
                print_debug(msg=f'bad: "{image_name}"  wrong type of file {image_np.shape}')
                continue
            except ValueError:
                print_debug(msg=f'bad: "{image_name}" wrong shape {image_np.shape}')
                continue

        # aggregate the classes of interest
        aggregated_np = np.zeros((y_size, x_size))
        for index, attribute in enumerate(ATTRIBUTES):

            # apply the multi ensemble classification filter
            attribute_np = np.sum(np.where(images_np == attribute, 1, 0), axis=0)
            attribute_np[attribute_np < MEC] = 0
            aggregated_np = np.where(attribute_np != 0, attribute, aggregated_np)

        # filter out small patches if required
        if bool(PATCH_THRESHOLD):
            mask_np = ~aggregated_np.astype(bool)
            mask_np = remove_small_holes(ar=mask_np, area_threshold=PATCH_THRESHOLD, connectivity=1)
            aggregated_np[mask_np] = 0

        seagrass_filepath = seagrass_file_path(self.Config, good_images)
        seagrass_filename = os.path.split(seagrass_filepath)[1]
        if os.path.exists(seagrass_filepath):
            print_debug(f'WARNING: same parameters already used "{seagrass_filename}"')
        else:
            write_geotiff(output_array=aggregated_np,
                          output_file_path=seagrass_filepath,
                          geometry=geometry)
            print_debug()
            print_debug(msg=f'saved multi-ensemble classification: "{seagrass_filename}"')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance and implement the run function
    obj = RF4EO_FOCUS()
    obj.run()
