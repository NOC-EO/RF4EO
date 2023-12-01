#!/usr/bin/env python
import os
import sys
import numpy as np
from skimage.morphology import remove_small_holes

from src.utils.naming import classified_dir_path, seagrass_file_paths
from src.utils.filing import get_image_paths, read_geotiff, write_geotiff
from src import print_debug, Config


class RF4EO_MEC(object):

    def __init__(self):
        self.Config = Config()

    def run(self):

        VERSION = self.Config.SETTINGS["version"]
        MEC = int(self.Config.SETTINGS["MEC"])
        PATCH_THRESHOLD = int(self.Config.SETTINGS['patch threshold'])

        classified_images_dir_path = classified_dir_path(configuration=self.Config)
        MEC_image_paths = get_image_paths(images_directory=classified_images_dir_path, identifier='v' + VERSION)
        number_images = len(MEC_image_paths)

        if not number_images:
            print_debug(f'WARNING: no images found here "{classified_images_dir_path}"', force_exit=True)

        image_np, geometry = read_geotiff(MEC_image_paths[0])
        image_np = np.squeeze(image_np)
        (y_size, x_size) = image_np.shape

        print_debug(msg=f'number of classes: {len(np.unique(image_np))}')
        print_debug(msg=f'image shape: ({y_size}, {x_size})')
        print_debug()

        # import the classied images
        images_np = np.zeros((number_images, y_size, x_size))
        for image_index, image_path in enumerate(MEC_image_paths):
            image_name = os.path.split(image_path)[1]
            try:
                image_np = read_geotiff(image_path)[0].squeeze()
                images_np[image_index] = image_np
                print_debug(msg=f'good: {image_name}')
            except:
                print_debug(msg=f'bad: {image_name} {np.unique(image_np)} {image_np.shape}')
                continue

        # select the two classes of interest to be mapped
        seagrass_exposed_np = np.sum(np.where(images_np == 3, 1, 0), axis=0)
        #seagrass_submerged_np = np.sum(np.where(images_np == 4, 1, 0), axis=0)

        # apply the MCS filter
        seagrass_exposed_np[seagrass_exposed_np < MEC] = 0
        #seagrass_submerged_np[seagrass_submerged_np < MEC] = 0

        # aggregate the two classes into a single array
        aggregated_np = np.where(seagrass_exposed_np != 0, 1, 0)
        #aggregated_np = np.where(seagrass_submerged_np != 0, 2, aggregated_np)

        # filter out small patches
        mask_np = ~aggregated_np.astype(bool)
        mask_np = remove_small_holes(ar=mask_np, area_threshold=PATCH_THRESHOLD, connectivity=1)
        aggregated_np[mask_np] = 0

        seagrass_filepath = seagrass_file_paths(self.Config, number_images)
        write_geotiff(aggregated_np, seagrass_filepath, geometry)
        print_debug()
        print_debug(msg=f'saved multi-ensemble result: "{os.path.split(seagrass_filepath)[1]}"')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance and implement the run function
    obj = RF4EO_MEC()
    obj.run()
