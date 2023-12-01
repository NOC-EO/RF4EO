#!/usr/bin/env python
import os
import sys
import numpy as np
from skimage.morphology import remove_small_holes

from src.utils.naming import results_dir_path, seagrass_file_path
from src.utils.filing import get_image_paths, read_geotiff, write_geotiff
from src import print_debug, Config


class RF4EO_MCS(object):

    def __init__(self):
        self.Config = Config()

    def run(self):

        VERSION = self.Config.SETTINGS["version"]
        MCS = int(self.Config.SETTINGS["MCS"])
        PATCH_THRESHOLD = int(self.Config.SETTINGS['patch threshold'])

        results_directory = results_dir_path(configuration=self.Config)
        images_to_aggregate = get_image_paths(images_directory=results_directory, identifier='v' + VERSION)
        number_images = len(images_to_aggregate)

        if not number_images:
            print_debug(f'WARNING: no images found here "{results_directory}"', force_exit=True)

        image_np, geometry = read_geotiff(images_to_aggregate[0])
        image_np = np.squeeze(image_np)
        (y_size, x_size) = image_np.shape

        print_debug(f'number of classes: {len(np.unique(image_np))}')
        print_debug(f'image shape: ({y_size}, {x_size})')
        print_debug()

        # import the classied images
        images_np = np.zeros((number_images, y_size, x_size))
        for image_index, image_path in enumerate(images_to_aggregate):
            image_name = os.path.split(image_path)[1]
            try:
                image_np = read_geotiff(image_path)[0].squeeze()
                images_np[image_index] = image_np
                print_debug(f'good: {image_name}')
            except:
                print_debug(f'bad: {image_name} {np.unique(image_np)} {image_np.shape}')
                continue

        # select the two classes of interest to be mapped
        seagrass_exposed_np = np.sum(np.where(images_np == 3, 1, 0), axis=0)
        #seagrass_submerged_np = np.sum(np.where(images_np == 4, 1, 0), axis=0)

        # apply the MCS filter
        seagrass_exposed_np[seagrass_exposed_np < MCS] = 0
        #seagrass_submerged_np[seagrass_submerged_np < MCS] = 0

        # aggregate the two classes into a single array
        aggregated_np = np.where(seagrass_exposed_np != 0, 1, 0)
        #aggregated_np = np.where(seagrass_submerged_np != 0, 2, aggregated_np)

        # filter out small patches
        mask_np = ~aggregated_np.astype(bool)
        mask_np = remove_small_holes(ar=mask_np, area_threshold=PATCH_THRESHOLD, connectivity=1)
        aggregated_np[mask_np] = 0

        write_geotiff(aggregated_np, seagrass_file_path(self.Config), geometry)


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance and implement the run function
    obj = RF4EO_MCS()
    obj.run()
