#!/usr/bin/env python
import os

from src import print_debug


def base_dir_path(configuration):
    base_path = os.path.join(configuration.PATHS['data partition'], configuration.PATHS['study location'])
    return base_path


def results_dir_path(configuration):
    results_path = os.path.join(base_dir_path(configuration), 'results')
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    return results_path


def classified_dir_path(configuration):
    classification_path = os.path.join(base_dir_path(configuration), 'classified')
    if not os.path.isdir(classification_path):
        os.makedirs(classification_path)
    return classification_path


def images_dir_path(configuration):
    images_path = os.path.join(base_dir_path(configuration), 'images')
    if not os.path.isdir(images_path):
        print_debug(f'ERROR: directory with images required: "{images_path}"', force_exit=True)
    return images_path


def training_file_path(configuration):
    training_filepath = os.path.join(base_dir_path(configuration), 'training',
                                     configuration.PATHS['training shapefile']+'.shp')
    if not os.path.isfile(training_filepath):
        print_debug(f'ERROR: training shapefile required: "{training_filepath}"', force_exit=True)
    return training_filepath


def logging_dir_path(configuration):
    logging_path = os.path.join(base_dir_path(configuration), 'logs')
    if not os.path.isdir(logging_path):
        os.makedirs(logging_path)
    return logging_path


def classified_file_paths(configuration, image_name):
    classified_filename = image_name.replace(configuration.SETTINGS["image identifier"],
                                             f'class_v{configuration.SETTINGS["version"]}')
    classified_filepath = os.path.join(classified_dir_path(configuration), classified_filename)
    return classified_filepath


def seagrass_file_paths(configuration, number_images):
    seagrass_filename = f'{configuration.PATHS["study location"]}_' +\
                        f'seagrass_{configuration.SETTINGS["image identifier"]}_' +\
                        f'v{configuration.SETTINGS["version"]}_' +\
                        f'MEC{configuration.SETTINGS["MEC"]}-{number_images}.tif'
    seagrass_filepath = os.path.join(results_dir_path(configuration), seagrass_filename)
    return seagrass_filepath
