#!/usr/bin/env python
import os
import glob
import numpy as np
import logging
from osgeo import gdal, ogr

from src.utils.naming import training_file_path, logging_dir_path
from src import print_debug


def get_image_paths(images_directory, identifier: str = None):
    pattern = ('\\*.tif', f'\\*{identifier}.tif')[bool(identifier)]
    image_file_paths = glob.glob(pathname=images_directory + pattern, recursive=False)
    return sorted(image_file_paths)


def read_geotiff(image_file_path):

    image_dataset = gdal.Open(image_file_path, gdal.GA_ReadOnly)
    image_np = image_dataset.ReadAsArray()

    if len(image_np.shape) == 3:
        image_np = np.moveaxis(image_np, 0, -1)

    geotransform = image_dataset.GetGeoTransform()
    projection = image_dataset.GetProjection()
    y_size = image_np.shape[0]
    x_size = image_np.shape[1]

    return image_np, (geotransform, projection, y_size, x_size)


def write_geotiff(output_array, output_file_path, geometry):

    if len(output_array.shape) == 2:
        output_array = np.expand_dims(output_array, axis=0)
    (geotransform, projection, y_size, x_size) = geometry
    num_bands = output_array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    try:
        output_ds = driver.Create(output_file_path, xsize=x_size, ysize=y_size,
                                  bands=num_bands, eType=gdal.GDT_Float32)
        output_ds.SetProjection(projection)
        output_ds.SetGeoTransform(geotransform)
        for band in range(num_bands):
            output_ds.GetRasterBand(band + 1).WriteArray(output_array[band])
    except IOError:
        print_debug(f'IOError: gdal dataset for: {output_file_path} cannot be created', force_exit=True)

    output_ds.FlushCache()


def load_ground_truth_data(configuration, geometry):

    ATTRIBUTE = configuration.SETTINGS['training attribute']
    training_filepath = training_file_path(configuration)
    (geotransform, projection, y_size, x_size) = geometry

    try:
        training_ds = ogr.Open(training_filepath)
    except IOError as ex:
        print_debug(ex)
        print_debug(f'ERROR: could not open training shapefile "{os.path.split(training_filepath)[1]}"',
                    force_exit=True)

    layer_definition = training_ds.GetLayer().GetLayerDefn()
    attributes = [(layer_definition.GetFieldDefn(field_index)).name
                  for field_index in range(layer_definition.GetFieldCount())]
    if ATTRIBUTE not in attributes:
        print_debug(f'ERROR: "{ATTRIBUTE}" not in the training shapefile')
        print_debug(f'       available attributes are: {attributes}', force_exit=True)

    destination_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 1, gdal.GDT_Byte)
    destination_ds.SetProjection(projection)
    destination_ds.SetGeoTransform(geotransform)
    destination_raster_band = destination_ds.GetRasterBand(1)
    destination_raster_band.Fill(0)
    destination_raster_band.SetNoDataValue(0)
    gdal.RasterizeLayer(destination_ds, [1], training_ds.GetLayer(), None, None, [1],
                        [f'ATTRIBUTE={ATTRIBUTE}', "ALL_TOUCHED=TRUE"])

    training_np = destination_ds.ReadAsArray()

    return training_np


def get_logger(configuration, logger_name, image_name):

    logging_path = os.path.join(logging_dir_path(configuration),
                                image_name.replace('.tif', f'_v{configuration.SETTINGS["version"]}.txt'))
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(fmt='%(message)s')
    fileHandler = logging.FileHandler(filename=logging_path, mode='w')
    fileHandler.setFormatter(fmt=formatter)
    logger.setLevel(level=logging.INFO)
    logger.addHandler(hdlr=fileHandler)

    return logging.getLogger(logger_name)
