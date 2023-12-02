#!/usr/bin/env python
import os
import glob
import numpy as np
import logging
from osgeo import gdal, ogr

from src import print_debug

from src.utils.naming import training_file_path, logging_dir_path


def get_image_paths(images_directory, identifier: str = None):
    if identifier:
        image_file_paths = glob.glob(pathname=images_directory + f'\\*{identifier}.tif', recursive=False)
    else:
        image_file_paths = glob.glob(pathname=images_directory + '\\*.tif', recursive=False)
    return image_file_paths


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


def get_training_dataset(configuration):

    try:
        training_dataset = ogr.Open(training_file_path(configuration))
    except IOError as ex:
        print_debug(f'ERROR: could not open training shapefile {ex}', force_exit=True)

    attributes = []
    ldefn = training_dataset.GetLayer().GetLayerDefn()
    for field_index in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(field_index)
        attributes.append(fdefn.name)

    attribute = configuration.SETTINGS['training attribute']
    if attribute not in attributes:
        print_debug(f'ERROR: "{attribute}" not in the training shapefile')
        print_debug(f'       available attributes are: {attributes}', force_exit=True)

    return training_dataset


def get_training_array(training_dataset, geometry, attribute):

    shape_layer = training_dataset.GetLayer()

    (geotransform, projection, y_size, x_size) = geometry
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create('', x_size, y_size, 1, gdal.GDT_UInt16)
    mem_raster.SetProjection(projection)
    mem_raster.SetGeoTransform(geotransform)
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(0)
    mem_band.SetNoDataValue(0)

    att_ = 'ATTRIBUTE=' + attribute
    err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1], [att_, "ALL_TOUCHED=TRUE"])
    assert err == gdal.CE_None

    training_np = mem_raster.ReadAsArray()

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
