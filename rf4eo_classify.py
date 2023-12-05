#!/usr/bin/env python
import os
import sys
import json
import numpy as np
import pandas as pd
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.utils.naming import images_dir_path, classified_file_path
from src.utils.filing import get_image_paths, read_geotiff, write_geotiff,\
    get_logger, load_ground_truth_data
from src import print_debug, Config


class RF4EO_Classify(object):

    def __init__(self):
        self.Config = Config()

    def run(self):

        # load the required configuration from the ini file
        # which was parsed by the configuring script
        # and made available here using a class attribute
        NUMBER_OF_TREES = int(self.Config.CLASSIFIER['number trees'])
        NUMBER_OF_CORES = int(self.Config.CLASSIFIER['number cores'])
        ALGORITHM = self.Config.CLASSIFIER['algorithm']
        MAX_FEATURES = self.Config.CLASSIFIER['max features']
        MAX_FEATURES = (MAX_FEATURES, None)[MAX_FEATURES == 'None']

        GEOTIFFS = json.loads(self.Config.SETTINGS['geotiffs'])
        VERBOSE = int(self.Config.SETTINGS['verbose'])

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

            # select the required bands from image to be classified
            # e.g. to classify Sentinel-2 B2, B3, B4, B8 (red, green, blue and NIR) bands
            # but this will depend on the satellite bands in the images being used
            image_np = image_np[:, :, :4]
            (y_size, x_size, number_bands) = image_np.shape

            # read the training shapefile as a numpy array using GDAL
            # with the geometry from the image being classified
            ground_truth_np = load_ground_truth_data(configuration=self.Config, geometry=geometry)

            # transform the image and ground truth data numpy arrays into 1D arrays
            # masked to values where the ground truth data is not zero
            X = image_np[ground_truth_np > 0, :]
            y = ground_truth_np[ground_truth_np > 0]

            # split the image and ground truth data arrays 80:20 into training and validation data
            # using a random seed - that if used again produces the same split
            seed = randint(0, 2**32 - 1)
            X_train, X_validation, y_train, y_validation = train_test_split(X, y,
                                                                            train_size=0.8,
                                                                            test_size=0.2,
                                                                            random_state=seed)

            # create a Random Forest classifier
            classifier = RandomForestClassifier(n_estimators=NUMBER_OF_TREES,
                                                criterion=ALGORITHM,
                                                oob_score=True,
                                                max_features=MAX_FEATURES,
                                                verbose=VERBOSE,
                                                n_jobs=NUMBER_OF_CORES)

            # train the classifier - with correctly prepared data i.e. no nan or inf values
            X_train = np.nan_to_num(X_train)
            fit_estimator = classifier.fit(X=X_train, y=y_train)
            print_debug(f'OOB: {classifier.oob_score_*100:.1f}%')
            if VERBOSE:
                print_debug(msg=f'number of features: {fit_estimator.n_features_in_}')
                print_debug(msg=f'classes being fitted: {fit_estimator.classes_}')

            # then classify the full image using the trained classifier
            image_np = image_np.reshape((image_np.shape[0] * image_np.shape[1], image_np.shape[2]))
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

            # a vital step in any supervised classification analysis is an accuracy assessment
            # performed here in four parts that each log metrics to a file on disc

            # open a logger and write a header
            assessment_logger = get_logger(self.Config, f'logger_{image_index}', image_name)
            assessment_logger.info(msg='Random Forest Classifier')
            assessment_logger.info(msg='')
            assessment_logger.info(msg=f'number of trees: {NUMBER_OF_TREES}')
            assessment_logger.info(msg=f'fitting criterion:  {ALGORITHM}')
            assessment_logger.info(msg=f'max features:  {MAX_FEATURES}')
            assessment_logger.info(msg=f'random seed:  {seed}')
            assessment_logger.info(msg=f'out of bag:  {classifier.oob_score_*100:.1f}%')
            assessment_logger.info(msg='')

            # first log the relative importance of each band
            assessment_logger.info(msg='relative importance of bands')
            for band_index, importance in enumerate(fit_estimator.feature_importances_):
                assessment_logger.info(msg=f'band {band_index+1} importance: {importance:.2f}')
            assessment_logger.info(msg='')

            # then classify the validation data (20% of the training data split)
            # and log the confusion matrix that is produced
            y_predict = classifier.predict(X=np.nan_to_num(X_validation))
            df = pd.DataFrame({'y_validation': y_validation, 'y_predicted': y_predict})
            confusion_matrix = pd.crosstab(index=df['y_validation'],
                                           columns=df['y_predicted'],
                                           rownames=['Ground Truth'],
                                           colnames=['Predicted'])
            assessment_logger.info(msg=confusion_matrix)
            assessment_logger.info(msg='')

            # next log the classification report
            class_labels = np.unique(y)
            target_names = [str(name) for name in range(1, len(class_labels) + 1)]
            class_report = classification_report(y_true=y_validation,
                                                 y_pred=y_predict,
                                                 target_names=target_names)
            assessment_logger.info(msg=class_report)
            assessment_logger.info(msg='')

            # finally log the overall accuracy
            overall_accuracy = accuracy_score(y_true=y_validation, y_pred=y_predict)
            print_debug(msg=f'OAA: {overall_accuracy*100:.1f}%')
            print_debug()
            assessment_logger.info(msg=f'Kappa: {overall_accuracy*100:.1f}%')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance and implement the run function
    obj = RF4EO_Classify()
    obj.run()
