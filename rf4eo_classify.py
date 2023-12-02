#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.utils.naming import images_dir_path, classified_file_path
from src.utils.filing import get_image_paths, get_training_dataset, get_training_array, \
    read_geotiff, write_geotiff, get_logger
from src import print_debug, Config


class RF4EO_Classify(object):

    def __init__(self):
        self.Config = Config()

    def run(self):

        CLASSIFICATION_ATTR = self.Config.SETTINGS['training attribute']
        NUMBER_OF_TREES = int(self.Config.SETTINGS['number trees'])
        NUMBER_OF_CORES = int(self.Config.SETTINGS['number cores'])
        VERBOSE = int(self.Config.SETTINGS['verbose'])

        training_dataset = get_training_dataset(configuration=self.Config)
        images_directory = images_dir_path(configuration=self.Config)
        images_to_classify = get_image_paths(images_directory=images_directory)
        number_of_images = len(images_to_classify)
        if not number_of_images:
            print_debug(msg=f'ERROR: no images found here "{images_directory}"', force_exit=True)
        print_debug()

        for image_index, image_path in enumerate(images_to_classify):

            image_name = os.path.split(image_path)[1]
            classified_filepath = classified_file_path(self.Config, image_name)
            if os.path.exists(classified_filepath):
                print_debug(msg=f'WARNING: already classified "{image_name}" skipping')
                continue

            print_debug()
            print_debug(msg=f'classifying: "{image_name}"')
            processing_logger = get_logger(self.Config, f'processing_logger_{image_index}', image_name)

            try:
                image_np, geometry = read_geotiff(image_file_path=image_path)
            except AttributeError:
                print_debug(f'AttributeError: image file could not be read: "{image_path}"')
                continue

            # select bands from image to classify
            # only want to classify Sentinel-2 B2, B3, B4, B8 (red, green, blue and NIR) bands
            image_np = image_np[:, :, :4]
            (y_size, x_size, number_bands) = image_np.shape

            ground_truth_np = get_training_array(training_dataset=training_dataset,
                                                 geometry=geometry,
                                                 attribute=CLASSIFICATION_ATTR)
            class_labels = np.unique(ground_truth_np[ground_truth_np > 0])

            # split the ground truth data 80:20 into training and validation data
            X = image_np[ground_truth_np > 0, :]
            y = ground_truth_np[ground_truth_np > 0]
            seed = randint(0, 2**32 - 1)
            X_train, X_validation, y_train, y_validation = train_test_split(X, y,
                                                                            train_size=0.8,
                                                                            test_size=0.2,
                                                                            random_state=seed)

            # create a Random Forest classifier
            classifier = RandomForestClassifier(n_estimators=NUMBER_OF_TREES,
                                                criterion='gini',
                                                bootstrap=True,
                                                verbose=VERBOSE,
                                                n_jobs=NUMBER_OF_CORES)

            # train the classifier
            X_train = np.nan_to_num(X_train)
            fit_estimator = classifier.fit(X_train, y_train)
            if VERBOSE:
                print_debug(msg=f'number of features: {fit_estimator.n_features_in_}')
                print_debug(msg=f'classes being fitted: {fit_estimator.classes_}')

            # log the relative importance of each band
            processing_logger.info(msg='')
            processing_logger.info(msg='importance of each band')
            bands = range(1, number_bands + 1)
            for band, importance in zip(bands, fit_estimator.feature_importances_):
                processing_logger.info(msg=f'band {band} importance: {importance:.2f}')
            processing_logger.info(msg='')

            # classify the image
            image_np = image_np.reshape((image_np.shape[0] * image_np.shape[1], image_np.shape[2]))
            image_np = np.nan_to_num(image_np)
            try:
                classified_image_np = classifier.predict(image_np)
            except MemoryError:
                print_debug(msg=f'MemoryError: need to slice ...')

            # save classified image to file
            write_geotiff(output_array=classified_image_np.reshape((y_size, x_size)),
                          output_file_path=classified_filepath,
                          geometry=geometry)

            # perform an accuracy analysis

            # classify the validation data
            X_validation = np.nan_to_num(X_validation)
            y_predict = classifier.predict(X_validation)

            # produce the confusion matrix
            df = pd.DataFrame({'y_validation': y_validation, 'y_predicted': y_predict})
            confusion_matrix = pd.crosstab(index=df['y_validation'],
                                           columns=df['y_predicted'],
                                           rownames=['Ground Truth'],
                                           colnames=['Predicted'])
            processing_logger.info(msg=confusion_matrix)
            processing_logger.info(msg='')

            # then the classification report
            target_names = [str(name) for name in range(1, len(class_labels) + 1)]
            class_report = classification_report(y_validation, y_predict, target_names=target_names)
            processing_logger.info(msg=class_report)
            processing_logger.info(msg='')

            # lastly log the overall accuracy and send to console
            accuracy_message = f'Kappa = {(accuracy_score(y_validation, y_predict) * 100):.1f}%'
            print_debug(msg=accuracy_message)
            processing_logger.info(msg=accuracy_message)


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance and implement the run function
    obj = RF4EO_Classify()
    obj.run()
