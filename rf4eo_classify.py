#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
#    confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.utils.naming import images_dir_path, classified_file_paths
from src.utils.filing import get_image_paths, get_training_dataset, get_training_array, \
    read_geotiff, write_geotiff, get_logger
from src import print_debug, Config


class RF4EO_Classify(object):

    def __init__(self):
        self.Config = Config()
        print_debug()

    def run(self):

        CLASSIFICATION_ATTR = self.Config.SETTINGS['attribute']
        NUMBER_OF_TREES = int(self.Config.SETTINGS['number trees'])
        NUMBER_OF_CORES = int(self.Config.SETTINGS['number cores'])
        VERBOSE = int(self.Config.SETTINGS['verbose'])

        training_dataset = get_training_dataset(configuration=self.Config)
        images_directory = images_dir_path(configuration=self.Config)
        images_to_classify = get_image_paths(images_directory=images_directory)
        number_of_images = len(images_to_classify)

        if not number_of_images:
            print_debug(msg=f'ERROR: no images found here "{images_directory}"', force_exit=True)

        for image_index, image_path in enumerate(images_to_classify):
            image_name = os.path.split(image_path)[1]
            processing_logger = get_logger(self.Config, f'processing_logger_{image_index}', image_name)

            print_debug()
            print_debug(msg=f'classifying: "{image_name}"')

            image_np, geometry = read_geotiff(image_file_path=image_path)
            (y_size, x_size, number_bands) = image_np.shape

            training_np = get_training_array(training_dataset=training_dataset,
                                             geometry=geometry,
                                             attribute=CLASSIFICATION_ATTR)
            labels = np.unique(training_np[training_np > 0])

            # split the training data 80:20 into training and validation data
            X = image_np[training_np > 0, :]
            y = training_np[training_np > 0]
            X_train, X_validation, y_train, y_validation = train_test_split(X, y,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=42)

            # create a Random Forest classifier
            classifier = RandomForestClassifier(n_estimators=NUMBER_OF_TREES,
                                                bootstrap=True,
                                                oob_score=True,
                                                verbose=VERBOSE,
                                                n_jobs=NUMBER_OF_CORES)

            # train the classifier
            X_train = np.nan_to_num(X_train)
            fit_estimator = classifier.fit(X_train, y_train)

            # log the relative importance of each band
            processing_logger.info(msg='')
            processing_logger.info(msg='importance of each band')
            bands = range(1, number_bands + 1)
            for band, importance in zip(bands, fit_estimator.feature_importances_):
                processing_logger.info(msg=f'band {band} importance: {importance:.2f}')
            processing_logger.info(msg='')

            # classify the entire image
            new_shape = (image_np.shape[0] * image_np.shape[1], image_np.shape[2])
            image_ar = image_np.reshape(new_shape)
            image_ar = np.nan_to_num(image_ar)

            try:
                classified_image_np = classifier.predict(image_ar)
            except MemoryError:
                print_debug(msg=f'MemoryError: need to slice ...')

            # save classified image to file
            write_geotiff(output_array=classified_image_np.reshape((y_size, x_size)),
                          output_file_path=classified_file_paths(self.Config, image_name),
                          geometry=geometry)

            # perform accuracy analysis and log the results
            # first the confusion matrix
            X_validation = np.nan_to_num(X_validation)
            y_predict = classifier.predict(X_validation)
            confusion_matrix = pd.crosstab(y_validation, y_predict, margins=True)
            processing_logger.info(msg=confusion_matrix)
            processing_logger.info(msg='')

            # and then the classification report
            target_names = [str(name) for name in range(1, len(labels) + 1)]
            class_report = classification_report(y_validation, y_predict, target_names=target_names)
            processing_logger.info(msg=class_report)
            processing_logger.info(msg='')

            # lastly log the overall accuracy and send to console
            message = f'OAA = {(accuracy_score(y_validation, y_predict) * 100):.1f}%'
            print_debug(msg=message)
            processing_logger.info(msg=message)


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance and implement the run function
    obj = RF4EO_Classify()
    obj.run()
