import os
import configparser

from src.utils.printing import print_debug

CONFIG_FILE = "config\\rf4eo_config.ini"


class Config(object):

    def __init__(self):
        """Constructor method.
        """
        try:
            # parse the config file
            configuration = configparser.ConfigParser()
            configuration.read(CONFIG_FILE)

            # add sections
            self.PATHS = configuration['PATHS']
            self.CLASSIFIER = configuration['CLASSIFIER']
            self.SETTINGS = configuration['SETTINGS']

            # add defaults if these have not been set

            if not configuration.has_option('PATHS', 'piggy-back location'):
                self.PATHS['piggy-back location'] = "None"

            if not configuration.has_option('CLASSIFIER', 'algorithm'):
                self.CLASSIFIER['algorithm'] = "gini"
            if not configuration.has_option('CLASSIFIER', 'number trees'):
                self.CLASSIFIER['number trees'] = "1000"
            if not configuration.has_option('CLASSIFIER', 'number cores'):
                self.CLASSIFIER['number cores'] = "-1"
            if not configuration.has_option('CLASSIFIER', 'max features'):
                self.CLASSIFIER['max features'] = "None"

            if not configuration.has_option('SETTINGS', 'version'):
                self.SETTINGS['version'] = "1"
            if not configuration.has_option('SETTINGS', 'training attribute') or \
                    configuration.has_option('SETTINGS', 'training attribute') == "":
                self.SETTINGS['training attribute'] = "id"
            if not configuration.has_option('SETTINGS', 'patch threshold'):
                self.SETTINGS['patch threshold'] = "0"
            if not configuration.has_option('SETTINGS', 'classes to map'):
                self.SETTINGS['classes to map'] = "0"
            if not configuration.has_option('SETTINGS', 'identifier'):
                self.SETTINGS['identifier'] = ""
            if self.SETTINGS['identifier'] != "":
                self.SETTINGS['identifier'] = '_' + self.SETTINGS['identifier']
            if not configuration.has_option('SETTINGS', 'save geotiffs'):
                self.SETTINGS['save geotiffs'] = "true"
            if not configuration.has_option('SETTINGS', 'save classifier'):
                self.SETTINGS['save classifier'] = "false"
        except KeyError as key_ex:

            print_debug(f'badly formatted: "{os.path.abspath(CONFIG_FILE)}"')
            print_debug(f'missing "{key_ex}" section', force_exit=True)

        print_debug('configuration loaded')
        print_debug(msg=f'region being processed: "{self.PATHS["study location"]}"')
