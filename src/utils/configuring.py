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
            if not configuration.has_option('CLASSIFIER', 'algorithm'):
                self.SETTINGS['algorithm'] = "gini"
            if not configuration.has_option('CLASSIFIER', 'number trees'):
                self.SETTINGS['number trees'] = "1000"
            if not configuration.has_option('CLASSIFIER', 'number cores'):
                self.SETTINGS['number cores'] = "-1"

            if not configuration.has_option('SETTINGS', 'version'):
                self.SETTINGS['version'] = "1"
            if not configuration.has_option('SETTINGS', 'training attribute'):
                self.SETTINGS['training attribute'] = "id"
            if not configuration.has_option('SETTINGS', 'patch threshold'):
                self.SETTINGS['patch threshold'] = "0"
            if not configuration.has_option('SETTINGS', 'classes to map'):
                self.SETTINGS['classes to map'] = "0"
            if not configuration.has_option('SETTINGS', 'image identifier'):
                self.SETTINGS['image identifier'] = ""
            if self.SETTINGS['image identifier'] != "":
                print_debug('woof')
                self.SETTINGS['image identifier'] = '_' + self.SETTINGS['image identifier']
        except KeyError as key_ex:

            print_debug(f'badly formatted: "{os.path.abspath(CONFIG_FILE)}"')
            print_debug(f'missing "{key_ex}" section', force_exit=True)

        print_debug('configuration loaded')
        print_debug(msg=f'region being processed: "{self.PATHS["study location"]}"')
