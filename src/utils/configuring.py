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
            self.SETTINGS = configuration['SETTINGS']

            # add defaults these have not been set
            if not configuration.has_option('SETTINGS', 'version'):
                self.SETTINGS['version'] = "1"
            if not configuration.has_option('SETTINGS', 'number trees'):
                self.SETTINGS['number trees'] = "1000"
            if not configuration.has_option('SETTINGS', 'number cores'):
                self.SETTINGS['number cores'] = "-1"
            if not configuration.has_option('SETTINGS', 'attribute'):
                self.SETTINGS['training attribute'] = "id"
            if not configuration.has_option('SETTINGS', 'patch threshold'):
                self.SETTINGS['patch threshold'] = "0"
            if not configuration.has_option('SETTINGS', 'classes to map'):
                self.SETTINGS['classes to map'] = "0"
        except KeyError as key_ex:

            print_debug(f'badly formatted: "{os.path.abspath(CONFIG_FILE)}"')
            print_debug(f'missing "{key_ex}" section', force_exit=True)

        print_debug('configuration loaded')
        print_debug(msg=f'region being processed: "{self.PATHS["study location"]}"')
