import os
import time
import pickle
import argparse

import numpy as np

import torch
import torchvision

from thesis.data.experience_replay_episodes import ExperienceReplay
from thesis.rssm.rssm import RSSM


class Log:

    """
    Object for managing a log
    """

    def __init__(self, args: argparse.Namespace):

        self._log_dir = args.log_directory
        # Map existing logs to the type of values that they store
        self._logs = dict()
        # Map existing image folders to a list of images that they contain (to keep track of an ordering)
        self._image_folders = dict()
        # Set flag that controls whether logged messages should be printed
        self._print_msg = args.print_log

        # Ensure the log directory exists
        if not os.path.isdir(self._log_dir):
            os.mkdir(self._log_dir)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Get an argparse.ArgumentParser object for parsing arguments passed to the program
        :return: an argparse.ArgumentParser object for parsing hyperparameters
        """
        parser = argparse.ArgumentParser('Log Arguments')

        parser.add_argument('--log_directory',
                            type=str,
                            default='./log',
                            help='The name of the directory in which a log should be built')
        parser.add_argument('--print_log',
                            action='store_true',
                            help='Log message entries are printed when this flag is set')

        return parser

    def log_message(self, message: str, log_name='log'):
        """
        Write a message to the log file
        :param message: the message string to be written to the log file
        :param log_name: the name of the log file
        """
        with open(self._log_dir + f'/{log_name}.txt', 'a') as f:
            timestamp = time.strftime('%d-%m-%Y %H:%M:%S')
            f.write(f'[{timestamp}] {message}\n')
        # Print the message if required
        if self._print_msg:
            print(message)

    def log_args(self, args: argparse.Namespace):
        """
        Log the parsed arguments
        :param args: argparse.Namespace object containing the parsed arguments
        """
        args_dir = self._log_dir + '/args'
        # Create a directory for storing args
        if not os.path.isdir(args_dir):
            os.mkdir(args_dir)
        # Save the args in a text file
        with open(args_dir + '/args.txt', 'w') as f:
            for arg in vars(args):
                val = getattr(args, arg)
                if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                    val = f"'{val}'"
                f.write('{}: {}\n'.format(arg, val))
        # Pickle the args for possible reuse
        with open(args_dir + '/args.pickle', 'wb') as f:
            pickle.dump(args, f)

    def create_log(self, log_name: str, key_name: str, *value_names):  # TODO -- add to logs if it already exists!
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self._logs.keys():
            raise Exception('Log already exists!')
        # Add to existing logs
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with open(self._log_dir + f'/{log_name}.csv', 'w') as f:
            f.write(','.join((key_name,) + value_names) + '\n')

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        """
        if log_name not in self._logs.keys():
            raise Exception('Log not existent!')
        if len(values) != len(self._logs[log_name][1]):
            raise Exception('Not all required values are logged!')
        # Write a new line with the given values
        with open(self._log_dir + f'/{log_name}.csv', 'a') as f:
            f.write(','.join(str(v) for v in (key,) + values) + '\n')

    def create_image_folder(self, folder_name: str):
        """
        Create a folder used for logging images
        :param folder_name: The name of the folder
        """
        if folder_name in self._image_folders.keys():
            raise Exception('Image folder already exists!')
        # Register as image folder
        self._image_folders[folder_name] = []
        # Create folder
        # Ensure the log directory exists
        if not os.path.isdir(self._log_dir + f'/{folder_name}'):
            os.mkdir(self._log_dir + f'/{folder_name}')

    # def log_image(self,
    #               folder_name: str,
    #               image_name: str,
    #               image: np.ndarray
    #               ):
    #     if folder_name not in self._image_folders.keys():
    #         raise Exception('Image folder not registered!')
    #     if not os.path.isdir(self._log_dir + f'/{folder_name}'):
    #         raise Exception('Image folder does not exist!')
    #
    #     pass  # TODO

    def log_observations(self,
                         folder_name: str,
                         image_name: str,
                         o_true: torch.Tensor,
                         o_pred: torch.Tensor,
                         extension: str = 'png'
                         ):
        """
        Log two observation tensors:
            - The true observation obtained from the true environment
            - The predicted observation obtained from the learned environment model
        :param folder_name: The name of the folder in which the observations are stored
        :param image_name: The name of the file in which the observations should be stored
        :param o_true: The true observation tensor
                        shape: (num_channels, width, height)
        :param o_pred: The predicted observation tensor
                        shape: (num_channels, width, height)
        :param extension: determines in what format the file is stored. (.png by default)
        """
        if folder_name not in self._image_folders.keys():
            raise Exception('Image folder not registered!')
        if not os.path.isdir(self._log_dir + f'/{folder_name}'):
            raise Exception('Image folder does not exist!')

        path = self._log_dir + f'/{folder_name}/{image_name}.{extension}'

        observations = torchvision.utils.make_grid([o_true, o_pred], range=(0, 255), normalize=True)
        torchvision.utils.save_image(observations, path)

        self._image_folders[folder_name].append(f'{image_name}.{extension}')

    def log_n_observations(self,  # TODO -- had no time to implement this properly (merge with function above)
                           folder_name: str,
                           image_name: str,
                           observations: list,
                           extension: str = 'png'
                           ):
        """
        Log multiple observation tensors
        :param folder_name: The name of the folder in which the observations are stored
        :param image_name: The name of the file in which the observations should be stored

        :param extension: determines in what format the file is stored. (.png by default)
        """
        if folder_name not in self._image_folders.keys():
            raise Exception('Image folder not registered!')
        if not os.path.isdir(self._log_dir + f'/{folder_name}'):
            raise Exception('Image folder does not exist!')

        path = self._log_dir + f'/{folder_name}/{image_name}.{extension}'

        observations = torchvision.utils.make_grid(observations, range=(0, 255), normalize=True)
        torchvision.utils.save_image(observations, path)

        self._image_folders[folder_name].append(f'{image_name}.{extension}')

    def log_model(self, model_name: str, model: torch.nn.Module):
        """
        Save model (including weights) to disk
        :param model_name: the name of the model
        :param model: the model to save
        """
        path = f'{self._log_dir}/{model_name}.pth'
        torch.save(model, path)

    def log_model_state(self, model_name: str, model: torch.nn.Module):
        """
        Save model state to disk. This includes both the weights and internal state in case of RSSM
        :param model_name: the name of the model
        :param model: the model of which the state should be saved
        """
        state = {'weights': model.state_dict()}
        if isinstance(model, RSSM):
            state['state'] = model.get_state()

        path = f'{self._log_dir}/{model_name}_state.pth'
        torch.save(state, path)

    def load_model(self, model_name: str) -> torch.nn.Module:
        """
        Load the model that was saved to disk
        :param model_name: the filename in which the model was stored (without extension)
        """
        path = f'{self._log_dir}/{model_name}.pth'
        model = torch.load(path)
        return model

    def load_model_state(self):  # TODO
        pass

    def log_dataset(self,
                    file_name: str,
                    dataset: ExperienceReplay,
                    ):
        """
        Save a dataset to disk
        :param file_name: the name of the file to which it is saved (without extension)
        :param dataset: the dataset that should be saved
        """
        path = f'{self._log_dir}/{file_name}.pth'
        torch.save(dataset, path)

    def load_dataset(self, file_name: str) -> ExperienceReplay:
        """
        Load a dataset from disk
        :param file_name: the name of the file to which it is saved (without extension)
        :return: the loaded dataset
        """
        path = f'{self._log_dir}/{file_name}.pth'
        dataset = torch.load(path)
        return dataset
