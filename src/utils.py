import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from datetime import datetime
import math
from shutil import copyfile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_dir(dirname, recursive=False, verbose=True):
    """This function creates a directory
    in case it doesn't exist"""
    try:
        # Create target Directory
        os.mkdir(dirname)
        if verbose:
            print("Directory ", dirname, " was created")
    except FileExistsError:
        if verbose:
            print("Directory ", dirname, " already exists")
    return dirname


class ValidationAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
        self.current_best_accuracy_dict = {}
        for dataset in self.datasets:
            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

    def is_better(self, accuracies_dict):
        is_better = False
        is_better_count = 0
        for i, dataset in enumerate(self.datasets):
            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
                is_better_count += 1

        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
            is_better = True

        return is_better

    def replace(self, accuracies_dict):
        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "Validation Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line

    def get_current_best_accuracy_dict(self):
        return self.current_best_accuracy_dict


def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the log file.
    """
    print(message)
    log_file.write(message + '\n')


def get_log_files(checkpoint_dir, mode='', test_path=None, resume=0, feature_adaptation=''):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    if mode=='test': # test mode
        unique_checkpoint_dir = os.path.dirname(test_path)
    elif resume>0:   # train mode, resume
        print('resume model!')
        unique_checkpoint_dir = os.path.dirname(checkpoint_dir)
        checkpoint_path_final = os.path.join(unique_checkpoint_dir, 'fully_trained.pt')
        if not os.path.exists(checkpoint_path_final):
            print('fully_trained.pt not exists, copy from ', checkpoint_dir)
            copyfile(checkpoint_dir, checkpoint_path_final)
        else: 
            print('fully_trained.pt already exists')
    else:            # train mode from scratch
        unique_checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'-'+feature_adaptation)

    if not os.path.exists(unique_checkpoint_dir):
        os.makedirs(unique_checkpoint_dir)

    checkpoint_path_validation = os.path.join(unique_checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(unique_checkpoint_dir, 'fully_trained.pt')
    if mode=='test':
        logfile_path = os.path.join(unique_checkpoint_dir, 'logtest')
        logfile = open(logfile_path, "a")
    else:
        logfile_path = os.path.join(unique_checkpoint_dir, 'log')
        logfile = open(logfile_path, "a")

    return unique_checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final


def stack_first_dim(x):
    """
    Method to combine the first two dimension of an array
    """
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tensor - mean parameter of the distribution
    :param var: tensor - variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tensor - samples from distribution of size numSamples x dim(mean)
    """
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()


def loss(test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    log(\sum_{i=1}^N{p_y}). log average of target probabilities. similar to CEloss
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)

def loss2(test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    Cross Entropy Loss
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = F.cross_entropy(test_logits_sample[sample], test_labels, reduction='mean')
    #score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    #return -torch.sum(score, dim=0)
    return log_py.sum()

def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())

