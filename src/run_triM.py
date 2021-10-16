"""
Script to reproduce the multi-domain few-shot classification results on Meta-Dataset in:
"A Multi-Mode Modulator for Multi-Domain Few-Shot Classification"
https://csyanbin.github.io/papers/ICCV2021_tri-M.pdf

Requirements:
    Tensorflow 2
    Pytorch 1.0+
    GPU with 16GB+ memory

Set Enviroment:
    <First>   Download&Process Meta-Dataset following: 
              https://github.com/google-research/meta-dataset#downloading-and-converting-datasets
    <Second>  Download&Process 3 extra datasets (Mnist, Cifar10, Cifar100) following: 
              https://github.com/cambridge-mlg/cnaps --> Installation --> 3. Install additional test datasets (MNIST, CIFAR10, CIFAR100)
    <Third>   Set the PROJECT_ROOT, META_DATASET_ROOT, and META_RECORDS_ROOT in datareader/path.py
              ulimit -n 50000

Training:
    python run_triM.py --learning_rate 2e-3 --feature_adaptation MahSpecCoop -T 150000 --tasks_per_batch=16 
Testing:
    python run_triM.py --learning_rate 2e-3 --feature_adaptation MahSpecCoop -T 150000 --tasks_per_batch=16 --test_model_path TEST_MODEL_CKPT_PATH --mode test --test_datasets=traffic_sign 

"""

import torch
import numpy as np
import argparse
import random
import os
import sys
from utils import print_and_log, get_log_files, ValidationAccuracies, loss, loss2, aggregate_accuracy
from model_maha import TriMMaha
from datareader.meta_dataset_reader import MetaDatasetReader, SingleDatasetReader
import torch.nn.functional as F
import conf_dict
from torch.optim import lr_scheduler

import time
from shutil import copyfile

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warnings
tf.compat.v1.disable_eager_execution()

#NUM_TRAIN_TASKS = 150000
NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 600
SAVE_FREQUENCY = 5000
VALIDATION_FREQUENCY = 10000


def set_random_seed(logfile, seed=0):
    if seed>=0:
        print_and_log(logfile, 'Fix seed:'+str(seed))
        # tensorflow
        #tf.set_random_seed(seed)
        tf.random.set_seed(seed)
        # python
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED']=str(seed)
        # pytorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
    else:
        print_and_log(logfile, 'Seed<0, random test')


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.mode, self.args.test_model_path, self.args.resume, self.args.feature_adaptation)

        if os.path.exists(self.checkpoint_dir) and not os.path.exists(os.path.join(self.checkpoint_dir,"py")):
            os.makedirs(os.path.join(self.checkpoint_dir,"py"))
            os.makedirs(os.path.join(self.checkpoint_dir,"py/datareader"))
            config_dict = conf_dict.choices_dict
            config = config_dict[self.args.feature_adaptation]
            file_list = [config["feature_adaptation_file"]+".py", "run_triM.py", "model_maha.py", "modules.py", "set_encoder.py", "conf_dict.py", "config_networks.py", "resnet.py", "utils.py"]
            for name in file_list:
                copyfile(name, os.path.join(self.checkpoint_dir,"py",name))
            file_list = ["meta_dataset_config.gin", "meta_dataset_processing.py", "meta_dataset_reader.py", "paths.py", "pipeline.py"]
            for name in file_list:
                copyfile(os.path.join('datareader', name), os.path.join(self.checkpoint_dir,"py/datareader/",name))

        if self.args.mode=="test":
            #tf.reset_default_graph()
            tf.compat.v1.reset_default_graph()
            set_random_seed(self.logfile, self.args.seed)
        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        
        # print number of learnable parameters
        param_num = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_num += torch.numel(param)
        print("Total Number of Parameters:", param_num)

        # Dataset reader
        self.train_set, self.validation_set, self.test_set = self.init_data()
        if self.args.dataset == "meta-dataset":
            self.metadataset = MetaDatasetReader(self.args.data_path, self.args.mode, self.train_set, self.validation_set,
                                             self.test_set, max_way_train=40, max_way_test=50,
                                             max_support_train=400, max_support_test=500, image_size=self.args.image_size,
                                             shuffle=self.args.shuffle_dataset)
        else:
            self.metadataset = SingleDatasetReader(self.args.data_path, self.args.mode, self.args.dataset, way=5,
                                               shot=1, query_train=10, query_test=10)
        
        ## Loss
        if self.args.CE: # Use CE Loss instead of log average of target probability in CNAPs 
            self.loss = loss2
        else:
            self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        ## Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        self.validation_accuracies = ValidationAccuracies(self.validation_set)
        if self.args.resume>0:
            ckpt_path = self.checkpoint_path_final
            self.load_checkpoint(ckpt_path)
            print(self.optimizer)
            print(self.start_iteration)
        else:
            self.start_iteration = -1

        # lr_scheduler
        if self.args.learning_rate_scheduler=="step":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma, last_epoch=self.start_iteration) # 16w tasks=1w batches: [0,2500]:2e-3,1e-3,5e-4,2.5e-4
        elif self.args.learning_rate_scheduler=="exp":
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma, last_epoch=self.start_iteration) # 16w tasks=1w batches: 0.8**10=0.1, 0.8**5=0.33, 0.8**2=0.64

        self.optimizer.zero_grad()


    def init_model(self, test=False):
        model = TriMMaha(device=self.device, args=self.args).to(self.device)
        model.train()  # set encoder is always in train mode to process context data
        if test==True:
            print("feature adaptation network in eval mode")
            model.feature_adaptation_network.eval()
        model.feature_extractor.eval()  # feature extractor is always in eval mode
        return model

    def init_data(self):
        if self.args.dataset == "meta-dataset":
            train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
            validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower', 'mscoco']
            test_set = self.args.test_datasets
        else: # Only for 1 dataset
            train_set = [self.args.dataset]
            validation_set = [self.args.dataset]
            test_set = [self.args.dataset]

        return train_set, validation_set, test_set

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--image_size", type=int, default=84, help="Input image resolution.")
        parser.add_argument("--dataset", choices=["meta-dataset", "ilsvrc_2012", "omniglot", "aircraft", "cu_birds",
                                                  "dtd", "quickdraw", "fungi", "vgg_flower", "traffic_sign", "mscoco",
                                                  "mnist", "cifar10", "cifar100"], 
                                        default="meta-dataset", help="Dataset to use.")
        parser.add_argument('--test_datasets', nargs='+', help='Datasets to use for testing',
                            default=["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi",
                                     "vgg_flower", "traffic_sign", "mscoco", "mnist", "cifar10", "cifar100"])

        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--pretrained_resnet_path", default="../models/pretrained_resnet.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")

        parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument("--learning_rate_scheduler", "-lrsche", type=str, choices=["fix", "step", "exp"], default="fix", help="Learning rate scheduler.")
        parser.add_argument("--step_size", "-step", type=int, default=2500, help="Learning rate decay step_size.")
        parser.add_argument("--gamma", "-g", type=float, default=0.5, help="Learning rate decay gamma.")
        parser.add_argument("--weight_decay", "-WD", type=float, default=0.0, help="weight decay")

        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--feature_adaptation", choices=conf_dict.choices, default="MahSpecCoop",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
        parser.add_argument('--NUM_TRAIN_TASKS', '-T', type=int, default=150000, help='number of train tasks')

        parser.add_argument("--reg_scale", "-reg", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--seed", type=int, default=0,
                            help="random seed for the test set episodes.")
        parser.add_argument('--CE', dest='CE', action='store_true')
        parser.add_argument('--no-CE', dest='CE', action='store_false')
        parser.set_defaults(CE=False)

        ## Domain Classification Loss done by CE between gate and domain identifier
        parser.add_argument("--gate_loss", default="ce", help="Domain classification loss function")
        parser.add_argument("--gate_reg_mode", default="fix", choices=['fix', 'linear', 'linear1', 'fix1'], help="Domain classificaiton regularization mode")
        parser.add_argument("--gate_reg_val", type=float, default=0.1, help="Domain classification regularization value.")

        ## Following https://github.com/google-research/meta-dataset/issues/54, the evaluation on Traffic Sign should be done on shuffled samples
        parser.add_argument("--shuffle_dataset", type=bool, default=True,
                                            help="As per default, shuffles images before task generation. Set False to re-create paper results, and True for leaderboard results.")

        args = parser.parse_args()
        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            if self.args.mode == 'train' or self.args.mode == 'train_test':
                train_accuracies = []
                losses = []
                total_iterations = self.args.NUM_TRAIN_TASKS
                tic = time.time()
                for iteration in range(self.args.resume, total_iterations):
                    torch.set_grad_enabled(True)
                    task_dict, idx = self.metadataset.get_train_task(session)
                    task_loss, task_accuracy = self.train_task(task_dict, idx, iteration)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.args.learning_rate_scheduler in ["step", "exp"]:
                            self.scheduler.step()
                            lr = self.scheduler.get_lr()[0]

                    if (iteration + 1) % 1000 == 0:    # print training stats
                        print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}, time:{:.4f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item(), time.time()-tic) )
                        tic = time.time()
                        train_accuracies = []
                        losses = []

                    if (iteration + 1) % SAVE_FREQUENCY == 0:
                        save_ckpt = self.checkpoint_path_validation.replace("best_validation", str(iteration+1))
                        print('save ckpt', save_ckpt)
                        self.save_checkpoint(int((iteration+1)/self.args.tasks_per_batch), save_ckpt)

                    if ((iteration + 1) % VALIDATION_FREQUENCY == 0) and (iteration + 1) != total_iterations:
                        # validate
                        accuracy_dict = self.validate(session)
                        self.validation_accuracies.print(self.logfile, accuracy_dict)
                        # save the model if validation is the best so far
                        if self.validation_accuracies.is_better(accuracy_dict):
                            self.validation_accuracies.replace(accuracy_dict)
                            self.save_checkpoint(int((iteration+1)/self.args.tasks_per_batch), self.checkpoint_path_validation)
                            print_and_log(self.logfile, 'Best validation model was updated.')
                            print_and_log(self.logfile, '')

                # save the final model
                self.save_checkpoint(int((total_iterations)/self.args.tasks_per_batch), self.checkpoint_path_final)

            if self.args.mode == 'train_test':
                set_random_seed(self.logfile, self.args.seed) # set random seed to get deterministic results
                self.test(self.checkpoint_path_final, session)
                self.test(self.checkpoint_path_validation, session)

            if self.args.mode == 'test':
                self.test(self.args.test_model_path, session)

            self.logfile.close()


    def train_task(self, task_dict, idx, iteration=-1):
        context_images, target_images, context_labels, target_labels, idx = self.prepare_task(task_dict, idx)

        if True: 
            reg = self.args.gate_reg_val
            if iteration!=-1:
                if self.args.gate_reg_mode=='linear': ## linearly increase reg coeficient
                    reg = 0.02+min(reg-0.02, (iteration/self.args.NUM_TRAIN_TASKS)*(reg-0.02))
                elif self.args.gate_reg_mode=='linear1': ## linear increase then linear decrease 
                    sign = 2*(int(iteration<0.5*self.args.NUM_TRAIN_TASKS))-1 # pos/neg
                    reg = reg + sign*((iteration/(0.5*self.args.NUM_TRAIN_TASKS)-1) * (reg-0.01))
                elif self.args.gate_reg_mode=='fix1': # fix for [0,1w] and 0 for next iters
                    reg = int(iteration<10000) * reg
                
            target_logits, gate_logits = self.model(context_images, context_labels, target_images, mode='train', iteration=iteration)
            ## Domain classification loss 
            if self.args.gate_loss=='ce':
                celoss = torch.nn.CrossEntropyLoss()
                gate_loss = reg * celoss(gate_logits.view(1,-1), idx.view(-1))
        else:
            target_logits = self.model(context_images, context_labels, target_images, mode='train', iteration=iteration)

        ## Accumulate all losses: task_loss + domain_loss (gate_loss) + regularizer_loss
        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_loss += gate_loss
        regularization_term = (self.model.feature_adaptation_network.regularization_term())
        regularizer_scaling = self.args.reg_scale
        task_loss += regularizer_scaling * regularization_term
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def validate(self, session):
        self.model.feature_adaptation_network.eval()
        print(self.model.feature_adaptation_network.training)
        with torch.no_grad():
            accuracy_dict ={}
            for item in self.validation_set:
                accuracies = []
                for _ in range(NUM_VALIDATION_TASKS):
                    task_dict, _ = self.metadataset.get_validation_task(item, session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    target_logits, gate_logits= self.model(context_images, context_labels, target_images, mode='val', iteration=-1)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}

        self.model.feature_adaptation_network.train()
        print(self.model.feature_adaptation_network.training)
        return accuracy_dict

    def test(self, path, session):
        self.model = self.init_model(test=True)
        self.load_checkpoint(path)

        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))

        with torch.no_grad():
            NUM_TEST_TASKS = 600
            import time
            tic = time.time()
            for item in self.test_set:
                accuracies = []
                for _ in range(NUM_TEST_TASKS):
                    task_dict, _ = self.metadataset.get_test_task(item, session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    target_logits, gate_logits = self.model(context_images, context_labels, target_images, mode='test', iteration=path)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                print_and_log(self.logfile, '{0:3.1f}+/-{1:2.1f}'.format(accuracy, accuracy_confidence))
            toc = time.time()
            print("Time elapsed:", toc-tic)

    def prepare_task(self, task_dict, idx_np=None):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)

        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        target_images_np, target_labels_np = self.shuffle(target_images_np, target_labels_np)
        target_images = torch.from_numpy(target_images_np)
        target_labels = torch.from_numpy(target_labels_np)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        if idx_np is not None:
            idx = torch.tensor(idx_np)
            idx = idx.type(torch.LongTensor).to(self.device)
            return context_images, target_images, context_labels, target_labels, idx

        return context_images, target_images, context_labels, target_labels

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def save_checkpoint(self, iteration, save_path):
        torch.save({
            'args': self.args, 
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.validation_accuracies.get_current_best_accuracy_dict(),
        }, save_path)

    def load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            assert False, "Model is not found at {}".format(ckpt_path)

        checkpoint = torch.load(ckpt_path)
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            print("train args:{}",format(checkpoint['args']))
            print("running args:{}",format(self.args))
            print("best training iteration:{}",format(checkpoint['iteration']))
            
            self.start_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_validation = checkpoint.get('best_accuracy', None)
            if best_validation is not None:
                self.validation_accuracies.replace(best_validation)




if __name__ == "__main__":
    main()
