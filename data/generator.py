import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf

from config import *
from misc.utils import *
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder


class DataGenerator:

    def __init__(self, args):
        """ Data Generator

        Generates batch-iid and batch-non-iid under both
        labels-at-client and labels-at-server scenarios.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """

        self.args = args
        self.base_dir = os.path.join(self.args.output_path, self.args.task) 
        self.shape = (32,32,3)

    def generate_data(self):
        print('generating {} ...'.format(self.args.task))
        start_time = time.time()
        self.task_cnt = -1
        self.is_labels_at_server = True if 'server' in self.args.scenario else False
        self.is_imbalanced = True if 'imb' in self.args.task else False
        x = self.load_dataset(self.args.dataset_id)
        self.generate_task(x)
        print(f'{self.args.task} done ({time.time()-start_time}s)')

    """ def load_dataset(self, dataset_id):
        temp = {}
        if self.args.dataset_id_to_name[dataset_id] == 'cifar_10':
            temp['train'] = datasets.CIFAR10(self.args.dataset_path, train=True, download=True) 
            temp['test'] = datasets.CIFAR10(self.args.dataset_path, train=False, download=True) 
            x, y = [], []
            for dtype in ['train', 'test']:
                for image, target in temp[dtype]:
                    x.append(np.array(image))
                    y.append(target)

        x, y = self.shuffle(x, y)
        print(f'{self.args.dataset_id_to_name[self.args.dataset_id]} ({np.shape(x)}) loaded.')
        return x, y """


    def load_dataset(self, dataset_id):
        if self.args.dataset_id_to_name[dataset_id] == 'pacs':
            # Define a transform to convert images to tensor and normalize
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing with ImageNet stats
            ])
            
            # Load the 'art_painting' domain for the server with labels
            server_domain = 'art_painting'
            server_dataset = ImageFolder(os.path.join(self.args.dataset_path, 'PACS', server_domain), transform=transform)
            server_images, server_labels = zip(*[(image.numpy().transpose(1,2,0), target) for image, target in server_dataset])
            
            # Load the 'cartoon' and 'photo' domains for clients without labels
            client_domains = ['cartoon', 'photo']
            client_datasets = {domain: ImageFolder(os.path.join(self.args.dataset_path, 'PACS', domain), transform=transform) for domain in client_domains}
            client_images = {domain: [image.numpy().transpose(1,2,0) for image, _ in client_datasets[domain]] for domain in client_domains}
            
            # Load the 'sketch' domain for testing with labels
            test_domain = 'sketch'
            test_dataset = ImageFolder(os.path.join(self.args.dataset_path, 'PACS', test_domain), transform=transform)
            test_images, test_labels = zip(*[(image.numpy().transpose(1,2,0), target) for image, target in test_dataset])

            # Return a dictionary of datasets
            datasets = {
                'server': {'images': server_images, 'labels': server_labels},
                'clients': {domain: images for domain, images in client_images.items()},
                'test': {'images': test_images, 'labels': test_labels}
            }
            return datasets


    def generate_task(self, datasets):
        self.split_train_test_valid(datasets['test'])
        #s, u = self.split_s_and_u(x_train, y_train)
        self.split_s(datasets['server'])
        self.split_u(datasets['clients'])

    def split_train_test_valid(self,s):
        
        x,y = self.shuffle(s['images'],s['labels'])
        self.save_task({
            'x': x,
            'y': tf.keras.utils.to_categorical(y, 7),
            'labels':np.unique(s['labels']),
            'name': f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        
    def split_s(self, s):
        x_labeled = []
        y_labeled = []
        x_labeled = [*x_labeled, *s['images']]
        y_labeled = [*y_labeled, *s['labels']]
        x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
        self.save_task({
            'x': x_labeled,
            'y': tf.keras.utils.to_categorical(y_labeled, 7),
            'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
            'labels': np.unique(y_labeled)
        })

    def split_u(self, u):
                # batch-imbalanced
        x_unlabeled = []
        y_unlabeled = []
        i=0
        for d in u.items():
            x_unlabeled = [*x_unlabeled, *d[1]]
            x_unlabeled = np.array(x_unlabeled)  
            self.save_task({
                'x': x_unlabeled,
                'y':'null',
                'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                'labels': 'null'
                }) 
            i+=1   

    def save_task(self, data):
        np_save(base_dir=self.base_dir, filename=f"{data['name']}.npy", data=data)
        print(f"filename:{data['name']}, labels:[{','.join(map(str, data['labels']))}], num_examples:{len(data['x'])}")
    
    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random.seed(self.args.seed)
        random.shuffle(idx)
        return np.array(x)[idx], np.array(y)[idx]











        