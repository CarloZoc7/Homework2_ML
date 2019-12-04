!pip3 install --upgrade pillow

from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        f = open(self.split , "r")
        
        samples_file = [str(row).replace("\n", "") for row in f]
        self.labels = [label.split("/")[0] for label in samples_file] # if label.split("/")[0] != 'BACKGROUND_Google' to remove label of background
        self.dir_labels = {}

        for key, label in enumerate(self.labels):
          if label not in self.dir_labels.keys():
            self.dir_labels[label] = key

        print(len(self.dir_labels))
        self.elements = []

        for sample in samples_file:
          self.elements.append(pil_loader(root+"/"+sample))
       
        self.final_labels = []
        self.images_tensor = []

        for i in range(len(self.elements)):
          image, label = self.__getitem__(i)  

          self.images_tensor.append(image)
          self.final_labels.append(label)

        self.elements.clear()
        self.labels.clear()
        self.dir_labels.clear()
        return None

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        # image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int
        
        image = self.elements[index]
        lables_name = self.labels[index]
        label = self.dir_labels[lables_name]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.final_label) # Provide a way to get the length (number of elements) of the dataset
        return length

    def get_dataset_labels(self):
      return self.images_tensor, self.final_labels