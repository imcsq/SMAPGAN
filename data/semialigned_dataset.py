import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image


class SemialignedDataset(BaseDataset):
    """A dataset class for both paired & unpaired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.AB_size = len(self.AB_paths)  # get the size of paired dataset
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
        self.img_pool = [(1, path) for path in self.AB_paths]    # 1 stands for paired data.
        
        # ==== UNPAIRED PART
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))
        
        if self.A_size < self.B_size: # 
            self.A_paths += random.sample(self.A_paths, self.B_size - self.A_size)
            self.unpair_size = self.B_size
        else:
            self.B_paths += random.sample(self.B_paths, self.A_size - self.B_size)
            self.unpair_size = self.A_size

        self.img_pool += [(0, pathA, pathB) for pathA, pathB in zip(self.A_paths, self.B_paths)]    # 0 stands for unpaired data.
        
        random.shuffle(self.img_pool)
        random.shuffle(self.img_pool)
        random.shuffle(self.img_pool)
                                
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            
            datatype (int) - - 0 for unpaired data, 1 for paired data
        """
        
        if self.img_pool[index][0] == 1: # paired data
            AB_path = self.img_pool[index][1]
            AB = Image.open(AB_path).convert('RGB')
            # split AB image into A and B
            w, h = AB.size
            w2 = int(w / 2)
            
            A_path = AB_path
            B_path = AB_path
            
            A_img = AB.crop((0, 0, w2, h))
            B_img = AB.crop((w2, 0, w, h))
        else:
            A_path = self.img_pool[index][1]
            B_path = self.img_pool[index][2]
            
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
                                
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'datatype': self.img_pool[index][0]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_pool)
