import os
from PIL import Image

import torch
from torch.utils.data import Dataset
# from skimage import io
#import cv2 as cv
#from cv2 import resize

try:
    from cv2 import resize
    import cv2 as cv
except Exception as e:
    print("\n\t can not import cv2")


try:
    from modules.utils import generate_phoc_vector, generate_phos_vector, set_phos_version, set_phoc_version
except Exception as e:
    from ResPhoSCNetZSL.modules.utils import generate_phoc_vector, generate_phos_vector, set_phos_version, set_phoc_version
    
# from utils import generate_phoc_vector, generate_phos_vector, set_phos_version, set_phoc_version

import pandas as pd
import numpy as np


class phosc_dataset(Dataset):
    def __init__(self,args, data_dict,  language='eng', transform=None):
        
        print("\n\t args.lang:",args.lang)
        
        set_phos_version(args.lang)
        set_phoc_version(args.lang)

        self.args = args
        self.data_dict = data_dict
        self.df_all = dict()
        self.allWords = []
        self.wordPhosc = dict()
        print("\n\t keys tot:",len(self.data_dict.keys()))


    def getPhosc(self):

        for key in self.data_dict.keys():
            self.allWords.append(self.data_dict[key]["label"])

        for wordValue in set(self.allWords):
            
            wordValue = wordValue.replace(" ","")
            wordValue = wordValue.replace("_","")

            
            #print("\n\t wordValue:",wordValue)
            phos = generate_phos_vector(wordValue)
            
            if self.args.phosc == 1:
                phoc = np.array(generate_phoc_vector(wordValue),dtype=np.float32)
            
            if self.args.phosc == 1:  
                phosc = np.concatenate((phos, phoc))
            elif self.args.phos == 1:
                phosc = phos
                

            self.wordPhosc[wordValue] = phosc.astype(np.int64)

        
        return self.wordPhosc


    def __getitem__(self, index):
        #img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
        #image = cv.imread(img_path)


        try:
            img_path = os.path.join(self.root_dir, self.df_all.loc[index,"Image"])
        except Exception as e:
            img_path = os.path.join(self.root_dir, self.df_all.loc[index,"Images"])
        
        #print("\n\t path:",img_path," \t is file:",os.path.isfile(img_path))

        try:
            image = cv.imread(img_path)
        except Exception as e:
            image = Image.open(img_path)
    
        try:
            image=resize(image, (250, 50))
        except Exception as e:
            image = image.resize((250, 50), Image.ANTIALIAS)
    
        # print(image.shape)

        if self.transform:
            image = self.transform(image)
        word = self.df_all.iloc[index, 1]

        phos = torch.tensor(self.df_all.iloc[index, -3])
        phoc = torch.tensor(self.df_all.iloc[index, -2])
        phosc = torch.tensor(self.df_all.iloc[index, -1])

        item = {
            'image': image.float(),
            'word': word,
            'y_vectors': {
                'phos': phos.float(),
                'phoc': phoc.float(),
                'phosc': phosc.float()
            }
        }

        return item
        # return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)

class CharacterCounterDataset(Dataset):
    def __init__(self, longest_word_len, csvfile, root_dir, transform=None):
        self.df_all = pd.read_csv(csvfile)
        self.root_dir = root_dir
        self.transform = transform

        words = self.df_all["Word"].values

        targets = []

        for word in words:
            target = np.zeros((longest_word_len))
            target[len(word)-1] = 1
            targets.append(target)

        self.df_all["target"] = targets

        # print(self.df_all)

        # print(self.df_all.iloc[0, 5].shape)
        # print(self.df_all.to_string())

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
        image = cv.imread(img_path)

        y = torch.tensor(self.df_all.iloc[index, len(self.df_all.columns) - 1])

        if self.transform:
            image = self.transform(image)

        # returns the image, target vector and the corresponding word
        return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)


if __name__ == '__main__':
    from torchvision.transforms import transforms

    # dataset = phosc_dataset('image_data/IAM_Data/IAM_valid_unseen.csv', 'image_data/IAM_Data/IAM_valid', 'nor', transform=transforms.ToTensor())
    dataset = phosc_dataset('image_data/GW_Data/cv1_valid_seen.csv', 'image_data/GW_Data/CV1_valid', 'eng', transform=transforms.ToTensor())
    # dataset = phosc_dataset('image_data/norwegian_data/train_gray_split1_word50.csv', 'image_data/norwegian_data/train_gray_split1_word50', 'nor', transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, 5)
    # print(dataset.df_all)


    for batch in dataloader:
        print(batch['image'].shape)
        print(batch['y_vectors']['phos'].shape)
        print(batch['y_vectors']['phoc'].shape)
        print(batch['y_vectors']['phosc'].shape)
        quit()

    # print(dataset.__getitem__(0))
