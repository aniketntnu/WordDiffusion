"""
    1. non-phosc
    for diffusion without PHOSC change
    gt_train , csvRead
    MAX_CHARS = 42
    
    
    2. phosc
    trascriptionPlusOCR
    phosc,phos
    MAX_CHARS = 10
    
    checklist change following options
    
    from config.py
    
    MAX_CHARS, gt_train, csvRead, authorBasePath, ckptModelName, emaModelName, save_path,saveModelName
    
    # phosc, phos, trascriptionPlusOCR

    
"""


import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from torch import optim
import random
import copy
import argparse
import json
from diffusers import AutoencoderKL
#from hiGanBase import hiNetModel
import sys

#from hiGan.networks import BigGAN_networks as hiModel
#from hiGan.lib import alphabets
#from hiGan.lib.alphabet import strLabelConverter

from unet import UNetModel
#from unetOriginal import UNetModel
from unetPhosc import UNetModelPhosc

import wandb
import pandas as pd
from  ResPhoSCNetZSL.modules.datasets import phosc_dataset
#from utils.dumpImages import dump_images
#from utils.dumpImages import dump_images

import pickle
from config import *
import torch.nn.functional as F

import torchvision.transforms as transforms
from PIL import Image, ImageDraw

MAX_CHARS = MAX_CHARS

print("\n\t MAX_CHARS = :",MAX_CHARS)

OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>
c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
cdict = {c:i for i,c in enumerate(c_classes)}
icdict = {i:c for i,c in enumerate(c_classes)}

#label_converter = strLabelConverter("all")
ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / ly.shape[0]

import random
import torch
import numpy as np

import random
import torch


def dump_images2(imgNames,images_tensor, output_dir):

  # Get shapes and determine line params
  
  print("\n\t images_tensor.shape:",images_tensor.shape)
  batch_size, channels, height, width = images_tensor.shape
  num_lines = random.randint(10,20)
  x_coords = torch.randint(0,width,(batch_size*num_lines,)) 

  # Draw lines directly on tensor
  for x in x_coords:
    images_tensor[:,:,:,x] = 1

  return images_tensor

"""

def dump_images(imgNames,images_tensor, output_dir):

  # Convert tensor and get shapes
  images_array = images_tensor.permute(0,2,3,1).numpy()   
  batch_size, height, width, channels = images_array.shape  

  # Create canvas and directly paste full images array
  canvas_array = np.zeros((batch_size*height, width*batch_size, channels))
  canvas_array[:,::width] = images_array  

  # Determine line params without looping
  num_lines = random.randint(10,20)  
  x_coords = [random.randint(0,width-1) for _ in range(num_lines)]

  # Draw lines directly on canvas array
  for x in x_coords:
    canvas_array[:, x] = 255
    
  canvas_array = torch.from_numpy(canvas_array).permute(0,3,1,2)
  images_tensor += canvas_array
    
  # Return modified tensor without cropping
  return images_tensor 
"""

def dump_images(imgNames,tensor, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Denormalize the tensor and convert to PIL images
    denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    tensor = denorm(tensor)
    images = [transforms.ToPILImage()(img.clamp(0, 1)) for img in tensor]
    
    #modified_tensors = [] #images.clone()
    for i, img in enumerate(images):
        #print("\n\t PIL img.shape:",len(img.getbands()))

        draw = ImageDraw.Draw(img)
        width, height = img.size
        num_lines = random.randint(10, 20)  # Random number of lines between 10 and 20
        for _ in range(num_lines):
            x = random.randint(0, width)  # Random x-coordinate
            draw.line([(x, 0), (x, height)], fill=(255,), width=6)  # Draw white line
        nm = imgNames[i]
        img_path = os.path.join(output_dir, f"{nm}_{i}.png")
        
        img = img.convert("RGB")
        #img.save(img_path)
        
        # Convert the modified PIL image back to a tensor and append it to the list
        img = transforms.ToTensor()(img)
        #print("\n\t tensor img.shape:",img.shape)
        
        #modified_tensors.append(modified_tensor)
    
    return img #modified_tensors




def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    
    labels = labels.replace(" ", "_")
    #print("\n\t labels:",labels)
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    #ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    # create json object from dictionary if you want to save writer ids
    json_dict_l = json.dumps(letter2index)
    l = open("letter2index.json","w")
    l.write(json_dict_l)
    l.close()
    index2letter = {v: k for k, v in letter2index.items()}
    json_dict_i = json.dumps(index2letter)
    l = open("index2letter.json","w")
    l.write(json_dict_i)
    l.close()
    return len(labels), letter2index, index2letter


char_classes, letter2index, index2letter = labelDictionary()
tok = False
if not tok:
    tokens = {"PAD_TOKEN": 52}
else:
    tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
num_tokens = len(tokens.keys())
print('num_tokens', num_tokens)


print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens


def save_images(epoch,images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    print("\n\t path:",path," images.shape:",images.shape)
    im.save(path)
    return im

class IAMDataset(Dataset):
    def __init__(self, full_dict, image_path, writer_dict, args, transforms=None):

        self.data_dict = full_dict
        self.image_path = image_path
        self.writer_dict = writer_dict
    
        self.transforms = transforms
        
        
        
        self.output_max_len = OUTPUT_MAX_LEN
        self.max_len = MAX_CHARS
        self.n_samples_per_class = 16
        self.indices = list(full_dict.keys())
        self.args = args
        
        
        if self.args.imgConditioned ==1:

            # Folder 1: charLevelIamAnnotationProcessed
            folder1 = '/cluster/datastore/aniketag/allData/wordStylist/charLevelIamAnnotationProcessed/'
            self.charImgDict = {}
            for filename in os.listdir(folder1):
                #crop_name = os.path.splitext(filename)[0]
                self.charImgDict[filename] = 1 

            # Folder 2: allCrops_preprocess
            folder2 = '/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/'
            self.wordImgDict = {}
            for filename in os.listdir(folder2):
                #crop_name = os.path.splitext(filename)[0]
                self.wordImgDict[filename] = 1

            print("\n\t No of Keys:",len(self.charImgDict.keys())," ",len(self.wordImgDict.keys()))
            
            """
            # Print the dictionaries
            print("Crops in folder 1 (charLevelIamAnnotationProcessed):")
            for crop_name, filename in crops1.items():
                print(f"{crop_name}: {filename}")

            print("\nCrops in folder 2 (allCrops_preprocess):")
            for crop_name, filename in crops2.items():
                print(f"{crop_name}: {filename}")

            """

                
        if self.args.phos ==1 or self.args.phosc ==1:
        
            phoscClass = phosc_dataset(self.args,self.data_dict)
            #phoscClass = phosc_dataset.getPhosc(self.data_dict)

            if 1:#not os.path.isfile("./wordPhos.pkl"):
                self.wordPhosc = phoscClass.getPhosc()
                
                with open("./wordPhos.pkl", 'wb') as file:
                    # Use pickle.dump() to write the dictionary to the file
                    pickle.dump(self.wordPhosc, file)            

                    print("\n\t new wordPhosc created")
                    
            else:
                with open("./wordPhos.pkl", 'rb') as file:
                    # Use pickle.load() to load the dictionary from the file
                    self.wordPhosc = pickle.load(file)    
                    print("\n\t old wordPhosc read")


            print("\n\t total in phosc/phoc dir is:",len(self.wordPhosc.keys()))

   
        with open("/cluster/datastore/aniketag/allData/wordStylist/writerStyle/cropStyleDict_Numpy.pkl", 'rb') as f:
            # Load the object from the pickle file
            cropStyleDict = pickle.load(f)

        self.cropStyleDict = cropStyleDict

        self.latentPath1 = "/cluster/datastore/aniketag/allData/wordStylist/imageWordLineVae3.pkl"
        self.latentPath2 ="/cluster/datastore/aniketag/allData/wordStylist/imageWordLineVae3OnlyChar.pkl"
        
        if (self.args.vaeFromDict==1 and os.path.isfile(self.latentPath1)):
        
            print("\n\t reading imageWordLineVae3.pkl from the path:",self.latentPath1)
            with open(self.latentPath1,"rb") as f:
                self.imageTesorDict1 = pickle.load(f)        
                                 
            self.imageTesorkeys1 = self.imageTesorDict1.keys()
            
            print("\n\t original word latent keys:",len(self.imageTesorkeys1))

            
        if (self.args.vaeFromDict==1 and os.path.isfile(self.latentPath2)):
        
            print("\n\t reading imageWordLineVae3.pkl from the path:",self.latentPath2)
            with open(self.latentPath2,"rb") as f:
                self.imageTesorDict2 = pickle.load(f)        

            self.imageTesorkeys2 = self.imageTesorDict2.keys()

            print("\n\t original character latent keys:",len(self.imageTesorkeys2))

        self.found = 0
        self.miss = 0
    
        self.dummyTensor = torch.zeros((1, 4, 8, 32), requires_grad=False)
            
    def __len__(self):
        return len(self.indices)
            
    
    def __getitem__(self, idx):

        image_name = self.data_dict[self.indices[idx]]['image']
        
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)
        
        if self.args.phos ==1 or self.args.phosc ==1:
            phoscLabel = self.wordPhosc[label]#.astype(np.float32)
        else:
            phoscLabel = "NeglectMe"        
        
        
        if self.args.vaeFromDict ==0:
            
            if image_name in self.charImgDict.keys():
                img_path = os.path.join("/cluster/datastore/aniketag/allData/wordStylist/charLevelIamAnnotationProcessed/", image_name)
            elif image_name in self.wordImgDict.keys():
                img_path = os.path.join("/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/", image_name)
            else:
                print("\n\t image not found:",image_name)
            
  
            image = Image.open(img_path).convert('RGB')
            image =  image.convert('RGB')
            
            #print("\n\t 1.image =",image.size) # (256,64)

            # Get the number of channels
            num_channels = len(image.getbands())

            #print(f"Number of channels: {num_channels}")


            # Convert the PIL image to a NumPy array
            #image_array = np.array(image)

            # Check the original shape
            #print("Original shape:", image_array.shape) # (64, 256, 3)

            # Add the color channel dimension
            #image_3d = np.expand_dims(image_array, axis=2)

            # The new shape should be (256, 64, 3)
            #print("New shape:", image_3d.shape)            

            #print("\n\t 11.image =",image.shape)

            image = self.transforms(image)
            #print("\n\t 2.image =",image.shape)

            image = dump_images(image_name,image,"./imageDump")
            #print("\n\t 3.image =",image.shape)

            
            #image = torch.from_numpy(image)
            
        elif self.args.vaeFromDict ==1:
            
            #print("\n\t %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            try:
                # check in word dict
                
                try:
                    imageGlyphDict  = self.imageTesorDict1[image_name]
                except Exception as e:
                    imageGlyphDict = self.imageTesorDict2[image_name]
                
                            
                if self.args.charImages == 1:
                
                    image_name2 = image_name.split(".png")[0]+"_"
                                        
                    temp_img_list = [self.dummyTensor.clone() for _ in range(MAX_CHARS)]
                
                    for l in range(len(label)):
                        
                        image_name3= image_name2+str(l)+"_"+".png"
                        #imageDict  = self.imageTesorDict2[image_name3]
                        # 
                        if image_name3 in self.imageTesorkeys2:   
                            #print("\n\t Found:",image_name3)
                            self.found+=1
                            
                            try:
                                imgTempDict = self.imageTesorDict2[image_name3]
                            except Exception as e:
                                imgTempDict = self.imageTesorDict2[image_name]
                                                        
                            #print("\n\t keys:",imgTemp.keys())
                            
                            #print("\n\t imgTemp.shape:",imgTemp.shape)
                            #temp_img_list.append(imgTemp)
                            
                            imgTemp = imgTempDict['images']
                            temp_img_list[l] = imgTemp
                            
                            """
                            if tempImg== None:
                                tempImg = imageGlyphDict['images']
                            else:
                                tempImg+ = imageGlyphDict['images']
                                #torch.cat(temp_img_list, dim=2)
                            """ 
                        else:
                            self.miss+=1
                            #print("\n\t Miss:",image_name3," \t miss:")
                    
                        tempImg = torch.cat(temp_img_list, dim=0)
                        #print("\n\t concatenated :",tempImg.shape," len:",len(tempImg))
                    
            except Exception as e:
                # else check in char dict
                imageGlyphDict  = self.imageTesorDict2[image_name]
                
            
            #print("\n\t self.found=",self.found,"\t self.miss=",self.miss)
            
            #imageGlyph = imageGlyphDict["imageGlyph"]            
            #imageGlyph = imageGlyph.squeeze()
            image = imageGlyphDict["images"]
            image = image.squeeze()

        
        
        word_embedding = label_padding(label, num_tokens) 
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()    
        
        if self.args.wrdChrWrStyl ==1:
            wrdChrWrStyl = torch.from_numpy(self.cropStyleDict[image_name])
            wrdChrWrStyl = wrdChrWrStyl.squeeze()
            
            return image_name,"None_tempImg","None_temp_img_list",image,wrdChrWrStyl, word_embedding, wr_id,label,phoscLabel
        else:
            
            if self.args.charImages == 1:
                tempImg = tempImg.squeeze(0)

                return image_name,tempImg,temp_img_list,image,"None_wrdChrWrStyl", word_embedding, wr_id,label,phoscLabel
            else:
                return image_name,"None_tempImg","None_temp_img_list",image,"None_wrdChrWrStyl", word_embedding, wr_id,label,phoscLabel
                

class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class Diffusion:
    def __init__(self, noise_steps=600, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    #predicted_noise = model(x,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
    
    #ema_sampled_images = diffusion.sampling(model, vae, wrdChrWrStyl,phoscLabels,n=n, x_text=x_text, labels=labels, args=args)
    #   sampling(      model, vae, x_text, words,n, labels, args)
    def sampling(self, model, vae,latents,x_text,words,n,labels, args):
        #print("\n\t sampling!!!")
        model.eval()
        tensor_list = []
        #if mix_rate is not None:
         #   print('mix rate', mix_rate)
        with torch.no_grad():
            
            words = [x_text]*n
            
            print("\n\t words =",words)
            for word in words:
                transcript = label_padding(word, num_tokens) #self.transform_text(transcript)
                word_embedding = np.array(transcript, dtype="int64")
                word_embedding = torch.from_numpy(word_embedding).long()#float()
                tensor_list.append(word_embedding)
            text_features = torch.stack(tensor_list)
            text_features = text_features.to(args.device)
            
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                
                #predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)

                s_id = torch.ones(text_features.shape[0], dtype=torch.int).to(self.device)
                
                if args.wrdChrWrStyl ==0:
                    wrdChrWrStyl = None
                    #print("\n\t 1 sampling")
                    #print("\n\t 00.text_features.device:",text_features.device)

                    if args.attentionMaps==1:
                        predicted_noise,attn1,attn2,attn3 = model(x,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                    elif args.attentionMaps==0:
                        predicted_noise = model(x,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                        
                    #print("\n\t 1.predicted_noise =",predicted_noise.shape,attn1.shape,attn2.shape,attn3.shape)
                                        
                elif args.wrdChrWrStyl ==1:
                    #print("\n\t 2")

                    predicted_noise = model(x,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                    #print("\n\t 1.predicted_noise =",predicted_noise.shape)

                if 0:#cfg_scale > 0:
                    # uncond_predicted_noise = model(x, t, text_features, sid)
                    # predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    #uncond_predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                    #predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    pass
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
    
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x

def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, num_classes, vocab_size, transforms, args):
    model.train()
    
    print('Training started....')
    # noise = transform1(noise)
    if args.augMaps == 1:
        transforms1 = torchvision.transforms.Compose([torchvision.transforms.RandomRotation(degrees=(-3, 3)),])
    
    
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        #pbar = tqdm(loader)
        
        
        if 0:

            labels = torch.arange(16).long().to(args.device)
            n=len(labels)
                    
            #words = ['text', 'getting', 'prop']

            #words = ['t','e','x','t', 'g','e','t','i','n','g', 'p','r','o','A',"n","i","k","e","t","M","a","y","u","A","a","y","U",]
            words = list(string.ascii_uppercase) + list(string.ascii_lowercase)

            #words = [words[i] + words[i + 1] for i in range(0, len(words) - 1, 2)]
            words = [words[i]  for i in range(0, len(words) - 1)]

            #sampling4(epoch,x_t,words, model,vae, n, x_text, labels, args)
            
            for x_text in words: 
                ema_sampled_images = diffusion.sampling(model, vae,latents, x_text, words,n, labels, args)
                
                print("\n\t ema_sampled_images.shape:",ema_sampled_images.shape)
                
                sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{epoch}.jpg"), args)
                        
        
        if 1:
            for i, (image_names,tempImg,temp_img_list,images,wrdChrWrStyl, word, s_id,label,phoscLabels) in enumerate(loader):
                
                #tempImg = tempImg.squeeze(1)
                #print("\n\t image_names:",image_names," tempImg.shape:",tempImg.shape," len temp_img_list:",len(temp_img_list))
                
                try:
                    with open(args.stopFlag,"r") as f:
                        stopValue = int(f.readline())
                    
                except Exception as e:
                    print("\n\t stop flag issue:",e)
    
    
                if stopValue == 0:
                    exit()
    
                images = images.to(args.device)
                original_images = images
                text_features = word.to(args.device)
                #print("\n\t i:",i," \t images.shape:",images.shape)
                
                """
                print("\n\t i:",i," \t images.shape:",images.shape)
                print("\n\t wordLabel:",label)
                print("\n\t word:",word.shape)

                #print("\n\t label:",label)
                
                #print("\n\t phoscLabels:",phoscLabels)
                print("\n\t phoscLabels:",phoscLabels.shape)
                
                input("check here")
                """
                
                #print("\n\t images.shape:",images.shape,"\t word:",word.shape,"\t wrdChrWrStyl.shape:",wrdChrWrStyl.shape," \t i:",i)
        
                s_id = s_id.to(args.device)
                
                if args.wrdChrWrStyl ==1:
                    wrdChrWrStyl = wrdChrWrStyl.to(args.device)

                if args.latent == True and args.vaeFromDict !=1: # 
                    images = vae.encode(images.to(torch.float32)).latent_dist.sample()
                    images = images * 0.18215
                    latents = images
                    
                if args.vaeFromDict ==1:
                    latents = images 
                if args.augMaps == 1:
                    images = transforms1(images)
                
                t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
                x_t, noise = diffusion.noise_images(images, t)
                
                if np.random.random() < 0.1:
                    labels = None
                
                
                if args.phosc ==1 or args.phos ==1:
                    predicted_noise = model(x_t, phoscLabels,timesteps=t,context=text_features, y=s_id)        
                else:                
                    if args.imgConditioned == 0:
                        
                        if args.charImages == 0 and args.ocrTraining == 0:

                            #print("\n\t ---device:", next(model.parameters()).device)
                                        
                            #try:

                            if args.attentionMaps == 1 and args.ocrTraining == 0:
                                
                                predicted_noise,att1,att2,att3 = model(x_t,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                            
                            elif args.attentionMaps == 0 and args.ocrTraining == 0:
                                predicted_noise = model(x_t,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                        
                            #except Exception as e:
                            #print("\n \t exception::",e)
                            #predicted_noise = model(x_t,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                                
                        elif args.charImages == 0 and args.ocrTraining == 1:
                            predicted_noise,att1,att2,att3,ocrPredVec = model(x_t,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                            #print("\n\t ocrPredVec.shape =",ocrPredVec.shape," ocrPredVec.size(0):",ocrPredVec.size(0))

                        elif args.charImages == 1 and args.ocrTraining == 0:
                            tempImg = tempImg.to(args.device)
                            predicted_noise,att1,att2,att3 = model(x_t,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id,charContextImages=tempImg)
                        elif args.charImages == 0 and args.ocrTraining == 1:
                            predicted_noise,att1,att2,att3,ocrPredVec = model(x_t,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                            #print("\n\t ocrPredVec.shape =",ocrPredVec.shape," ocrPredVec.size(0):",ocrPredVec.size(0))


                    elif args.imgConditioned == 1:
                        predicted_noise,att1,att2,att3 = model(x_t,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id,latents = latents)
                    
                    if args.ocrTraining ==1:
                        act_lens = torch.IntTensor(ocrPredVec.shape[1] * [ocrPredVec.size(0)])

                        label_lens = torch.IntTensor([len(t) for t in word])
                        
                        #labelCTC = word
                        
                        #print("\n\t word:",word)
                        print("\n\t word.shape:",word.shape," label_lens =",label_lens.shape)
                        
                        labelCTC = word.flatten()
                        """
                        try:
                            labelCTC = torch.IntTensor([c for c in ''.join(word)])
                        except Exception as e:
                            labelCTC = [c for c in ''.join(word)]
                        """ 

                        #print("\n\t labelCTC.shape =",labelCTC.shape," act_lens =",act_lens.shape," ocrPredVec.shape:",ocrPredVec.shape)

                        ctcLossVal = ctc_loss(ocrPredVec.cpu(), labelCTC, act_lens, label_lens)
                        
                            
                    #forward               ( x, original_images=None, timesteps=None, context=None, y=None, original_context=None, or_images=None, mix_rate=None, **kwargs)
                #print("\n\t predicted_noise.shape:",predicted_noise.shape)
                #predicted_noise = model(x_t, original_images=original_images, timesteps=t, context=text_features, y=s_id, or_images=None)
                        
                #input("check!!!")
                #continue
                
                
                #noise = transforms1(noise)
                loss = mse_loss(noise, predicted_noise)
            
                #                     print("\n\t ctcLossVal =",ctcLossVal.item())            
                
                try:           
                    print("\n\t epoch:",epoch,"\t batch:",i,"t loss:",loss.item(),"\t ctcLossVal =",ctcLossVal.item())
                except Exception as e:
                    print("\n\t epoch:",epoch,"\t batch:",i,"t loss:",loss.item())
                
                if args.ocrTraining ==1:
                    loss += ctcLossVal        
                
                """
                try:
                    print("\n\t att1.shape:",att1.shape," att2.shape:",att2.shape," att3.shape:",att3.shape," latents.shape:",latents.shape)
                except Exception as e:
                    pass
                """        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model)
                #pbar.set_postfix(MSE=loss.item())
                
        if epoch % 5 == 0:
                    
                print("\n\t save path:",os.path.join(args.save_path,"models", "ckpt_"+args.saveModelName))
                
                try:
                    torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt_"+args.saveModelName))
                    torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_"+args.saveModelName))
                    #torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim_"+args.saveModelName))   
                except Exception as e:
                    torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
                    torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema.pt"))
                    #torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))   
                
            

import pickle

def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='iam', help='iam or other dataset') 
    
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4, help='if latent is True channels should be 4, else 3')  
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    #parser.add_argument('--save_path', type=str, default='./save_path/')
    parser.add_argument('--device', type=str, default=device) 
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json') #
    parser.add_argument('--stable_dif_path', type=str, default="/cluster/datastore/aniketag/allData/supportingSoftwares/stableDiffusion/", help='path to stable diffusion')
    parser.add_argument('--iam_path', type=str, default='/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/', help='path to iam dataset (images 64x256)')

    # experiment wise changing parameter
    
    parser.add_argument('--gt_train', type=str, default=gt_train) #  

    
    parser.add_argument('--csvRead', type=str, 
                        default=csvRead, 
                        help='training info from .csv instead of authors file') 
    
    parser.add_argument('--loadPrev', type=int, default=0,help ="model from authorBasePath gets loaded")


    parser.add_argument('--save_path', type=str, default=save_path,help = "this is location where it savesthe new model" ) 
    #parser.add_argument('--saveModelName', type=str, default= saveModelName ,help = "by this name save model at save_path" ) 
    parser.add_argument('--saveModelName', type=str, default= saveModelName,help = "by this name save model at save_path" ) 

    #ema_ema_charImageNoWriter_2_qkvChange.pt
    
    parser.add_argument('--trascriptionPlusOCR', type=int, default=0,help = "it joins transcription and OCR prediction as a conditional input")

    parser.add_argument('--phosc', type=int, default=1)
    parser.add_argument('--phos', type=int, default=0)
    parser.add_argument('--authorBasePath', type=str, default= authorBasePath,help = "This is old model path") # './wordStyleOutPut_600_preprocess_0/'
    #parser.add_argument('--lang', type=str, default= ["eng","nor"][0],help = "language") 

    parser.add_argument('--stopFlag', type=str, default = "./flags/stopFlag.txt",help ="flag to stop program") # partialLoad
    parser.add_argument('--partialLoad',  type=int, default=0)

    parser.add_argument('--imgConditioned', type=int, default=0,help = "entire original image passed through preprocessing part and those embedding added with text embeddings")
    parser.add_argument('--vaeFromDict', type=int, default=1)
    parser.add_argument('--wrdChrWrStyl', type=int, default=0)
    parser.add_argument('--charImages', type=int, default=0)
    parser.add_argument('--augMaps', type=int, default=0,help = "This augments the feature map ath the training time")
    parser.add_argument('--attentionMaps', type=int, default=0,help= "return attention maps")
    parser.add_argument('--attentionVisualition', type=int, default=0,help= "visualise attention maps")
    #parser.add_argument('--noWriter', type=int, default=0,help= "visualise attention maps")
    parser.add_argument('--ocrTraining', type=int, default=0) 
    parser.add_argument('--erase', type=int, default=0,help = "draw verticle lines which erases input image ") 
    parser.add_argument('--charLevelEmb', type=int, default=0,help = "the word level embeddings are calculated by concatenating char level embeddings")
    parser.add_argument('--lang', type=str, default= lang,help = "language") 

    args = parser.parse_args()
    
    print("\n Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")    

    print("\n")

    
    assert args.phosc != 1 or MAX_CHARS == 10, "MAX_CHARS should be 10 when args.phosc is 1"
    assert args.phos != 1 or MAX_CHARS == 10, "MAX_CHARS should be 10 when args.phos is 1"

    assert not (args.phosc == 1 and args.trascriptionPlusOCR == 1), "both can not be 1 at same time"
    assert not (args.phosc == 1 and args.phos == 1), "both can not be 1 at same time"

    assert args.trascriptionPlusOCR != 1 or MAX_CHARS == 42, "MAX_CHARS should be 42 when args.trascriptionPlusOCR is 1"
    assert args.trascriptionPlusOCR != 1 or MAX_CHARS == 42, "MAX_CHARS should be 42 when args.trascriptionPlusOCR is 1"

    assert not (args.phosc == 1 and args.trascriptionPlusOCR == 1), "both can not be 1 at same time"
    assert not (args.phos == 1 and args.trascriptionPlusOCR == 1), "both can not be 1 at same time"



    if args.wandb_log==True:
        runs = wandb.init(project='DIFFUSION_IAM', name=f'{args.save_path}', config=args)

        wandb.config.update(args)
    
    #create save directories
    setup_logging(args)

    print('character vocabulary size', vocab_size)
    
    if args.dataset == 'iam':
        class_dict = {}
        for i, j in enumerate(os.listdir(f'{args.iam_path}')):
            class_dict[j] = i

        transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

        if args.csvRead:
            df = pd.read_csv(args.csvRead)
            imgValues = set(df.imageName)
            
            with open("./gt/image_wr_dict.pkl", "rb") as pickle_file:
                imgWriteDict = pickle.load(pickle_file)


            keys = set(imgWriteDict.keys())

            imgValues = imgValues.intersection(keys)
        
        with open(args.gt_train, 'r') as f:
            train_data = f.readlines()
            #print("\n\t train_data:",train_data)
            
            if not args.csvRead:
                train_data = [i.strip().split(' ') for i in train_data]
                
            #train_data = train_data[:10]

                
            wr_dict = {}
            full_dict = {}
            image_wr_dict = {}
            img_word_dict = {}
            wr_index = 0
            idx = 0
            
            if args.partialLoad:
                breakIndex = int(len(train_data) * args.partialLoad)
                #logger.info("\n\t working on parialLoad mode!!!")

            # /cluster/datastore/aniketag/allEnv/newDiffusion/bin/python
            # /cluster/datastore/aniketag/newWordStylist/WordStylist
            for rowNo,i in enumerate(train_data):
                """
                if rowNo==100:
                    break
                """
                if args.partialLoad:
                    if rowNo == breakIndex:
                        break

                
                #print("\n\t i:",i)
                
                try:
                    s_id = i[0].split(',')[0]
                    image = i[0].split(',')[1] + '.png'
                    transcription = i[1]
                except Exception as e:
                    
                    """
                        this part will be active only when input is .csv 
                    """
                    
                    if rowNo>=df.shape[0]:
                        break
                    
                    actualText = df.loc[rowNo,"Actual"]
                    image = df.loc[rowNo,"imageName"]

                    
                    if args.trascriptionPlusOCR:
                        
                        transcription = df.loc[rowNo,"Predicted_All"]

                        if len(transcription)<32:
                            transcription = transcription.ljust(32)

                        if isinstance(actualText, float):
                            actualText = "None"

                        if len(actualText)<10:
                            actualText = actualText.ljust(10)

                        
                        try:
                            transcription = actualText+transcription
                        except Exception as e:
                            transcription = actualText+transcription
                        
                        #print("\n\t actualText df =",actualText)
                        #print("\n\t transcription df =",transcription)
                        #print("\n\t totallength:",len(transcription))
                    
                    else:

                        if isinstance(actualText, float):
                            actualText = "None"

                        if len(actualText)<10:
                            actualText = actualText.ljust(10)

                        transcription = actualText
                                
                    #print("\n\t len transcription =",len(transcription))
                
                    s_id = imgWriteDict[image]

                #print("\n\t sid dict:",s_id)
                
                #print(s_id)
                full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
                
                #print("\n\t full_dict[idx] =",full_dict[idx])

                
                #input("check11")

                
                image_wr_dict[image] = s_id
                img_word_dict[image] = transcription
                idx += 1
                if s_id not in wr_dict.keys():
                    wr_dict[s_id] = wr_index
                    wr_index += 1
        
            print('number of train writer styles', len(wr_dict))
            style_classes=len(wr_dict)
        
        # create json object from dictionary if you want to save writer ids
        json_dict = json.dumps(wr_dict)
        f = open("writers_dict_train.json","w")
        f.write(json_dict)
        f.close()
        
        train_ds = IAMDataset(full_dict, args.iam_path, wr_dict, args, transforms=transforms)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        print("\n\t train_loader length:",len(train_loader))
    #unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device)    
    
    
        
    if args.phosc == 1 or args.phos == 1:
        print("\n\t phosc")
        unet = UNetModelPhosc(image_size = args.img_size, in_channels=args.channels,
                        model_channels=args.emb_dim, out_channels=args.channels,
                        num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), 
                        channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes,
                        context_dim=args.emb_dim, vocab_size=vocab_size, 
                        args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device) 
        
        
    else:
        #print("\n\t original")
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels,
                        model_channels=args.emb_dim, out_channels=args.channels,
                        num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), 
                        channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes,
                        context_dim=args.emb_dim, vocab_size=vocab_size, 
                        args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device)    

    print("\n\t trying to load models!!!")

    #modelPath = "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/ema_charLevelEmb_1200.pt"
    #unet.load_state_dict(torch.load(modelPath,map_location=device),strict=False)
    
    print("\n\t is model:",os.path.isfile(args.save_path+saveModelName))
    
    if args.loadPrev == 1 and os.path.isfile(args.save_path+saveModelName):
        
        unet.load_state_dict(torch.load(args.save_path+saveModelName,map_location=device),strict=False)
        print("\n\t unet model loaded from:",args.save_path+saveModelName," on:")


        device_unet = next(unet.parameters()).device
        print("\t unet model is currently on device:", device_unet)

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)

    if 0:#args.loadPrev == 1 and os.path.isfile(args.authorBasePath+"optim.pt"):
        optimizer = optimizer.load_state_dict(torch.load(args.authorBasePath+"optim.pt",map_location=device))
        print("\n\t optimizer loaded from ",args.authorBasePath+"optim.pt")
    
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    #ema_model.load_state_dict(torch.load(modelPath,map_location=device),strict=False)
    
    if 0:#args.loadPrev == 1 and os.path.isfile(args.save_path+saveModelName):
        ema_model.load_state_dict(torch.load(args.save_path+saveModelName,map_location=device),strict=False)
        print("\n\t ema model loaded from ",args.save_path+saveModelName)
    else:
        ema_model = unet
    
    if args.latent==True:
        print('Latent is true - Working on latent space')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)
        
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        print('Latent is false - Working on pixel space')
        vae = None
    
    train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, style_classes, vocab_size, transforms, args)


if __name__ == "__main__":
    main()
  
  
