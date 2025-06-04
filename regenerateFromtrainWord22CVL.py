import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image,ImageOps
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import json
from diffusers import AutoencoderKL

from unetOriginal import UNetModel
from unetPhosc import UNetModelPhosc
from  ResPhoSCNetZSL.modules.datasets import phosc_dataset


import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from utils.tensorProcess import *
import random 
import pandas as pd
import shutil
from utils.dataGenerationConfigICPR_CVL import *
import random
# 
#os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
os.environ['PYTHONUNBUFFERED'] = '1'


"""
    styleClasses in createDataLoader 
    check authorBasePath,phosc,phosc,allInOneIndx (No of characters)
    gtPath,savePath, sidChange,fullSampling
"""

import logging
import inspect



logging.basicConfig(
    format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('//cluster/datastore/aniketag/newWordStylist/WordStylist/logs/cvlGenerationLogs.log'),  # Add a FileHandler
        logging.StreamHandler()  # Add a StreamHandler for console output
    ]
)
logger = logging.getLogger('cvlGeneration')
logger.info('--- cvlGeneration ---')

def log_with_line_number(message):
    frame = inspect.currentframe().f_back
    line_number = frame.f_lineno
    logging.info(f"Line {line_number}: {message}")



#MAX_CHARS = 10
OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>

#c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'

if lang == "ENG" and lang == "GerEng":
    
    """
    #c_classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#&'()*+,-./0123456789:;?_"
    c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz=!#&\'()*+,-./0123456789:;? "'
    # c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüæøå=!#&\'()*+,-./0123456789:;?_"'

    c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüæøå=!#&\'()*+,-./0123456789:;?_"' # phos model

    #c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
    #    c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
    #c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    #c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # only for PHOSC model trained on college gpu
    
    # Mse_text_Phos_condi_FromScratch
    c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'

    # 4head_1res_IamOfficialSplit_textPhos_WordCharJoint_ema_1.pt
    #c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüæøå=!#&\'()*+,-./0123456789:;?_"'

    # wordstylist
    #c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    """
    c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüæøå=!#&\'()*+,-./0123456789:;?_"'

    

elif lang == "NOR":
    c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅabcdefghijklmnopqrstuvwxyzæøå'
    classes = '_' + ''.join(c_classes)

#classes = "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
#classes = ' '+ ''.join(classes)

c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüæøå=!#&\'()*+,-./0123456789:;?_"'

ocr_classes = ' _ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


cdict = {c:i for i,c in enumerate(c_classes)}
icdict = {i:c for i,c in enumerate(c_classes)}

cdict_ocr = {c:i for i,c in enumerate(ocr_classes)}
icdict_ocr = {i:c for i,c in enumerate(ocr_classes)}



def print_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
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
    
    print("\n\t c_classes in labelDictionary:",c_classes)
    
    
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
print('num_tokens:', num_tokens)


print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens


def save_images(dumpPath,images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im

def createDataLoader(args,transforms):
    
    allowedCrops = dict()
    
    print("\n\t scanning folder for images:",dumpBasePath+split+batchFolder)
    
    
    os.makedirs(dumpBasePath+split+batchFolder,exist_ok=True) 
     
    if 1:#os.path.isdir(dumpBasePath+split+batchFolder):
        # "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//WordStylistOovAllWritersTrainSet5/"
        #genAcceptImageList = os.listdir("//cluster/datastore/aniketag/allData/syntheticData/train/icpr//WordStylistOovAllWritersTrainSet3/") #os.listdir(dumpBasePath+split+batchFolder)
        # //cluster/datastore/aniketag/allData/syntheticData/train/icpr//icprRebuital/
        #genAcceptImageList = os.listdir("//cluster/datastore/aniketag/allData/syntheticData/train/icpr//randomExp//") #os.listdir(dumpBasePath+split+batchFolder)

        genAcceptImageList = os.listdir("///cluster/datastore/aniketag/allData/syntheticData/train/icpr//icprRebuitalCVL1//") #os.listdir(dumpBasePath+split+batchFolder)

        
        for nmImage in genAcceptImageList:
            
            nmImage = nmImage.split(".png")[0]#+'jpg'#+'.png'
            
            nmImage = nmImage.split("_")[:3]
            nmImage = "_".join(nmImage)
            
            #nmImage = nmImage.join(nmImage.split('_')[:-2])
            
            
            #print("\n\t nmImage =",nmImage)
            
            allowedCrops[nmImage] = 1 
            #exit()
        
    print("\n\t No of images already generated :",len(allowedCrops))
    #print("\n\t No of images already generated :",list(allowedCrops)[:5])


    with open(args.gtPath, 'r') as f:
        train_data = f.readlines()
        train_data = [i.strip().split(' ') for i in train_data]
        wr_dict = {}
        full_dict = {}
        image_wr_dict = {}
        img_word_dict = {}
        wr_index = 0
        idx = 0

        if args.partialLoad:
            breakIndex = int(len(train_data) * args.partialLoad)

        skippedImages = 0
        
        for indx,i in enumerate(train_data):
            
            if args.partialLoad:
                if indx == breakIndex:
                    break
                
            s_id = i[0].split(',')[0]
            image = i[0].split(',')[1] #+ '.jpg'
            
            """
            if indx<5:
                print("\n\t image =",image)
            """
            #print("\n\t 1.:"," len:",len(allowedCrops.keys()))
            
            if  image in allowedCrops.keys() and len(allowedCrops.keys()):
                skippedImages = skippedImages+1
                continue
            
            transcription = i[1]
            
            """
                        
            if len(transcription) <=9:
                transcription =transcription + "z"
                 
            transcription = list(transcription)
            transcription[0] = "c"
            transcription[-1] = "m"

            transcription = "".join(transcription)
            """

            #print(s_id)
            full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
            image_wr_dict[image] = s_id
            img_word_dict[image] = transcription
            idx += 1
            if s_id not in wr_dict.keys():
                wr_dict[s_id] = wr_index
                wr_index += 1
            
            
        #print('number of train writer styles', len(wr_dict)," \t indx:",indx)
        
        if args.lang == "ENG":
            style_classes= 339#len(wr_dict)
        else:
            style_classes= 48
    #print("\n\t i:",i," length:",len(wr_dict)," \t indx:",indx," \t leN:",len(full_dict))
    # create json object from dictionary if you want to save writer ids
    
    json_dict = json.dumps(wr_dict)
    f = open("writers_dict_train.json","w")
    f.write(json_dict)
    f.close()
    
    train_ds = IAMDataset(full_dict, args.iam_path, wr_dict, args, transforms=transforms)
    
    print("\n\t train_ds =",len(train_ds)," \t skippedImages:",skippedImages," \n\t len full_dict:",len(full_dict))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    return train_loader,style_classes,wr_dict,full_dict,image_wr_dict,img_word_dict

# /cluster/datastore/aniketag/WordStylist

def callOCR(net,image):

    decodeOutput = []
    
    
    #print("\n\t image.device:",image.device," net device:",net.parameters().__next__().device," image.shape:",image.shape)
    with torch.no_grad():
        o = net(image[:,0,:,:].unsqueeze(1).to(image.device))

        if head_type == 'both':
            o = o[0]

        #print("\n\t o:",o.shape," image.shape:",image.shape)
    
        tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        # icdict_ocr
    for indx,tdec1 in enumerate(tdec):
        tt = [v for j, v in enumerate(tdec1) if j == 0 or v != tdec1[j - 1]]
        #print("\n\t tdec =:",tt)
        dec_transcr = ''.join([icdict_ocr[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()
        #print("\n\t dec_transcr:",dec_transcr,"\t actual trans:",wordLabel[indx])
        decodeOutput.append(dec_transcr)
        
    return o,decodeOutput


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
        
        if self.args.phos ==1 or self.args.phosc ==1:
        
            phoscClass = phosc_dataset(self.args,self.data_dict)
            #phoscClass = phosc_dataset.getPhosc(self.data_dict)

            if 1:#not os.path.isfile("./wordPhos.pkl"):
                self.wordPhosc = phoscClass.getPhosc()
                
                with open("./wordPhos1.pkl", 'wb') as file:
                    # Use pickle.dump() to write the dictionary to the file
                    pickle.dump(self.wordPhosc, file)            

                    print("\n\t new wordPhosc created")
                    
            else:
                with open("./wordPhos.pkl", 'rb') as file:
                    # Use pickle.load() to load the dictionary from the file
                    self.wordPhosc = pickle.load(file)    
                    print("\n\t old wordPhosc read")


            print("\n\t total in phosc/phoc dir is:",len(self.wordPhosc.keys()))
            

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

                
        #     parser.add_argument('--imgConditioned', type=int, default=1)
        
        word_embedding = label_padding(label, num_tokens) 
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()   
    
        #print("\n\t word_embedding =",word_embedding.shape)
        
        #print("\n\t idx:",idx," \t image_name:",image_name) 
        
        #return  image_name,word_embedding, wr_id,label,phoscLabel
        
        if self.args.imgConditioned == 1:
            img_path = os.path.join(self.image_path, image_name)
            
            image = Image.open(img_path).convert('RGB')
            image = self.transforms(image)
            return image,image_name,word_embedding,wr_id,label,phoscLabel
        
        else:
            return image_name,word_embedding,wr_id,label,phoscLabel

            

class Diffusion:
    def __init__(self, noise_steps=600, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), args=None):
        self.noise_steps = 600 #noise_steps
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
        #return torch.randint(low=299, high=300, size=(n,))

             

    def sampling3(self,epoch,x_t,words,phoscLabels, model,model1, vae,emaOld,noiseInput, n, x_text, labels, args, mix_rate=None, cfg_scale=3):
        
        modelCall = 0
        
        #print("\n\t 1.words:",words)
        if emaOld==1:
            model = model1
        
        noise_dict = {}#collections.defaultdict(list)
        model.eval()
        tensor_list = []
        all_noises = []
        allX = []  # predicted images
        allT = []  # original        
        
        allImages = []
        #if mix_rate is not None:
         #   print('mix rate', mix_rate)
         
         
        # Load the noise dictionary if it exists
        try:
            noise_dict = {}
            #noise_dict = torch.load('noise_dict.pt')
        except FileNotFoundError:
            noise_dict = {}
         
        
        with torch.no_grad():
            
            if len(x_text)>1:
                x_text = list(x_text)
            else:
                words = [x_text]*n

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
            
            if noiseInput ==0:
                x = x_t #+ torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)

            #t = (torch.ones(n) * i).long().to(self.device)

            #predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
            
            for i in reversed(range(1, self.noise_steps)):
                

                with open("./flagGen.txt","r") as f:
                    flag = int(f.read())
                
                
                if flag == 0:
                    epoch = args.epochs+1
                    i = 0
                    break                
            
                t = (torch.ones(n) * i).long().to(self.device)
                
                #print("\n\t i:",i," flag =",flag)

                if args.fullSampling or ((i%(100)  ==0 or i%5==0 or i==self.noise_steps or i==(self.noise_steps-1) or (epoch>3 and i%(25) ==0) or (epoch>5 and i%(15) ==0) or (epoch>10 and i%(10) ==0) or epoch>50==0)):
                    
                    
                    """
                    if args.phosc ==1:
                        print("##############")
                        predicted_noise = model(x, phoscLabels,None, t, text_features, labels, mix_rate=mix_rate)
                    else:
                        predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                    """
                    
                    if args.phosc ==1 or args.phos ==1:
                        
                        #print(x.shape)#," phoscLabels.shape:",phoscLabels.shape," text_features.shape:",text_features.shape," \t labels:",labels.shape)
                        
                        
                        #print("\n\t before text_features:",text_features)
                        #print("\n\t text_features:",text_features)
                        
                        
                        
                        # Create a mask where elements are True if they are not equal to 52, and False if they are equal to 52
                        mask = text_features != 52

                        # Count the number of 52 in each row
                        count_52 = (~mask).sum(dim=1)

                        # Expand the mask to have the same shape as text_features
                        mask = mask.unsqueeze(2).repeat(1, 1, 2)

                        # Create a new tensor by combining the original tensor and a tensor of 52s using the mask
                        #text_features = torch.cat([text_features[mask], torch.full_like(text_features, 52)], dim=2)

                        # Adjust the count of 52 to match the original count in each r"""ow
                        
                        """
                        for i in range(result.size(0)):
                            num_to_insert = count_52[i]
                            text_features[i, -num_to_insert:] = 52
                        
                        
                        """
                        # Print the size of 'text_features' before and 'result' after transformation
                        #print("Size Before:", text_features.size())
                        #print("Size After:", result.size())
                        
                        #print("\n\t After text_features:",text_features)
                        
                        
                        #input("check!!!")
                        modelCall+=1
                        predicted_noise = model(x, phoscLabels,timesteps=t,context=text_features, y=labels)        
                    else:                
                        #predicted_noise = model(x,None,timesteps=t,context=text_features,y=labels)                    
                    
                        predicted_noise = model(x, None, t, text_features, labels)

                    
                else:
                    pass
                #allT.append(predicted_noise)
                
                all_noises.append(predicted_noise.detach().cpu())  # Append the noise tensor to the list
                
                if 0:#cfg_scale > 0:
                    pass
                    # uncond_predicted_noise = model(x, t, text_features, sid)
                    # predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    uncond_predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                if args.fullSampling:
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise            
                else:
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) #+ torch.sqrt(beta) * (noise/10)        
                    
                
                #x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            #model.train()
            if args.latent==True:
                latents = 1 / 0.18215 * x
                image = vae.decode(latents).sample

                image = (image / 2 + 0.5).clamp(0, 1)
                            
                allT.append(image)
                
                image = image.cpu().permute(0, 2, 3, 1).numpy()
        
                image = torch.from_numpy(image)
                #x = image.permute(0, 3, 1, 2)
                allX.append(image.permute(0, 3, 1, 2))

            else:
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)

        #print("\n\t modelCall:",modelCall)


        allT = torch.stack(allT)
        allT = allT.squeeze(0)

        return 0,allX,allT


def makeDir(dumpPath):
    if not os.path.isdir(dumpPath):
        os.mkdir(dumpPath)



"""

source /cluster/datastore/aniketag//allEnv/newDiffusion/bin/activate
cd /cluster/datastore/aniketag/newWordStylist/WordStylist/
python3 regenerateFromtrain2.py &>./logs/OovAllWritersTrainSet4.txt

"""



def train(diffusion, model,net, ema, ema_model,ema_model1, vae, optimizer, mse_loss, loader, num_classes, vocab_size, transforms, args):
    model.eval()
    
    print('Data generation started....')
    
    tick = time.time()

    allAcceptedImages = dict()

    #dumpPath = "/cluster/datastore/aniketag/allData/syntheticData/train/icpr//ocrGen//"
    
    #dumpPath = "/cluster/datastore/aniketag/allData/syntheticData/train/icpr//exp//sameWriterRepalce2//"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//OovAllWritersTrainSet1/"
    
    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//OovAllWritersTrainSet2/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//OovAllWritersTrainSet3/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//OovAllWritersTrainSet5/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//WordStylistOovAllWritersTrainSet1/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//WordStylistOovAllWritersTrainSet2/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//WordStylistOovAllWritersTrainSet3/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//WordStylistOovAllWritersTrainSet4/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//WordStylistOovAllWritersTrainSet5/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//CVLOovAllWritersTrainSet1/"

    #dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//randomExp/"

    dumpPath = "//cluster/datastore/aniketag/allData/syntheticData/train/icpr//icprRebuitalCVL1/"


    for epoch in range(args.epochs):
        
        print('\n\t Epoch:', epoch)

        writerDumpPath = dumpPath#+"//"+str(epoch)+"//"/cluster/datastore/aniketag/allData/syntheticData/train/icpr/wordStylistAuthorModel1
        if not os.path.isdir(writerDumpPath):
            os.mkdir(writerDumpPath)                                                            
            
            print("\n\t writerDumpPath:",writerDumpPath)
        
        with open("./flagGen.txt","r") as f:
            flag = int(f.read())
        
        
        if flag == 0:
            epoch = args.epochs+1
            break

                
        """
            add images to dataloader except epoch 0
        """

        if 0:#epoch ==-1:
            
            print("\n\t initial length:",len(loader)*args.batch_size)
            pbar = tqdm(loader)
        
        else:
            
            """
                create new dataloader
            """
            train_loader,style_classes,wr_dict,full_dict,image_wr_dict,img_word_dict= createDataLoader(args,transforms)
            
            print("\n\t train_loader len:",len(train_loader),flush=True)
            pbar = train_loader #tqdm(train_loader)
        
        
        for i, (image_name,word, s_id, wordLabel,phoscLabels) in enumerate(pbar):
            
            
            #print("\n\t epoch:",epoch," \t i:",i,"\t image_name:",image_name,"\t word:",word," \t s_id:",s_id,"\t wordLabel:",wordLabel,"\t phoscLabels:",phoscLabels)

            with open("./flagGen.txt","r") as f:
                flag = int(f.read())
            
            
            if flag == 0:
                epoch = args.epochs+1
                break

            print("\n\t i:",i," flag =",flag)

            
            text_features = word.to(args.device)
            
            #s_id1 = s_id+epoch
            #s_id1[s_id1 > 339] = s_id[s_id1 > 339]  

            if args.sidChange == 1:# and epoch >1:
                
                
                if args.lang == "ENG":
                    if random.random()<0.5:
                        s_id1 = s_id+random.randint(1,170)
                        s_id1[s_id1>338] = s_id[s_id1>338]

                    else:                
                        s_id1 = s_id-random.randint(1,170)
                        s_id1[s_id1<0] = s_id[s_id1<0]
                        
                else:
                    
                    if random.random()<0.5:
                        s_id1 = s_id+random.randint(1,23)
                        s_id1[s_id1>48] = s_id[s_id1>48]

                    else:                
                        s_id1 = s_id-random.randint(1,23)
                        s_id1[s_id1<0] = s_id[s_id1<0]                    
                    
                    #print("\n\t sid1:",s_id1)
                    
            else:
                s_id1 = s_id #* 0 + epoch +220
                        
            s_id = s_id1.to(args.device)
            
        
            t = diffusion.sample_timesteps(s_id.shape[0]).to(args.device)
            #print("\n\t t:",t )
            
            #print("\n\t t.shape:",t.shape,"\t x_t.shape:",x_t.shape,"\tnoise.shape:",noise.shape)
            
            
            if np.random.random() < 0:
                labels = None
            
            #predicted_noise = model(_, original_images=original_images, timesteps=t, context=text_features, y=s_id, or_images=None)
                        
    

            if args.ocr ==1:# and random.random()< 0.1 and args.partialCTC: 

                    
                #print("\n\t calculating and using CTC loss")
                #dumpPath = args.save_path+str(epoch) #+'//images//'
                #delAll(dumpPath) # clears previous images
                """
                if args.savedOcrImages:
                    print("\n\t making:",dumpPath)
                    makeDir(dumpPath)
                """
                #dumpPath = args.save_path+'//images//'+str(epoch)+"//"
                
                #print("\n\t dumpPath:",dumpPath)
                """
                if args.savedOcrImages:
                    makeDir(dumpPath)
                                
                """
                noiseInput = 1 
                emaOld = 0
                x_t = None
                #ema_sampled_images,allImages,allTensors = diffusion.sampling3(x_t,wordLabel, ema_model,ema_model1, vae,emaOld,noiseInput, n=len(s_id), x_text=wordLabel, labels=s_id, args=args)
                ema_sampled_images,allImages,allTensors = diffusion.sampling3(epoch,x_t,wordLabel,phoscLabels, ema_model,ema_model1, vae,emaOld,noiseInput, n=len(s_id), x_text=wordLabel, labels=s_id, args=args)

                
                if args.savedOcrImages:
                    
                    
                    #dumpPath = args.save_path+str(epoch)+"//" #+ wordLabel[tempIndx]
                    #dumpPath = args.save_path+str(0)+"//" 
                    #dumpPath = "/cluster/datastore/aniketag/allData/syntheticData/train/norwegian/batch2/"#savePath1+str(0)+"//"
                    
                    if not os.path.isdir(dumpPath):
                        os.mkdir(dumpPath)                                                    
                    
                    print("\n\t writing images at location:",writerDumpPath)
                    for imageNo,ema_sampled_images1 in enumerate(allImages):# allImages[-1:]
                        
                        """
                            take images into batch one by one
                        """
                        
                        gt = []
                        
                        for tempIndx,tempImage in enumerate(ema_sampled_images1):
                            
                            #writeImgName = f"{image_name[tempIndx]}_{wordLabel[tempIndx]}_{i}__{epoch}_{str(imageNo)}_{t[tempIndx]}.png" # last 0 is for non random noise based generation, noise is image generated from forward process

                            writerID = str(s_id[tempIndx].item())
                            txt = wordLabel[tempIndx]
                            imgNameWrite = image_name[tempIndx] 
                            imgNameWrite = imgNameWrite.split(".png")[0]
                            imgNameWrite = imgNameWrite+"_"+writerID 
                            
                            writeImgName = f"{imgNameWrite}_{txt}.png" # last 0 is for non random noise based generation, noise is image generated from forward process

                            
                            #print("\n\t writeImgName:",writeImgName," .shape",tempImage.shape," tempIndx:",tempIndx," writing:",os.path.join(dumpPath, writeImgName))
                            gt.append([dumpPath+writeImgName,wordLabel[tempIndx]])
                            
                            sampled_ema = save_images(writerDumpPath,tempImage, os.path.join(writerDumpPath, writeImgName), args)
                            
                
                continue
                #myDataset = IAMDataset1
                args.dataset_folder = dumpPath #os.path.join(wordStylistBase,"wordStyleOutPut_1000_preprocess_0_mini","images","20//")
                
                allTensors = torchProcess(allTensors)
                ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / allTensors.shape[0] #args.batch_size
                
                #dumpImages(allTensors,"./savedTensors/",str(1))
                #print("\n\t 1.allTensors:",allTensors[0].min().item()," \t max:",allTensors[0].max().item())

                fheight, fwidth = fixed_size[0], fixed_size[1]
                
                allTensors = tensor_centered(allTensors, (fheight, fwidth), centering=(.5, .5), border_value=0.0)

                #allTensors = allTensors.cpu().numpy()
                #allTensors = (allTensors * 255).astype(np.uint8) #[0]

                #dumpImages(allTensors,"./savedTensors/",str(1))

            
                output1,dec_transcr1 = callOCR(net,allTensors)
                
                output1.requires_grad = False
                
                #print("\n\t dec_transcr:",dec_transcr)
                
                """
                for w,d,d1 in zip(wordLabel,output1,dec_transcr1):
                    #print("\n\t original:",w,"\t d1:",d1," \t match:",d==d1)
                    print('orig:' + w,"\t d1:",d1, file=sys.stderr)
                """


                correctCount = 0
                totCount = 0
                delCount = 0
                
                delImageName = []
                
                #image_names.append(currImgName[0])
                
                """
                    1. copy generated images
                    2. Delete ocr failed images
                """       
               # print("\n\t copying and deleting images!!!")

                #copyTo = gt[totCount][0].split("0//")[1]

                
                #print("\n\t copying correct image to location:",dumpBasePath+split+batchFolder)
                
                if i %1 ==0:
                    for w,d1 in zip(wordLabel,dec_transcr1):
                        

                        if w==d1:
                            #print("\n\t gt value:",gt[totCount][0])
                            correctCount+=1
                            
                            #copyTo = gt[totCount][0].split("dataGenPipeline/0//")[1]
                            
                            copyTo =  os.path.basename(gt[totCount][0])
                            writeImgName = copyTo
                            allAcceptedImages[copyTo] = 1

                            tempImage = ema_sampled_images1[totCount]
                            sampled_ema = save_images(writerDumpPath,tempImage, os.path.join(writerDumpPath, writeImgName), args)

                            #os.copy(gt[totCount][0],"./dataGenPipeline/accepted/"+copyTo)
                            #shutil.copy("","") # "./dataGenPipeline/accepted/"
                            
                        else:
                            delCount+=1
                        """
                        else:
                            
                            writeImgName = f"{image_name[tempIndx]}_{wordLabel[tempIndx]}.png"
                            delImageName.append(gt[totCount][0])
                            os.remove(gt[totCount][0])
                            delCount+=1
                        """
                        totCount+=1
                    #os.remove(savePath1)

                    #shutil.rmtree(savePath1+"//0//")
                    #os.makedirs(savePath1+"//0//",exist_ok=True)
                    
                print("\n\t correctCount:",correctCount," \t totCount:",totCount,"\t accuracy:",(correctCount*1.0/totCount),"\t tot accepted:",len(os.listdir(writerDumpPath)),
                        " \t delCount:",delCount," tot images:",len(os.listdir(writerDumpPath)))
            
            
            #exit()   
            #input("check!!!")
            # Create a dictionary to store layer-wise weight averages
            weight_averages = {}
    
            
            #print("\n\t time taken by batch:",epoch,"\t : ",print_time(time.time()-tick)," \t batch number:",i," correctCount:",correctCount)   
            
        print("\n\t time taken by epoch:",epoch,"\t : ",print_time(time.time()-tick))   
        


import warnings
warnings.filterwarnings("ignore")

#from htr.utils import word_dataset,iam_dataset
#from htr.utils.iam_dataset import *
#from htr.utils import config
from htr.models import HTRNet
from htr.utils.config import head_type,cnn_cfg,head_cfg,flattening,stn,fixed_size

#from htr.htrInference import *
from  ResPhoSCNetZSL.modules.datasets import phosc_dataset
import pickle

"""
    check authorBasePath,phosc,phosc,allInOneIndx (No of characters)
    gtPath,savePath, sidChange,fullSampling,trascriptionPlusOCR
"""
def main():
    '''Main function'''
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--epochs', type=int, default=657)
    
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=32)# batchSize
    parser.add_argument('--num_workers', type=int, default=8) 
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    gtFilePath = [gt_train][0]


    parser.add_argument('--gtPath', type=str, default=gtPath) #'./gt/gan.iam.tr_va.gt.filter27'
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4, help='if latent is True channels should be 4, else 3')  
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    #parser.add_argument('--save_path', type=str, default=savePath) # './wordStyleOutPut_600_preprocess_0/'
    

    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json')
    #parser.add_argument('--stable_dif_path', type=str, default=stableDiffPath, help='path to stable diffusion')
    
    parser.add_argument('--stable_dif_path', type=str, default="/cluster/datastore/aniketag/allData/supportingSoftwares/stableDiffusion/", help='path to stable diffusion')

    parser.add_argument('--loadPrev', type=int, default=1)
    parser.add_argument('--ddp', type=int, default=0)

    parser.add_argument('--ocr', type=int, default=1, help='perform OCR of the image!!!')   # savedOcrImages
    parser.add_argument('--savedOcrImages',  type=int, default=1)
    parser.add_argument('--partialLoad',  type=int, default=0)
    parser.add_argument('--saveLogs',  type=int, default=0)
    parser.add_argument('--torch', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=0) 
    parser.add_argument('--partialCTC', type=int, default=0, help='use CTC loss only 10 percent of time')
    parser.add_argument('--fullSampling', type=int, default=0, help='call model every time')
    parser.add_argument('--loadPrevPath', type=str, default="/cluster/datastore/aniketag/allData/Htr/model/nor/norwegian9000_train_0_22sep/temp.pt")


    """
    parser.add_argument('--gt_train', type=str, default=gt_train) #  

    parser.add_argument('--csvRead', type=str, 
                        default=csvRead, 
                        help='training info from .csv instead of authors file') 
    """
    parser.add_argument('--authorBasePath', type=str, default= authorBasePath) # './wordStyleOutPut_600_preprocess_0/'

    parser.add_argument('--trascriptionPlusOCR', type=int, default=0,
                        help = "it joins transcription and OCR prediction as a conditional input")
    parser.add_argument('--phosc', type=int, default=0)
    parser.add_argument('--phos', type=int, default=1)
    parser.add_argument('--sidChange', type=int, default=0,help = "this modifies the writer while generating data")
    
    #parser.add_argument('--dataset', type=str, default='iam', help='iam or other dataset') 
    parser.add_argument('--dataset', type=str, default='norwegian', help='iam or other dataset') 
    parser.add_argument('--iam_path', type=str, default=dst_dir, help='path to iam dataset (images 64x256)')
    parser.add_argument('--device', type=str, default=device) 
    parser.add_argument('--device1', type=str, default=device1) 

    parser.add_argument('--lang', type=str, default= lang,help = "language") 
    parser.add_argument('--imgConditioned', type=int, default=0)

    
    n_gpus = torch.cuda.device_count()
    #assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    
    #print("\n\t no of gpu:",world_size)
    
    args = parser.parse_args()
    if args.wandb_log==True:
        runs = wandb.init(project='DIFFUSION_IAM', name=f'{args.save_path}', config=args)

        wandb.config.update(args)

    #print("\n\t args:",args)
    
    print("\n Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")    

    print("\n")

    print("\n\t is model:",os.path.isfile(args.authorBasePath+ckptModelName))
    print("\n\t is model2:",os.path.isfile(args.authorBasePath+emaModelName))

    print("\n\t is model:",args.authorBasePath+ckptModelName)
    print("\n\t is model2:",args.authorBasePath+emaModelName)
    print("\n\t loadPrevPath:",args.loadPrevPath)


    #exit()


    #assert args.phosc != 1 or MAX_CHARS == 10 or MAX_CHARS == 25, "MAX_CHARS should be 10 when args.phosc is 1"
    #assert args.phos != 1 or MAX_CHARS == 10 or MAX_CHARS == 25, "MAX_CHARS should be 10 when args.phos is 1"

    
    #create save directories
    #setup_logging(args)
    
    if args.lang == "ENG":
        net = HTRNet(cnn_cfg, head_cfg,54, head=head_type, flattening=flattening, stn=stn)
    elif args.lang == "NOR":
        net = HTRNet(cnn_cfg, head_cfg,54, head=head_type, flattening=flattening, stn=stn)
    
    
        
    if args.ddp==1:
        net = torch.nn.DataParallel(net).to(args.device)


    ocrModel = "/cluster/datastore/aniketag/newHTR/icpr/HTR-best-practices/model/temp_AllData.pt"

    if 1:#os.path.isfile(args.loadPrevPath):
        
        #net.load_state_dict(torch.load(args.loadPrevPath),strict= False)

        net.load_state_dict(torch.load(ocrModel),strict= True)

        print("\n\t Loading OCR model complete from path:",args.loadPrevPath)    
        
        """
        device = "cuda:1"
        device1 = 1

        """
        
        #print("\n\t is cuda:",torch.cuda.is_available())
        
        net.to(args.device)
        
        #print("\n\t net:",net)
        

    print('character vocabulary size', vocab_size)
    
    if 1:#args.dataset == 'iam' and args.dataset == 'norwegian':
        class_dict = {}
        for i, j in enumerate(os.listdir(f'{args.iam_path}')):
            class_dict[j] = i

        transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

            
    train_loader,style_classes,wr_dict,full_dict,image_wr_dict,img_word_dict = createDataLoader(args,transforms)
    
    #print("\n\t len train_loader:",len(train_loader)*args.batch_size)
    
    #print(full_dict)
    
    #input("check!!!")
    if args.ddp==1:
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, args=args, max_seq_len=OUTPUT_MAX_LEN)    
        unet = torch.nn.DataParallel(unet).to(args.device)
    else:    
        
        if args.phosc == 1 or args.phos == 1:
            
            #input("check!!")
            unet = UNetModelPhosc(image_size = args.img_size, in_channels=args.channels,
                            model_channels=args.emb_dim, out_channels=args.channels,
                            num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), 
                            channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes,
                            context_dim=args.emb_dim, vocab_size=vocab_size, 
                            args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device) 
            
            
        else:
            unet = UNetModel(image_size = args.img_size, in_channels=args.channels,
                            model_channels=args.emb_dim, out_channels=args.channels,
                            num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), 
                            channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes,
                            context_dim=args.emb_dim, vocab_size=vocab_size, 
                            args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device)    
            #input("check!!")
    
    #unet = torch.load("./wordStyleOutPut/models/ema_ckpt.pt")
    """    
    if args.loadPrev == 1 and os.path.isfile("./wordStyleOutPut/models/ckpt.pt"):
        unet.load_state_dict(torch.load("./wordStyleOutPut/models/ckpt.pt"))
        print("\n\t unet model loaded!!!")
    """
    # authorBasePath
    
    """
    if args.loadPrev == 1 and os.path.isfile("./regeneratedImages/models/ckpt4.pt"):
        unet.load_state_dict(torch.load("./regeneratedImages/models/ckpt4.pt"))
        print("\n\t unet model loaded!!!")
    """
    
    
    if args.loadPrev == 1 and os.path.isfile(args.authorBasePath+ckptModelName):
        
        print("\n\t device:",device," args.device:",args.device," \t model:",args.authorBasePath+ckptModelName)

        state_dict = torch.load(args.authorBasePath+ckptModelName,map_location=device)
        
        first_five_keys = list(state_dict.keys())[:5]
        print(first_five_keys)
        
        
        unet.load_state_dict(state_dict)
        print("\n\t loaded:",args.authorBasePath+ckptModelName)
    
    
    #import os

    #print("unet loading complete!!!",os.getcwd())

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)

    #optimizer = optimizer.load_state_dict(torch.load("./regeneratedImages/models/optim4.pt"))
    #optimizer = optimizer.load_state_dict(torch.load("./wordStyleOutPut/models/optim.pt"))

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    if args.ddp==1:
        ema_model = torch.nn.DataParallel(ema_model).to(args.device)
    
    """    
    if args.loadPrev == 1 and os.path.isfile("./wordStyleOutPut/models/ema_ckpt.pt"):
        ema_model.load_state_dict(torch.load("./wordStyleOutPut/models/ema_ckpt.pt"))
        print("\n\t ema model loaded!!!")
    """
    """
    if args.loadPrev == 1 and os.path.isfile("./regeneratedImages/models/ema_ckpt4.pt"):
        ema_model.load_state_dict(torch.load("./regeneratedImages/models/ema_ckpt4.pt"))
        print("\n\t ema model loaded!!!")
    """
    if args.loadPrev == 1 and os.path.isfile(args.authorBasePath+emaModelName):

        print("\n\t trying to loaded from:",args.authorBasePath+emaModelName)

        ema_model.load_state_dict(torch.load(args.authorBasePath+emaModelName,map_location=device))
        print("\n\t ema model loaded from ",args.authorBasePath+emaModelName)


    ema_model1 = copy.deepcopy(unet).eval().requires_grad_(False)

    if args.ddp==1:
        ema_model1 = torch.nn.DataParallel(ema_model1).to(args.device)

    """    
    if args.loadPrev == 1 and os.path.isfile("./wordStyleOutPut/models/ema_ckpt.pt"):
        ema_model1.load_state_dict(torch.load("./wordStyleOutPut/models/ema_ckpt.pt"))
        print("\n\t ema model loaded!!!")
    """
    """
    if args.loadPrev == 1 and os.path.isfile("./regeneratedImages/models/ema_ckpt4.pt"):
        ema_model1.load_state_dict(torch.load("./regeneratedImages/models/ema_ckpt4.pt"))
        print("\n\t ema model loaded!!!")
    """
    
    if args.loadPrev == 1 and os.path.isfile(args.authorBasePath+emaModelName):
        ema_model1.load_state_dict(torch.load(args.authorBasePath+emaModelName,map_location=device))
        print("\n\t ema1 model loaded!!!")

    #input("check!!!")
    
    
    if args.latent==True:
        print('Latent is true - Working on latent space')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        
        if args.ddp==1:
            vae = torch.nn.DataParallel(vae).to(args.device)
            
        else:
            vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        print('Latent is false - Working on pixel space')
        vae = None

    exit()
    
    train(diffusion, unet,net, ema, ema_model, ema_model1,vae, optimizer, mse_loss, train_loader, style_classes, vocab_size, transforms, args)


if __name__ == "__main__":
    main()
  
  
