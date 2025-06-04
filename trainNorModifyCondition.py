"""

    lang: This parameter decides max characters
    
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
import copy
import argparse
import json
from diffusers import AutoencoderKL
from unet import UNetModel
#from unetPhosc import UNetModelPhosc

import wandb
import pandas as pd

import sys

from utils.createNorDataLoader import createDataLoader

sys.path.append("/cluster/datastore/aniketag/newWordStylist/WordStylist/ResPhoSCNetZSL/")

from  ResPhoSCNetZSL.modules.datasets import phosc_dataset
#from utils.createNorDataLoader import createDataLoader

import pickle
from configNor import *
#from utils.dataGenerationNorConfig import *

MAX_CHARS = MAX_CHARS

print("\n\t 1.MAX_CHARS = :",MAX_CHARS)

OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>

if lang == "ENG":
    c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
elif lang == "NOR":
    c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅabcdefghijklmnopqrstuvwxyzæøå'

cdict = {c:i for i,c in enumerate(c_classes)}
icdict = {i:c for i,c in enumerate(c_classes)}


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

def readFlags(args):
    
    with open(args.stopFlag,"r") as f:
        stopValue = int(f.readline())
    
    return stopValue


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


def save_images(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
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

        
            
    def __len__(self):
        return len(self.indices)
            
    
    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        
        
        if self.args.lang == "nor":
            wr_id = torch.tensor(0).to(torch.int64)
        else:
            wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)
        
        if self.args.phos ==1 or self.args.phosc ==1:
            phoscLabel = self.wordPhosc[label]#.astype(np.float32)
        else:
            phoscLabel = "NeglectMe"        
        
        img_path = os.path.join(self.image_path, image_name)
        
        #print("\n\t img_path=",img_path)
        #print("\n\t img_path=",os.path.isfile(img_path))

        
        image = Image.open(img_path).convert('RGB')
        
        #print("\n\t image.shape:",image.size)
        
        image = self.transforms(image)
        
        word_embedding = label_padding(label, num_tokens) 
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()    
        
        #print("\n\t 0.word_embedding.shape:",word_embedding.shape)
        
        #return image, word_embedding, wr_id, label,phoscLabel
        return image_name,image,word_embedding,wr_id,label,phoscLabel



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


    def sampling(self, model, vae, n, x_text, labels, args, mix_rate=None, cfg_scale=3):
        
        try:
            model.eval()
            tensor_list = []
            #if mix_rate is not None:
            #   print('mix rate', mix_rate)
            with torch.no_grad():
                
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
                
                for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                    t = (torch.ones(n) * i).long().to(self.device)
                    
                    if args.phosc ==1 or args.phos ==1:
                        predicted_noise = model(x, phoscLabels,timesteps=t,context=text_features)        
                    else:                
                        #predicted_noise = model(x,None,text_features,original_images=original_images, timesteps=t,  y=s_id, or_images=None)

                        #predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)

                        predicted_noise = model(x,None,timesteps=t,context=text_features,y=labels)

                    
                    if 0:#cfg_scale > 0:
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

        except Exception as e1:
            print("\n\t error in sampling:",e1)
            
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("\n\t line number:", exc_tb.tb_lineno)
                


def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, num_classes, vocab_size, transforms, args):
    model.train()
    
    print('Training started....')
    
    stopValue = readFlags(args)

    if stopValue == 0:
        #logger.info('Stopping Epoch stopValue:%s',stopValue)
        print("\n\t Stopping Epoch stopValue:",stopValue)
        exit()
    
    
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader,disable=False)
        
        for i, (image_name,images, word, s_id,label,phoscLabels) in enumerate(pbar):
            
            stopValue = readFlags(args)

            if stopValue == 0:
                #logger.info('Stopping Epoch stopValue:%s',stopValue)
                print("\n\t Stopping Epoch stopValue:",stopValue)
                exit()

            
            images = images.to(args.device)
            original_images = images
            text_features = word.to(args.device)
            """
            try:
                print("\n\t i:",i," \t images.shape:",images.shape," image_name:",image_name)
                print("\n\t wordLabel:",label)
                print("\n\t 2.word:",word.shape)

                #print("\n\t 3.s_id:",s_id.shape)
                ##print("\n\t 4.label.shape::",len(label))                
                #print("\n\t 5.phoscLabels:",phoscLabels.shape)

            except Exception as e:
                print(e)
                
            """

            #input("check here")
            
            s_id = s_id.to(args.device)
            
            if args.latent == True:
                images = vae.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
                latents = images
            
            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            x_t, noise = diffusion.noise_images(images, t)
            
            if np.random.random() < 0.1:
                labels = None
            
            
            if args.phosc ==1 or args.phos ==1:
                predicted_noise = model(x_t, phoscLabels,timesteps=t,context=text_features, y=s_id)        
            else:                
                predicted_noise = model(x_t,None,timesteps=t,context=text_features,y=s_id)
        
                #forward               ( x, original_images=None, timesteps=None, context=None, y=None, original_context=None, or_images=None, mix_rate=None, **kwargs)
            #print("\n\t predicted_noise.shape:",predicted_noise.shape)
            #predicted_noise = model(x_t, original_images=original_images, timesteps=t, context=text_features, y=s_id, or_images=None)
                    
            #input("check!!!")
            #continue
            

            loss = mse_loss(noise, predicted_noise)
            
            print("\n\t epoch:",epoch,"\t batch:",i,"t loss:",loss.item())

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            
    
        if epoch % 25 == 0:
            
            stopValue = readFlags(args)

            if stopValue == 0:
                #logger.info('Stopping Epoch stopValue:%s',stopValue)
                print("\n\t Stopping Epoch stopValue:",stopValue)
                exit()

            #print("\n\t epoch:",epoch)            
            # if args.img_feat is True:
            #     n=16
            #     labels = image_features
            # else:
            labels = torch.arange(16).long().to(args.device)
            n=len(labels)
            
            words = ['senere', 'angaaende', 'Bestemmelser']
            
            try:
                phoscLabels = None
                for x_text in words: 
                    print("\n\t saving images at location:",os.path.join(args.save_path, 'images', f"{x_text}_{epoch}.jpg"))
                    ema_sampled_images = diffusion.sampling(ema_model, vae, n=n, x_text=x_text,labels=labels, args=args)
                    sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{epoch}.jpg"), args)
                    if args.wandb_log==True:
                        wandb_sampled_ema= wandb.Image(sampled_ema, caption=f"{x_text}_{epoch}")
                        wandb.log({f"Sampled images": wandb_sampled_ema})
                        
            except Exception as e:
                
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("\n\t line number:", exc_tb.tb_lineno)
                
                
                print("\n\t error saving image:",e)
                exit()
                pass
            
            #torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
            #torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_ckpt.pt"))
            #torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))   
            
            try:
                print("\n\t saving model at location:",os.path.join(args.save_path,"models", args.saveModelName))
                torch.save(model.state_dict(), os.path.join(args.save_path,"models", args.saveModelName))
                torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models",args.saveModelName))
                #torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim_"+args.saveModelName))   
            except Exception as e:
                torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
                torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema.pt"))
                torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))   
                

import pickle

def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4, help='if latent is True channels should be 4, else 3')  
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    #parser.add_argument('--save_path', type=str, default='./save_path/')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json')
    parser.add_argument('--stable_dif_path', type=str, default="/cluster/datastore/aniketag/allData/supportingSoftwares/stableDiffusion//", help='path to stable diffusion')

    # experiment wise changing parameter

    parser.add_argument('--dataset', type=str, default='norwegian', help='iam or other dataset') 
    
    parser.add_argument('--iam_path', type=str, default=iam_path, help='path to iam dataset (images 64x256)')

    parser.add_argument('--device', type=str, default=device) 

    parser.add_argument('--gt_train', type=str, default=gt_train) #  
    parser.add_argument('--stopFlag', type=str, default = "./flags/stopFlagNor.txt",help ="flag to stop program") # partialLoad
    
    parser.add_argument('--csvRead', type=str, 
                        default=None, 
                        help='training info from .csv instead of authors file') 
    
    parser.add_argument('--loadPrev', type=int, default=0,help ="model from authorBasePath gets loaded")
    parser.add_argument('--save_path', type=str, default=save_path,help = "this is location where it savesthe new model" ) 
    parser.add_argument('--saveModelName', type=str, default= saveModelName ,help = "by this name save model at save_path" ) 
    
    parser.add_argument('--trascriptionPlusOCR', type=int, default=0,help = "it joins transcription and OCR prediction as a conditional input")

    parser.add_argument('--phosc', type=int, default=0)
    parser.add_argument('--phos', type=int, default=0)
    parser.add_argument('--authorBasePath', type=str, default= authorBasePath,help = "This is old model path") # './wordStyleOutPut_600_preprocess_0/'
    parser.add_argument('--lang', type=str, default= lang,help = "language") 

    parser.add_argument('--charLevelEmb', type=int, default=0,help = "the word level embeddings are calculated by concatenating char level embeddings")
    parser.add_argument('--charImages', type=int, default=0)
    parser.add_argument('--attentionMaps', type=int, default=0,help= "return attention maps")
    parser.add_argument('--ocrTraining', type=int, default=0) 
    parser.add_argument('--imgConditioned', type=int, default=0,help = "entire original image passed through preprocessing part and those embedding added with text embeddings")
    parser.add_argument('--wrdChrWrStyl', type=int, default=0)

    args = parser.parse_args()
    
    print("\n Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")    

    print("\n")

    assert args.phosc != 1 or (MAX_CHARS == 10 or MAX_CHARS == 25), "MAX_CHARS should be 10 when args.phosc is 1"
    assert args.phos != 1 or (MAX_CHARS == 10 or MAX_CHARS == 25), "MAX_CHARS should be 10 when args.phos is 1"

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
                
            wr_dict = {}
            full_dict = {}
            image_wr_dict = {}
            img_word_dict = {}
            wr_index = 0
            idx = 0
            

            for rowNo,i in enumerate(train_data):
                
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
    
    #unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device)    
    
    
    if args.dataset == "norwegian":
        full_dict,wr_dict,transforms = createDataLoader(args)

        print('number of train writer styles', len(wr_dict))
        style_classes=len(wr_dict)


        
        train_ds = IAMDataset(full_dict, args.iam_path, wr_dict, args, transforms=transforms)
        
        print("\n\t train_ds:",len(train_ds))
        
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        
        
        
    if args.phosc == 1 or args.phos == 1:
        
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

        print("\n\t unet model created:",OUTPUT_MAX_LEN)

    print("\n\t trying to load models from:",args.authorBasePath+"/models/"+ckptModelName," ",os.path.isfile(args.authorBasePath+"/models/"+ckptModelName))
    
    if args.loadPrev == 1 and os.path.isfile(args.authorBasePath+"/models/"+ckptModelName):
        
        unet.load_state_dict(torch.load(args.authorBasePath+"/models/"+ckptModelName,map_location=device))
        print("\n\t unet model loaded from:",args.authorBasePath+ckptModelName)

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)

    if 0:#args.loadPrev == 1 and os.path.isfile(args.authorBasePath+"optim.pt"):
        optimizer = optimizer.load_state_dict(torch.load(args.authorBasePath+"optim.pt",map_location=device))
        print("\n\t optimizer loaded from ",args.authorBasePath+"optim.pt")
    
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    if args.loadPrev == 1 and os.path.isfile(args.authorBasePath+emaModelName):
        ema_model.load_state_dict(torch.load(args.authorBasePath+emaModelName,map_location=device))
        print("\n\t ema model loaded from ",args.authorBasePath+emaModelName)
    
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
  
  
