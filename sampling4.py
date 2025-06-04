import torch
import torch.nn as nn
import argparse
import copy
from torch import optim
from train import setup_logging, Diffusion, EMA
from unet import UNetModel
from diffusers import AutoencoderKL
import os
import random
import torchvision
from PIL import Image
import cv2
import numpy as np    

"""
    convert from latent to image
"""
def latentToImage(self,x,vae):
    
    
    #print("\n\t 1.inside latentToImage x.shape:",x.shape)
    
    x = x.detach()
    
    latents = 1 / 0.18215 * x
    
    try:
        image = vae.decode(latents).sample
    except Exception as e:
        image = latents
        import torch.nn.functional as F

        image = F.interpolate(latents, size=(64, 256), mode='bilinear', align_corners=False)


    image = (image / 2 + 0.5).clamp(0, 1)
                        
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    image = torch.from_numpy(image)

    image = image.permute(0, 3, 1, 2)

    #print("\n\t 2.inside latentToImage image.shape:",image.shape) # torch.Size([44, 3, 64, 256])

    return image


class Diffusion:
    def __init__(self, noise_steps=600, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), args=None):
        self.noise_steps = 1000 #600 #noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.box = []
        
        self.device = args.device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t,args):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        
        #logger.info(" sqrt_alpha_hat.device:",sqrt_alpha_hat.device," x.device:",x.device," \t sqrt_one_minus_alpha_hat.device:",sqrt_one_minus_alpha_hat.device," Ɛ.device:",Ɛ.device)

        #return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ.to(args.device), Ɛ

        return x , Ɛ



    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
        #return torch.randint(low=299, high=300, size=(n,))

             

    """
        convert from latent to image
    """
    def latentToImage(self,x,vae):
        
        
        #print("\n\t 1.inside latentToImage x.shape:",x.shape)
        
        x = x.detach()
        
        latents = 1 / 0.18215 * x
        
        try:
            image = vae.decode(latents).sample
        except Exception as e:
            image = latents
            import torch.nn.functional as F

            image = F.interpolate(latents, size=(64, 256), mode='bilinear', align_corners=False)

    
        image = (image / 2 + 0.5).clamp(0, 1)
                            
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = torch.from_numpy(image)

        image = image.permute(0, 3, 1, 2)

        #print("\n\t 2.inside latentToImage image.shape:",image.shape) # torch.Size([44, 3, 64, 256])

        return image
    
        """    
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
                        
            allT1.append(image)
            
            image = image.cpu().permute(0, 2, 3, 1).numpy()
    
            image = torch.from_numpy(image)
            #x = image.permute(0, 3, 1, 2)
            allX1.append(image.permute(0, 3, 1, 2))
        """
     
    def masking(self,i,maskLatents,allMask_t_dilated):

        #print("\n\t 1.maskLatents =",maskLatents.shape," i:",i)
        
        maskLatents = torch.mean(maskLatents, dim=1, keepdim=True)  # Averaging over the channel dimension

        mask_t = (maskLatents > (1 - i/1000)).type(maskLatents.dtype) # So when t is high, most of the mask is 1 (fixed), but when t is low, most of the mask is 0 (variable)
        dilate_size = (int)(1.68*i/1000)
        #dilate_size = (int)(2*i /1000)

        mask_t_dilated = torch.nn.functional.max_pool2d(mask_t, dilate_size*2 + 1, stride=1, padding=dilate_size)
        
        #print("\n\t 2.mask_t_dilated =",mask_t_dilated.shape," i:",i)
        
        
        allMask_t_dilated.append(mask_t_dilated)
        
        #print("\n\t 2.mask_t_dilated =",mask_t_dilated.shape," i:",i," len:",len(allMask_t_dilated))
    
        
        
        """
        mask_image = mask_t_dilated.squeeze(1)
        print("\n\t 2.mask_t_dilated =",mask_t_dilated.shape)


        # Normalize to [0, 255] and convert to uint8
        mask_image = (mask_image - mask_image.min()) / (mask_image.max() - mask_image.min())  # Normalize to [0, 1]
        mask_image = (mask_image * 255).byte()  # Convert to [0, 255] and then to uint8

        # Convert tensor to PIL image
        mask_pil_image = transforms.ToPILImage()(mask_image.cpu())  # Convert to CPU and then to PIL format

        # Step 3: Save the image
        output_path = os.path.join("./imageDump/mask_t_dilated/", str(i)+"_"+"mask_t_dilated.png")
        mask_pil_image.save(output_path)

        """

        return mask_t_dilated,allMask_t_dilated






def sampling4(self,epoch,x_t,words,model, vae, n, x_text, labels, args):

    modelCall = 0
            
    print("\n\t 1.words:",words)
    
    noise_dict = {}#collections.defaultdict(list)
    model.eval()
    tensor_list = []
    all_noises = []
    allX = []  # predicted images
    allT = []  # original        

    allX1 = []  # predicted images
    allT1 = []  # original        
    allMask_t_dilated = []
    
    allImages = []
                
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
            
        args.fullSampling = 1
        
        x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                
        for i in reversed(range(1,600)):
            
            #print("\n\t resampling !!!!")
            with open("./flagGen.txt","r") as f:
                flag = int(f.read())
            
            if flag == 0:
                epoch = args.epochs+1
                i = 0
                break                
        
            t = (torch.ones(n) * i).long().to(self.device)
            
            #print("\n\t i:",i," flag =",flag)

            
            if args.fullSampling or ((i%(100)  ==0 or i%5==0 or i==self.noise_steps or i==(self.noise_steps-1) or (epoch>3 and i%(25) ==0) or (epoch>5 and i%(15) ==0) or (epoch>10 and i%(10) ==0) or epoch>50==0)):
                
                
                if 0:#args.phosc ==1 or args.phos ==1:
                    pass
                                                                    
                else:                
                    
                    #print("\n\t usinmg the mask latent!!!",labels)
                    
                    if 0:#i % 3==0 or i>300:
                        predicted_noise = model(x, None, t, text_features, labels)
                    else:
                        
                        randWrite = 0 #random.randint(0,100)
                        #print("\n\t randWrite =",randWrite)
                                                    
                        if 0:#args.attentionMaps == 1:
                            pass
                        else:
                            predicted_noise,attn1,attn2,attn3,_ = model(x, None, t, text_features, labels+randWrite)
                                                                                
            else:
                print("\n\t no denoising!!!")
                pass
            #allT.append(predicted_noise)
            try:
                all_noises.append(predicted_noise.detach().cpu())  # Append the noise tensor to the list
            except Exception as e:
                pass
                        
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            if args.fullSampling:
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) #+ torch.sqrt(beta) * noise            
            else:
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) #+ torch.sqrt(beta) * (noise/10)        
                
            if 0:#i%10  == 0:
                x = (0.1/7) * xOld + 1.000 * x
            
            if i ==1:
                latents_x0 = x
            elif i == 598:
                latents_x599 = x
                

        """
            converting mask latents to images
        """
        
        # x_t, noise = diffusion.noise_images(latents, t,args)
        mask_t_dilated1 = self.latentToImage(allMask_t_dilated[-1],vae)
        mask_t_dilated0 = self.latentToImage(allMask_t_dilated[0],vae)

        latents_x0 = self.latentToImage(latents_x0,vae)
        latents_x599 = self.latentToImage(latents_x599,vae)

        #model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
                        
            allT1.append(image)
            
            image = image.cpu().permute(0, 2, 3, 1).numpy()
    
            image = torch.from_numpy(image)
            #x = image.permute(0, 3, 1, 2)
            allX1.append(image.permute(0, 3, 1, 2))

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)

    print("\n\t modelCall:",modelCall)


    allT = torch.stack(allT)
    allT = allT.squeeze(0)

    if args.attentionMaps ==0:
        return 0,allX,allT,allX1,allT1,mask_t_dilated0,mask_t_dilated1,latents_x0,latents_x599
        