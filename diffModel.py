import argparse
import sys
import os

cwd = os.getcwd()

addPath = "/cluster/datastore/aniketag/generation/HiGANplus/HiGAN+/"
sys.path.append(addPath)
print("\n\t cwd:",os.getcwd())

from lib.utils import yaml2config
from networks.model import GlobalLocalAdversarialModel


all_models = {
    'gl_adversarial_model': GlobalLocalAdversarialModel
}

def get_model(name):
    
    print("\n\t all_models[name]:",all_models[name])
    return all_models[name]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/generation/HiGANplus/HiGAN+/configs/gan_iam.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--ckpt",
        nargs="?",
        type=str,
        default='/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/generation/HiGANplus/HiGAN+/pretrained/HiGAN+.pth',
        help="checkpoint for evaluation",
    )

    parser.add_argument(
        "--mode",
        nargs="?",
        type=str,
        default="text",
        help="mode: [rand] [style] [text] [interp]",
    )

    args = parser.parse_args()
    
    print("\n\t args =",args)
        
    cfg = yaml2config(args.config)


    model = get_model(cfg.model)(cfg, args.config)
    #model.load(args.ckpt, cfg.device)"
    
    model.set_mode('eval')
    
    print("\n\t model:",model)
    

