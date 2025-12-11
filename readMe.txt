
Code for the paper "Word-Diffusion: Diffusion-Based Handwritten Text Word Image Generation" .

This code is based on https://github.com/koninik/WordStylist

To train the model use trainModifyCondition.py

For inference use regenerateFromtrain2.py

2) For GW dataset use file trainGWModifyCondition.py and data in file washingtondb-v1.0.tar.gz

Change following in code 

    parser.add_argument('--stable_dif_path', type=str, default="", help='path to stable diffusion')
    parser.add_argument('--iam_path', type=str, default='/', help='path to iam dataset (images 64x256)')




