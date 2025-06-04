import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

lang = ["NOR","ENG"][1]

print("\n\t language:",lang)

allInOneIndx = 0
MAX_CHARS = [10,42][allInOneIndx]

if lang == "NOR":
    MAX_CHARS = 25

print("\n\t 0.MAX_CHARS=",MAX_CHARS)

if lang == "ENG":
    gt_train = ["./gt/charWordTrainIamAnnotation.txt",
                "./gt/characterWordLevelAnnotation.txt",
                "./gt/gan.iam.tr_va.gt.filter27",
                "/cluster/datastore/aniketag/newWordStylist/WordStylist/gt/delMe_1.txt",
                "/cluster/datastore/aniketag/newWordStylist/wordStylist2/WordStylist/gt/gan.iam.tr_va.gt.filter27",
                './gt/results_IAM_train.filter27'][-2]
elif lang == "NOR":
    gt_train = "/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess_norwegian_gt/norwegian9000_train_0_All.filter27"

if lang == "ENG":
    dataIndx = 1
elif lang == "NOR":
    dataIndx = 0

iam_path = [
            "/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess_norwegian/",
            '/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/'
            ][dataIndx]

print("\n\t 1.iam_path=",iam_path)

csvRead = [None,"/cluster/datastore/aniketag/newHTR/HTR-best-practices/allResults/IAM/resultsTrainForDiffusion.csv",None][allInOneIndx]

authorBasePath = [
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/authorsModel/models/models/",
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/",   
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/Mse_Style_only_text_condi_FromScratch/models/",
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/Mse_Style_text_condi_FromScratch/models/",
                 "/cluster/datastore/aniketag/allData/wordStylist/models/Norwegian/Mse_Nor_text_condi_FromScratch/models/",
                  "/cluster/datastore/aniketag/allData/wordStylist/models/Norwegian/Mse_Nor_text_Phos_condi_FromScratch/models/",
                  "/cluster/datastore/aniketag/newWordStylist/WordStylist/models/Mse_text_Phos_condi_FromScratch/models/",
                  "/cluster/datastore/aniketag/newWordStylist/WordStylist/models/Mse_Transcription_OcrPred_FromScratch/models/",  
                  "/cluster/datastore/aniketag/WordStylist/authorsModel/models/" +"/models/",
                  "/cluster/datastore/aniketag/WordStylist/regeneratedImages/models/",
                  "/cluster/datastore/aniketag/WordStylist/authorPlusHtrBestFineTuneOnlyCTCnoMSE/models/",
                  "/cluster/datastore/aniketag/WordStylist/CTCMfromScratch/models/",
                  "/cluster/datastore/aniketag/WordStylist/CtcMSEimageMSEfromScratch/models/",
                  "/cluster/datastore/aniketag/WordStylist/models/MseOnlyPhoscfromScratch/",
                  "/cluster/datastore/aniketag/WordStylist/models/Mse_OnlyPhocfromScratch/",
                  "/cluster/datastore/aniketag/WordStylist/models/Mse_OcrPredFromScratch/",
                  ][allInOneIndx]
# charWord
ckptModelName =["ema_ckpt.pt",
                "ckpt_ema_charLevelEmb.pt",
                "ema_TextMse.pt",
                "TextMse.pt",
                "ema_TextNoWriterMseCTCLoss.pt",
                "charImageNoWriter_2_qkvChange.pt",
                "charImageNoWriter_2.pt",
                "charImageNoWriter_1.pt",
                "ckpt_Mse_Image_text_condi_FromScratch.pt",
                "ckpt_Mse_CharImage_text_condi_FromScratch.pt",
                "ckpt_Mse_Style_only_text_condi_FromScratch.pt",
                "ckpt_Mse_Style_text_condi_FromScratch.pt",
                "ckpt_Mse_text_Phos_condi_FromScratch.pt",
                "ckpt_Transcription_OcrPred_FromScratch.pt",
                "ckpt_Mse_OnlyPhocFromScratch.pt",
                "ckpt_MseOnlyPhoscFromScratch.pt",
                "ckptHtrBestOnlyImageMSECTCMfromScratch.pt",
                "ckptauthorPlusHtrBestFineTuneOnlyCTCnoMSE.pt",][allInOneIndx]

emaModelName  = [ "ema_ckpt.pt",
                "ckpt_ema_charLevelEmb.pt",
                "ema_TextMse.pt",
                "TextMse.pt",
                "ema_TextNoWriterMseCTCLoss.pt",
                "ema_charImageNoWriter_2_qkvChange.pt",
                "ema_charImageNoWriter_1.pt",
                "ema_Mse_Image_text_condi_FromScratch.pt",
                "ema_Mse_CharImage_text_condi_FromScratch.pt",
                "ema_Mse_Style_only_text_condi_FromScratch.pt",
                "ema_Mse_Style_text_condi_FromScratch.pt",
                "ema_Mse_text_Phos_condi_FromScratch.pt",
                "ema_Transcription_OcrPred_FromScratch.pt",
                "ema_Mse_OnlyPhocFromScratch.pt",
                "ema_MseOnlyPhoscFromScratch.pt",
                "ema_ckptHtrBestOnlyImageMSECTCMfromScratch.pt",
                "ema_ckptauthorPlusHtrBestFineTuneOnlyCTCnoMSE.pt"][allInOneIndx]

if lang == "ENG":
    save_path = ["/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/variableDataGeneration/attentionExp1/",
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/models/", 
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/",                 
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/Mse_only_text_condi_HiGanArchitect/models/",                 
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/Mse_only_textAndImage_condi_FromScratch/models/",
                 "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/Mse_Style_only_text_condi_FromScratch/models/",
                "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/Mse_Style_text_condi_FromScratch/models/",
                 "./models/Mse_text_Phos_condi_FromScratch/","./models/Mse_Transcription_OcrPred_FromScratch/"][allInOneIndx]
elif lang == "NOR":
    save_path = "/cluster/datastore/aniketag/allData/wordStylist/models/Norwegian/Mse_Nor_text_condi_FromScratch_ICDAR/models/generatedData/"
    
    """
    ["/cluster/datastore/aniketag/allData/wordStylist/models/Norwegian/Mse_Nor_text_condi_FromScratch/models/",
                 "/cluster/datastore/aniketag/allData/wordStylist/models/Norwegian/Mse_Nor_text_Phos_condi_FromScratch/"][allInOneIndx]
    """
if lang == "ENG":
    saveModelName = [
                    "ema_ckpt.pt",
                    "ckpt_ema_charLevelEmb.pt",
                    "ema_charLevelEmb.pt",
                    "ema_TextMse.pt",
                    "TextMse.pt",
                     "ema_TextNoWriterMseCTCLoss.pt",
                     "ema_ema_charImageNoWriter_2_qkvChange.pt",
                     "charImageNoWriter.pt",
                     "Mse_Image_text_condi_HiGanArchitect.pt",
                     "Mse_CharImage_text_condi_HiGanArchitect.pt","Mse_only_text_condi_HiGanArchitect.pt","Mse_Style_only_text_condi_FromScratch.pt",
                     "Mse_Style_text_condi_FromScratch.pt","Mse_text_Phos_condi_FromScratch.pt","Mse_Transcription_OcrPred_FromScratch_1.pt"][allInOneIndx]
elif lang == "NOR":
    saveModelName = "temp.pt"
    
#optModelName = ["optim_Mse_text_Phos_condi_FromScratch.pt"][0]
device = "cuda:1"

batch_size = 2

"""
sys.path.append("/cluster/datastore/aniketag/WordStylist/ResPhoSCNetZSL//")

#gpu_id = "cuda:1"
MAX_CHARS = 32
OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>
c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

pwd = os.getcwd()
wordStylistBase = "/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/"
dst_dir = '/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/'
stableDiffPath = "/cluster/datastore/aniketag/WordStylist/stableDiffusion//"
gtPath = ["/cluster/datastore/aniketag/WordStylist/gt/gan.iam.tr_va.gt.filter27",
          "/cluster/datastore/aniketag/WordStylist/gt/results_IAM_train.filter27"][0]

#stableDiffPath = "/media/aniketag/c4eb0693-4a65-4f0c-8d65-a6dad4b97ff9/allCondaEnv/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0/"
#gtPath = "./gt/gan.iam.tr_va.gt.filter27"


"""
    #model name for loading
"""
ckptModelName =["ckpt_Mse_OnlyPhocFromScratch.pt",
                "ckpt_MseOnlyPhoscFromScratch.pt",
                "ckptHtrBestOnlyImageMSECTCMfromScratch.pt",
                "ckptauthorPlusHtrBestFineTuneOnlyCTCnoMSE.pt"][0]

emaModelName  = ["ema_Mse_OnlyPhocFromScratch.pt",
                 "ema_MseOnlyPhoscFromScratch.pt",
                 "ema_ckptHtrBestOnlyImageMSECTCMfromScratch.pt",
                 "ema_ckptauthorPlusHtrBestFineTuneOnlyCTCnoMSE.pt"][0]

#"/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/models/models/"
#save_path = pwd +'/wordStyleOutPut_1000_preprocess_0_mini/'

save_path = [
            "./CTCimageMSEfromScratch/",
            "./CTCfromScratch/",
            "./authorPlusHtrBestCTCMseFromScratch/",
             "./authorPlusHtrBestFineTuneOnlyCTCnoMSE/",
             "./authorPlusHtrBestFineTune/",
             "./htrPreprocessAgain/"][0]

htrBaseDir = "/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/wordStyleOutPut/HTR-best-practices/"

device= "cuda:1"
#classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#classes = '_' + ''.join(classes)
#cdict = {c:i for i,c in enumerate(classes)}
#icdict = {i:c for i,c in enumerate(classes)}

#cdict = {c:i for i,c in enumerate(c_classes)}
#icdict = {i:c for i,c in enumerate(c_classes)}

version = "eng"
if version == 'eng':
    alphabet_csv = "/cluster/datastore/aniketag/WordStylist/ResPhoSCNetZSL/modules/utils/Alphabet.csv"
elif version == 'gw':
    alphabet_csv = '/cluster/datastore/aniketag/WordStylist/ResPhoSCNetZSL/modules/utils/AlphabetGW.csv'
elif version == 'nor':
    alphabet_csv = "/cluster/datastore/aniketag/WordStylist/ResPhoSCNetZSL/modules/utils/AlphabetNorwegian.csv"

# fuser -v /dev/nvidia0
"""