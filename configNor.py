import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

lang = ["NOR","ENG"][0]

print("\n\t language:",lang)

allInOneIndx = 1
#MAX_CHARS = [10,42][allInOneIndx]

if lang == "NOR":
    MAX_CHARS = 25

print("\n\t MAX_CHARS=",MAX_CHARS)

if lang == "ENG":
    gt_train = ["./gt/charWordTrainIamAnnotation.txt","./gt/characterWordLevelAnnotation.txt","./gt/gan.iam.tr_va.gt.filter27","./gt/gan.iam.tr_va.gt.filter27",'./gt/results_IAM_train.filter27'][allInOneIndx]
elif lang == "NOR":
    #gt_train = "/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess_norwegian_gt/norwegian9000_train_0_All.filter27"

    gt_train ="/cluster/datastore/aniketag/newWordStylist/WordStylist/gt/norwegian/norwegian_train_data_sorted.txt"
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

authorBasePath = ["",
                  "/cluster/datastore/aniketag/allData/wordStylist/models/Norwegian/Mse_Nor_text_condi_FromScratch//"][allInOneIndx]
# charWord
ckptModelName =["","ckpt.pt"][allInOneIndx]

emaModelName  = ["","ema.pt" ][allInOneIndx]

if lang == "ENG":
    save_path = ["",""][allInOneIndx]
elif lang == "NOR":
    save_path = ["/cluster/datastore/aniketag/allData/wordStylist/models/Norwegian/Mse_Nor_text_condi_FromScratch/models/",
                 "/cluster/datastore/aniketag/allData/wordStylist/models/Norwegian/Mse_Nor_text_condi_FromScratch_ICDAR/"][allInOneIndx]

if lang == "ENG":
    saveModelName = [""][allInOneIndx]
elif lang == "NOR":
    saveModelName = ckptModelName
    
#optModelName = ["optim_Mse_text_Phos_condi_FromScratch.pt"][0]
device = "cuda:1"

#batch_size = 500

