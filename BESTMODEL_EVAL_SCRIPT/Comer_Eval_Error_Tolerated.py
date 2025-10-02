from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from torchvision.transforms import ToTensor
import torch
from PIL import Image
import pandas as pd
import os



ckpt = '<checkpoint_parth>'
model = LitCoMER.load_from_checkpoint(ckpt, strict=False)
print("Model Loading Successful.....!")
device = torch.device("cpu")
model = model.to(device)
model.eval()

data_path = "<CROHME data directory path>"

def count_mismatches(pred, caption):
    pred_len = len(pred)
    caption_len = len(caption)
    min_len = min(pred_len, caption_len)
    mismatches = abs(pred_len - caption_len)  # initial mismatches due to length differences

    for i in range(min_len):
        if pred[i] != caption[i]:
            mismatches += 1

    return mismatches

test_year = ["2014","2016","2019"]
print("Evaluation starting...")
result = "Expression Rate of CROHM Test Set ==============>"
for y in test_year:
    Correct_Pred = 0
    One_Mismatch_Pred = 0
    Two_Mismatch_Pred = 0
    Three_Mismatch_Pred = 0
    img_path = data_path + y + "/img"
    caption_file = data_path + y + "/caption.txt"

    # Reading the caption file
    df = pd.read_csv(caption_file, sep='\t', header=None, names=['img_name', 'caption'])

    count = 0
    for images in os.listdir(img_path):
        name = images  # keep full filename with extension
        img = Image.open(os.path.join(img_path, images))
        img = ToTensor()(img)
        mask = torch.zeros_like(img, dtype=torch.bool)
        hyp = model.approximate_joint_search(img.unsqueeze(0), mask)[0]
        pred = vocab.indices2label(hyp.seq)

        # Now match with caption file
        row = df.loc[df['img_name'] == name]
        if row.empty:
            print(f"Warning: No caption found for {name}")
            continue

        caption = row['caption'].values[0]
        count += 1
        result_correct = f"match => {count} =>{Correct_Pred} => {Correct_Pred}/{count}\n"
        #print(f"Processing {y} Image No: {count} and correct prediction {Correct_Pred} and Exprate {{Correct_Pred}/{count}}")
        print(f"Processing {y} Image No: {count} and correct prediction {Correct_Pred} and Exprate {(Correct_Pred / count) * 100:.2f}%")
        mismatches = count_mismatches(pred, caption)
        with open("results_printimage.txt", "a") as file:
                file.write(result_correct)

        if (pred == caption):
            Correct_Pred += 1
            One_Mismatch_Pred += 1
            Two_Mismatch_Pred += 1
            Three_Mismatch_Pred += 1
            #print("Match", images)
            result = f"match => {images} =>{caption} => {pred}\n"
#             with open("results_printimage.txt", "a") as file:
#                 file.write(result)
        
        if (mismatches == 1):
            One_Mismatch_Pred += 1
            Two_Mismatch_Pred += 1
            Three_Mismatch_Pred += 1
            result = f"One Missmatch => {images} =>{caption} => {pred}\n"
            #with open("results_printimage.txt", "a") as file:
                #file.write(result)
            
        if (mismatches == 2):
            Two_Mismatch_Pred += 1
            Three_Mismatch_Pred += 1
            result = f"Two Missmatch => {images} =>{caption} => {pred}\n"
            #with open("results_printimage.txt", "a") as file:
                #file.write(result)
            
        if (mismatches == 3):
            Three_Mismatch_Pred += 1
            result = f"Three Missmatch => {images} =>{caption} => {pred}\n"
#             with open("results_printimage.txt", "a") as file:
#                 file.write(result)

    print(f"Results for year {y}:")
    print(f"Exp_Rate: {Correct_Pred/(len(df))}")
    print(f"One_Mismatch_Exp-Rate: {One_Mismatch_Pred/(len(df))}")
    print(f"Two_Mismatch_Exp_Rate: {Two_Mismatch_Pred/(len(df))}")
    print(f"Three_Mismatch_Exp_Rate: {Three_Mismatch_Pred/(len(df))}")
    
    result = f"Results for year {y}:\n" +  f"Exp_Rate: {Correct_Pred/(len(df))}\n" + f"One_Mismatch_Exp-Rate: {One_Mismatch_Pred/(len(df))}\n" + f"Two_Mismatch_Exp_Rate: {Two_Mismatch_Pred/(len(df))}\n" + f"Three_Mismatch_Exp_Rate: {Three_Mismatch_Pred/(len(df))}\n"
    
    Exp_Rate = Correct_Pred/(len(df))

    result = result + f"Expression Rate of CROHME {y} Test Set ==============> {Exp_Rate}\n"
    print(result)
    
    # Writing the result to a file in append mode
    with open("results.txt", "a") as file:
        file.write(result)
