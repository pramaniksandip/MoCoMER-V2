import torch
import pandas as pd
import os
class DataPreperation():
    def __init__(self, root_path):
        self.root_path = root_path
        #self.files = []
        self.image = []
        #self.output = []
        #self.data_image = []
        #self.data_output = []

    def Train_Test_Data(self):
      count = 0
      print("Data Preperation Initiated!!!")

      for i in os.listdir(self.root_path):
        count+=1
        #print(i)
        self.image.append(i)
        if (count % 500 == 0 ):
          print("Number of Data Prepared =====>{}".format(count))

      self.df_train = pd.DataFrame(list(zip(self.image)),columns =['Image'])

      return self.df_train

def multi_hot_encode(caption, char_to_index):
    encoding = torch.zeros(len(char_to_index), dtype=torch.float32)  # Initialize vector of zeros
    for char in caption:
        if char in char_to_index:
            encoding[char_to_index[char]] = 1  # Set the respective index to 1
        # else:
        #     print(f"{char}charater is not present in dictionary!")
    return encoding