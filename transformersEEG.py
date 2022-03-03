from transformers import DistilBertModel, DistilBertConfig
import sys
import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import Trainer
npz = np.load('hilbert.npz')
np_eeg = npz['EEG']
np_labels = npz['labels']
np_eeg_processed = []
np_labels_processed = []
#print(np_eeg)
#print(np_labels)
for item in np_eeg:
    np_eeg_processed.append(item[0])
for item in np_labels:
    np_labels_processed.append(item[1])
#print(np_eeg_processed)
#print(np_labels_processed)
X_train, X_test, y_train, y_test = train_test_split(np_eeg_processed,np_labels_processed,test_size=0.2,random_state=28)
class MyDataset(Dataset):

    def __init__(self,file_name):
        price_df=pd.read_csv(file_name)

        x=price_df.iloc[:,0:8].values
        y=price_df.iloc[:,8].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

print("ok")
def trainingArgs(
        epochs: int,
        trainDir: str,
        batchSizeTrain=16,
        batchSizeEval=32,
        training_set_len = len(train_dataset)
):
    """Return a TrainingArguments instance to be passed to Trainer class."""
    totalSteps = int((training_set_len / batchSizeTrain) * epochs)
    warmup = int(totalSteps * 0.05)
    return TrainingArguments(
        output_dir=f"./{trainDir}/results",
        logging_dir=f"./{trainDir}/logs",

        overwrite_output_dir=True,
        # trains faster without evaluation
        #evaluate_during_training=False,

        per_device_train_batch_size=batchSizeTrain,
        per_device_eval_batch_size=batchSizeEval,

        num_train_epochs=epochs,
        warmup_steps=warmup,

        # I won't be logging or checkpointing since
        # training occurs fairly quickly
        logging_steps=9999,
        save_steps=9999,
        save_total_limit=1,

        # standard arguments
        learning_rate=5e-5,
        weight_decay=1e-2,
    )
# training arguments
trainDir = "training"
saveModelDir = "tuned-bert"
epochsList = [2, 3, 4]

embArgs= trainingArgs(2, trainDir)
trainDs = train_dataset
testDs = test_dataset
# training arguments
trainDir = "training"
saveModelDir = "tuned-bert"
epochsList = [2, 3, 4]

embArgs= trainingArgs(2, trainDir)
trainDs = train_dataset
testDs = test_dataset

#Trainer(model="distilbert-base-uncased",train_dataset=X_train,eval_dataset=X_test)

