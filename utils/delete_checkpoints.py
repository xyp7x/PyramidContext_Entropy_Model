
import os
import numpy as np


def findCheckPoint(modelFolder, before_epoch,remainEpoch):

    delete_files=[]
    delete_checkpointEpoch=[]

    if not os.path.exists(modelFolder):
        return -1

    fileList=os.listdir(modelFolder)
    for fname in fileList:
        if 'epoch' in fname:
            checkpointEpoch=int(fname.split('epoch')[1].split('_')[0])

            if checkpointEpoch<before_epoch and checkpointEpoch not in remainEpoch:
                delete_files.append(os.path.join(modelFolder,fname))
                delete_checkpointEpoch.append(checkpointEpoch)
                
    return delete_files,delete_checkpointEpoch


modelFolder='/home/dingding/MM/VCM/FCVCM/CfP/DKIC_CfP/saveModels/opimg_det/2.0'
BeforeEpoch=400
remainEpoch=[200,250,300,350,399]

delete_files,delete_checkpointEpoch=findCheckPoint(modelFolder,BeforeEpoch,remainEpoch)
delete_index=np.argsort(np.array(delete_checkpointEpoch))
delete_files=np.array(delete_files)[delete_index]


for file in delete_files:   
    print(file)

for file in delete_files: 
    os.remove(file)
