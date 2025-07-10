def getMTLregions(MTL_labels):
    ''' see brain_labels.py for MTL_labels ''' 
    HPC_labels = [MTL_labels[i] for i in [0,1,2,3,4,9,10,11,12,13,25,30,35,40,45,46,49,52,53,56]] # all labels within HPC
    ENT_labels = [MTL_labels[i] for i in [6,15,21,24,29,34,39,47,54]] # all labels within entorhinal
    PHC_labels = [MTL_labels[i] for i in [7,16,20,26,31,36,41,48,55]] # all labels within parahippocampal
    return HPC_labels,ENT_labels,PHC_labels  
