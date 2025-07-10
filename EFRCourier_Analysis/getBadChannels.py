def getBadChannels(pairs,elec_cats,remove_soz_ictal):
    ''' load information on seizure onset zone and bad electrodes'''
    import numpy as np
    bad_bp_mask = np.zeros(len(pairs))
    if elec_cats != []:
        if remove_soz_ictal == True:
            bad_elecs = elec_cats['bad_channel'] + elec_cats['soz'] + elec_cats['interictal']
        else:
            bad_elecs = elec_cats['bad_channel']
        for row_num in range(0,len(pairs)): 
            labels=pairs.iloc[row_num]['label']
            elec_labels = labels.split('-')
            if elec_labels[0] in bad_elecs or elec_labels[1] in bad_elecs:
                bad_bp_mask[row_num] = 1
    return bad_bp_mask