def brain_label():
    from getMTLregions import getMTLregions
#     %run '/home1/noaherz/Long2017/git_repos/False-memory/brain_labels.py'
    MTL_stein = ['left ca1','left ca2','left ca3','left dg','left sub','left prc','left ec','left phc','left mtl wm',
             'right ca1','right ca2','right ca3','right dg','right sub','right prc','right ec','right phc','right mtl wm',
             'left amy','right amy'] # including amygdala in MTL
    MTL_ind = ['parahippocampal','entorhinal','temporalpole',   
               ' left amygdala',' left ent entorhinal area',' left hippocampus',' left phg parahippocampal gyrus',' left tmp temporal pole', # whole-brain names
               ' right amygdala',' right ent entorhinal area',' right hippocampus',' right phg parahippocampal gyrus',' right tmp temporal pole',
               'left amygdala','left ent entorhinal area','left hippocampus','left phg parahippocampal gyrus','left tmp temporal pole',
               'right amygdala','right ent entorhinal area','right hippocampus','right phg parahippocampal gyrus','right tmp temporal pole',
               '"ba35"','"ba36"','"ca1"', '"dg"', '"erc"', '"phc"', '"sub"',
               'ba35', 'ba36','ca1','dg','erc','phc','sub']
    MTL_labels = MTL_stein+MTL_ind

    HPC_labels,ENT_labels,PHC_labels =getMTLregions(MTL_labels)
    PHG_labels=ENT_labels+PHC_labels+['prc', 'ba35', 'ba36','left prc','right prc']
    return HPC_labels,PHG_labels
