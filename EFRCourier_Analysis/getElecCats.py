def getElecCats(reader):  
    try:
        elec_cats = reader.load('electrode_categories') # contains info about seizure onset zone, interictal spiking, broken leads.
        bad_elec_status=str(len(elec_cats))+' electrode categories'
    except:
        bad_elec_status= 'failed loading electrode categories'
        elec_cats = []
    return elec_cats,bad_elec_status       
    