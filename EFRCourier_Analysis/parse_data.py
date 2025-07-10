import json
import pandas as pd
import numpy as np
import glob
import os


def add_recalled(events):
    '''
    uses REC_WORD and WORD events to determine if a word is recalled.
    Requires list field to be populated.
    '''
    def find_recall(row):
        item = row["item"]
        recalls = events[(events["type"] == 'REC_WORD') & (events["trial"] == row["trial"])]["item"].values
        return 1 if item in recalls else 0

    events = events.sort_values("mstime")
    events.loc[events["type"] =="WORD", "recalled"] = events.loc[events["type"] == "WORD"].apply(find_recall, axis=1)
    return events    

def add_final_recalled(events):
    
    def mark_recall(row):
        word_evs = events.query('type == "WORD"')
        all_words = word_evs.item.values
        return 1 if row["item"] in all_words else 0
    
    def find_recall(row):
        item = row["item"]
        recalls = events[(events["type"] == 'FFR_REC_WORD')]["item"].values
        return 1 if item in recalls else 0
    
    events = events.sort_values("mstime")
    events.loc[events["type"] =="FFR_REC_WORD", "recalled"] = events.loc[events["type"] == "FFR_REC_WORD"].apply(mark_recall, axis=1)
    events.loc[events["type"] =="WORD", "final_recalled"] = events.loc[events["type"] == "WORD"].apply(find_recall, axis=1)
    return events

def get_item_id(item, wordpool):
    if item not in wordpool:
        return -1
    return wordpool.index(item)+1

def add_itemno(events, wordpool):
    events.loc[(events["type"] == 'WORD'), "itemno"] = \
    events.loc[(events["type"] == 'WORD'), "item"].apply(lambda x: get_item_id(x, wordpool))
    
    events.loc[(events["type"] == 'REC_WORD'), "itemno"] = \
    events.loc[(events["type"] == 'REC_WORD'), "item"].apply(lambda x: get_item_id(x, wordpool))
    
    events.loc[(events["type"] == 'CUED_REC_WORD'), "itemno"] = \
    events.loc[(events["type"] == 'CUED_REC_WORD'), "item"].apply(lambda x: get_item_id(x, wordpool))
    
    events.loc[(events["type"] == 'FFR_REC_WORD'), "itemno"] = \
    events.loc[(events["type"] == 'FFR_REC_WORD'), "item"].apply(lambda x: get_item_id(x, wordpool))
    return events

def add_intrusion(events):
    '''
    uses REC_WORD and WORD events to determine if a recalled word is a PLI/XLI
    '''
    events = events.sort_values("mstime")

    def check_list(row):
        presentation = events[(events["type"] == 'WORD') \
                              & (events["itemno"] == row["itemno"]) \
                              & (events["trial"] <= row["trial"])]
        if len(presentation.index) == 0:
            return -1
        else:
            # also captures repeated presentations
            list_delta = row["trial"] - presentation.iloc[-1]["trial"]
            # list_delta = list_delta.values[0] # Series are annoying
            return -1 if list_delta < 0 else list_delta
    
    def check_all_list(row):
        presentation = events[(events["type"] == 'WORD') \
                              & (events["itemno"] == row["itemno"])]
        if len(presentation.index) == 0:
            return -1
        else:
            # also captures repeated presentations
            list_delta = presentation.iloc[-1]["trial"]
            # list_delta = list_delta.values[0] # Series are annoying
            return -1 if list_delta < 0 else list_delta

    events.loc[events["type"] == 'REC_WORD', "intrusion"] = events[events["type"] == 'REC_WORD'].apply(check_list, axis=1)
    events.loc[events["type"] == 'CUED_REC_WORD', "intrusion"] = events[events["type"] == 'CUED_REC_WORD'].apply(check_list, axis=1)
    events.loc[events["type"] == 'FFR_REC_WORD', "intrusion"] = events[events["type"] == 'FFR_REC_WORD'].apply(check_all_list, axis=1)

    return events

def add_keypress(events, rec_type, rec_stop_type):
    rec_evs = events.query('type == @rec_type | type == @rec_stop_type')
    rec_evs_index = rec_evs.index
    
    for i in range(len(rec_evs)-1):
        
        curr_row = rec_evs.iloc[i]
        if curr_row.type != rec_type:
            continue
        next_row = rec_evs.iloc[i+1]

        curr_time = curr_row.mstime
        next_time = next_row.mstime

        EFR_candidates = events.query('type == "keypress" & \
                                       mstime > @curr_time & mstime < @next_time')
        if len(EFR_candidates) > 0:
            response_time = EFR_candidates.mstime.values[0]

            events.at[rec_evs_index[i], "keypress"] = True
            events.at[rec_evs_index[i], "keypress_mstime"] = response_time
        
        # if no correct key press happened, check for additional key presses
        else:
            next_candidates = events.query('type == "key press/release" & \
                                            mstime > @curr_time & mstime < @next_time')
            if len(next_candidates) > 0:
                responses = next_candidates["key code"].unique()
                responses = responses[~np.isnan(responses)]
                response_time = next_candidates.mstime.values[0]
                
                events.at[rec_evs_index[i], "keypress_keycode"] = responses.tolist()
                events.at[rec_evs_index[i], "keypress_mstime"] = response_time
    
    return events


def main():
    data_dir = "/scratch/new_courier_pilot/"
    sub_dirs = []
    
    # get wordpool 
    with open('/scratch/new_courier_pilot/dbpool.txt', "r") as f:
        wordpool = [w.strip().lower() for w in f.readlines()]
        
    # get subject level directory info
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith("session.jsonl"):
                sub_dirs.append(root)

    for sub_dir in sub_dirs:
        _, _, _, sub, sess_txt = sub_dir.split("/")
        sess = int(sess_txt[-1])
        print("subject {}, sess {}".format(sub, sess))

        # check if the data is already saved in data_dir
        file_name = "{}_sess_{}.csv".format(sub, sess)
        file_dir = os.path.join(data_dir, file_name)
        if os.path.exists(file_dir):
            print("already saved")
            continue


        sub_data = []
        # locate session.jsonl
        sess_dir = os.path.join(data_dir, sub_dir, "session.jsonl")
        for line in open(sess_dir, "r"):
            # replace this specific entry to empty string
            if '"point condition":SerialPosition,' in line:
                line = line.replace('"point condition":SerialPosition,', '')
            elif '"point condition":SpatialPosition,' in line:
                line = line.replace('"point condition":SpatialPosition,', '')
            elif '"point condition":Random,' in line:
                line = line.replace('"point condition":Random,', '')

            data_dict = json.loads(line)
            sub_data.append(data_dict)

        # create dataframe
        for d in sub_data:
            for k,v in d["data"].items():
                d[k] = v
        df = pd.DataFrame(sub_data)

        column_names = ["type", "time", "trial number", "item name", "item", "serial position", 
                        "store", "store name", "store position", 
                        "player position", "positionX", "positionZ", 
                        "correct direction (degrees)", "pointed direction (degrees)", "key code"]
        df = df[column_names]
        df = df.rename(columns={
                                'trial number': 'trial',
                                'time': 'mstime',
                                'serial position': 'serialpos',
                                'item': 'cued_item',
                                'store': 'cued_store',
                                'store name': 'store',
                                'item name': 'item'
                                })
        df["subject"] = sub
        df["session"] = sess
        df["recalled"] = -999
        df["intrusion"] = -999

        # change the type name to match lab's data frame formatting
        df = df.replace({'type': {'object presentation begins': 'WORD', 
                                  'free recall': 'REC_WORD',
                                  'object recall recording start': 'REC_START',
                                  'object recall recording stop': 'REC_STOP',
                                  'cued recall recording stop': 'CUED_REC_STOP',
                                  'cued recall': 'CUED_REC_WORD',
                                  'final store recall': 'FSR_REC_WORD',
                                  'final store recall recording stop': 'FSR_REC_STOP',
                                  'final object recall': 'FFR_REC_WORD',
                                  'final object recall recording stop': 'FFR_REC_STOP'
                                 }
                        })

        ###############################################################################################
        # FREE RECALL
        ###############################################################################################
        recall_df = pd.DataFrame(columns = df.columns)
        rec_starts = df.query('type == "REC_START"')
        # first REC_START will indicate practice recall stage for session 0
        if sess == 0:
            rec_starts = rec_starts[1:]

        count = 0
        for i, row in rec_starts.iterrows():
            rec_start_time = row.mstime

            # let's load appropriate .ann files
            annotation_file = "{}.ann".format(count)
            annotation_dir = os.path.join(data_dir, sub_dir, annotation_file)
            if not os.path.exists(annotation_dir):
                print("...Missing DD #{} Free Recall Annotation".format(count))
                continue
                
            with open(annotation_dir, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if line[0] != "#":
                        recall_info = line.strip("\n").split("\t")

                        if len(recall_info) == 3:
                            rectime = float(recall_info[0]) + rec_start_time
                            item = recall_info[2].lower()
                            itemno = recall_info[1]

                            recall_df = recall_df.append({"subject":sub,
                                                          "session":sess,
                                                          "type":"REC_WORD", 
                                                          "mstime":rectime, 
                                                          "trial":count, 
                                                          "item":item, 
                                                          "itemno":itemno
                                                         }, ignore_index=True)
            count += 1

        # then, fill out the blanks
        recall_df["store"] = recall_df["store"].astype(str)
        recall_df["store position"] = recall_df["store position"].astype(str)

        for i, row in recall_df.iterrows():
            word_evs = df[(df.trial == row.trial) & (df.type == "WORD")]

            row_item = row["item"]
            recall_word = word_evs.query('item == @row_item')

            if len(recall_word) != 0:        
                serialpos = recall_word["serialpos"].values[0]
                store = recall_word["store"].values[0]
                store_position = recall_word["store position"].values[0]

                recall_df.at[i, "serialpos"] = serialpos
                recall_df.at[i, "store"] = store
                recall_df.at[i, "store position"] = store_position

            else:
                recall_df.at[i, "serialpos"] = -999

        ###############################################################################################
        # CUED RECALL
        ###############################################################################################
        cued_rec_start = df.query('type == "start cued recall"').mstime.values
        if sess == 0:
            cued_rec_start = cued_rec_start[1]
        else:
            cued_rec_start = cued_rec_start[0]

        cued_rec_events = df.query('type == "cued recall recording start" & mstime >= @cued_rec_start')
        cued_recall_df = pd.DataFrame(columns = df.columns)

        for i, row in cued_rec_events.iterrows():
            trial = int(row.trial)
            cued_item = row.cued_item
            cued_store = row.cued_store
            cued_time = row.mstime

            word_evs = df.query('type == "WORD" & subject == @sub & trial == @trial')
            stores = word_evs.store.unique()

            if cued_store not in stores:
                continue

            # locate the .ann file
            annotation_file = "{}-{}.ann".format(trial, cued_store)
            annotation_dir = os.path.join(data_dir, sub_dir, annotation_file)
            if not os.path.exists(annotation_dir):
                print("...Missing DD #{} {} Cued Recall Annotation".format(trial, cued_store))
                continue
            with open(annotation_dir, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if line[0] != "#":
                        recall_info = line.strip("\n").split("\t")

                        if len(recall_info) == 3:
                            rectime = float(recall_info[0]) + cued_time
                            item = recall_info[2].lower()
                            itemno = recall_info[1]

                            cued_recall_df = cued_recall_df.append({"subject":sub,
                                                                    "session":sess,
                                                                    "type": "CUED_REC_WORD", 
                                                                    "mstime": rectime, 
                                                                    "trial": trial, 
                                                                    "item": item, 
                                                                    "itemno": itemno, 
                                                                    "cued_item": cued_item, 
                                                                    "cued_store": cued_store,
                                                                    "recalled": item == cued_item
                                                                    }, ignore_index=True)

        # then, fill out the blanks
        cued_recall_df["store"] = cued_recall_df["store"].astype(str)
        cued_recall_df["store position"] = cued_recall_df["store position"].astype(str)

        for i, row in cued_recall_df.iterrows():
            word_evs = df[(df.trial == row.trial) & (df.type == "WORD")]

            row_item = row["item"]
            recall_word = word_evs.query('item == @row_item')

            if len(recall_word) != 0:        
                serialpos = recall_word["serialpos"].values[0]
                store = recall_word["store"].values[0]
                store_position = recall_word["store position"].values[0]

                cued_recall_df.at[i, "serialpos"] = serialpos
                cued_recall_df.at[i, "store"] = store
                cued_recall_df.at[i, "store position"] = store_position

            else:
                cued_recall_df.at[i, "serialpos"] = -999

        ###############################################################################################
        # FINAL STORE RECALL
        ###############################################################################################
        FFR_store_df = pd.DataFrame(columns = df.columns)
        FFR_store_evs = df.query('type == "final store recall recording start"')
        if len(FFR_store_evs) != 0:
            FFR_store_start = FFR_store_evs.mstime.values[0]

            # let's load appropriate .ann files
            annotation_dir = os.path.join(data_dir, sub_dir, "final store-0.ann")
            if not os.path.exists(annotation_dir):
                print("...Missing Final Store Recall Annotation")
            else:
                with open(annotation_dir, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        if line[0] != "#":
                            recall_info = line.strip("\n").split("\t")

                            if len(recall_info) == 3:
                                rectime = float(recall_info[0]) + FFR_store_start
                                item = recall_info[2].lower()
                                itemno = recall_info[1]

                                FFR_store_df = FFR_store_df.append({"subject":sub,
                                                                    "session":sess,
                                                                    "type":"FSR_REC_WORD", 
                                                                    "mstime":rectime, 
                                                                    "item":item, 
                                                                    "itemno":itemno, 
                                                                   }, ignore_index=True)

        ###############################################################################################
        # FINAL ITEM RECALL
        ###############################################################################################
        FFR_item_df = pd.DataFrame(columns = df.columns)
        FFR_item_evs = df.query('type == "final object recall recording start"')
        if len(FFR_item_evs) != 0:
            FFR_item_start = FFR_item_evs.mstime.values[0]

            # let's load appropriate .ann files
            annotation_dir = os.path.join(data_dir, sub_dir, "final free-0.ann")
            if not os.path.exists(annotation_dir):
                print("...Missing Final Free Recall Annotation")
            else:
                with open(annotation_dir, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        if line[0] != "#":
                            recall_info = line.strip("\n").split("\t")

                            if len(recall_info) == 3:
                                rectime = float(recall_info[0]) + FFR_item_start
                                item = recall_info[2].lower()
                                itemno = recall_info[1]

                                FFR_item_df = FFR_item_df.append({"subject":sub,
                                                                  "session":sess,
                                                                  "type":"FFR_REC_WORD", 
                                                                  "mstime":rectime, 
                                                                  "item":item, 
                                                                  "itemno":itemno
                                                                 }, ignore_index=True)

                # then, fill out the blanks
                FFR_item_df["store"] = FFR_item_df["store"].astype(str)
                FFR_item_df["store position"] = FFR_item_df["store position"].astype(str)

                for i, row in FFR_item_df.iterrows():
                    word_evs = df.query('type == "WORD"')

                    row_item = row["item"]
                    recall_word = word_evs.query('item == @row_item')

                    if len(recall_word) != 0:        
                        serialpos = recall_word["serialpos"].values[0]
                        store = recall_word["store"].values[0]
                        store_position = recall_word["store position"].values[0]

                        FFR_item_df.at[i, "serialpos"] = serialpos
                        FFR_item_df.at[i, "store"] = store
                        FFR_item_df.at[i, "store position"] = store_position

                    else:
                        FFR_item_df.at[i, "serialpos"] = -999

        ###############################################################################################
        # ARRANGE
        ###############################################################################################
        # now put it in right place
        tmp = df.copy()
        tmp = tmp.append(recall_df)
        tmp = tmp.append(cued_recall_df)
        tmp = tmp.append(FFR_store_df)
        tmp = tmp.append(FFR_item_df)
        tmp = tmp.sort_values('mstime').reset_index(drop=True)

        tmp[["trial", "serialpos", "recalled", "itemno", "intrusion"]] = tmp[["trial", "serialpos", "recalled", "itemno", "intrusion"]].fillna(-999)
        tmp = tmp.astype({"trial":int, "serialpos":int, "recalled":int, "itemno":int, "intrusion":int})
        
        tmp = add_itemno(tmp, wordpool)
        tmp = add_recalled(tmp)
        tmp = add_final_recalled(tmp)
        tmp = add_intrusion(tmp)
        
        ###############################################################################################
        # add EFR presses
        ###############################################################################################
        tmp["keypress"] = np.nan
        tmp["keypress_keycode"] = np.nan
        tmp = tmp.astype({"keypress":object, "keypress_keycode":object})

        tmp = add_keypress(tmp, "REC_WORD", "REC_STOP")
        tmp = add_keypress(tmp, "CUED_REC_WORD", "CUED_REC_STOP")
        tmp = add_keypress(tmp, "FSR_REC_WORD", "FSR_REC_STOP")
        tmp = add_keypress(tmp, "FFR_REC_WORD", "FFR_REC_STOP")
        
        ###############################################################################################
        # SAVE
        ###############################################################################################
        tmp.to_csv(file_dir)
        

main()
