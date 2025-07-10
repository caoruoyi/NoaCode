import numpy as np
import pandas as pd

    
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

def get_item_id(item, wordpool):
    if item not in wordpool:
        return -1
    return wordpool.index(item)+1

def add_itemno(events, wordpool):
    events.loc[(events["type"] == 'WORD'), "itemno"] = \
    events.loc[(events["type"] == 'WORD'), "item"].apply(lambda x: get_item_id(x, wordpool))
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

    events.loc[events["type"] == 'REC_WORD', "intrusion"] = events[events["type"] == 'REC_WORD'].apply(check_list, axis=1)
    return events