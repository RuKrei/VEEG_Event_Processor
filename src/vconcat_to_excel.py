#!/usr/bin/env python

import pandas as pd
from utils import extract_ordered_groups
import os
import glob

def write_excel_table(vconcat=None):
    # vconcat holds source, description, time_from_onset
    # define a list to hold the row-names
    rows = []
    
    # define column - names (should be .edf-files)
    
    # fill rows with upper standard parameters
    for i in ["EEG-File", "Date", "Time", "Interpretation"]:
        rows.append(i)
    #print(rows)
    
    #search for all data in results-folder
    print(f"cwd --> {os.getcwd()}")
    
    all_folders = os.listdir(os.path.relpath("../results/"))
    #print(f"All folders --> {all_folders}")
    
    # exclude not needed files
    files = []
    for f in all_folders:
        if os.path.isdir(os.path.relpath("../results/" + f)):
            if f == "grand_average":
                pass
            else:
                all_data_name = "All_data_" + f + ".tsv"
                files.append(os.path.join("..", "results", f, "tables", all_data_name))              
    
    print(f"Working with tsvs --> {files}")
        
    for f in files:
        df = pd.read_csv(f, sep="\t")
        df['order'] = df.iloc[:,0]
        print(df.head())





    
    #eeg, semio, test = extract_ordered_groups(vconcat)
    #print(eeg)
    #print(semio)
    #print(test)



    
if __name__ == "__main__":
    vconcat = "../results/grand_average/tables/All_data_grand_average.tsv"
    write_excel_table(vconcat)