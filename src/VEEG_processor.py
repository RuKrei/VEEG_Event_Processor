#! /usr/bin/python
# Author: Rudi Kreidenhuber
# License: BSD (3-clause)


import plotly as py
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import os
from os.path import join
import glob
import mne
import re
from mne import Report
import pandas as pd
import numpy as np
from utils import (get_parent_dir, extract_lab_sec, raw_to_df, extract_ordered_groups, save_plotly_to_html,
                        create_results_folders, plot_interactive_subplot_with_table,
                        plot_interactive_tables, plot_interactive_eeg_and_semio, plot_interactive_eventcount,
                        plot_interactive_testing_results, plot_interactive_EEG_results, plot_interactive_semio_results,
                        win_create_results_folders, write_excel_table)

CONFIG_FILE = os.path.join(os.getcwd(), "VEEG_config.xlsx")
if os.path.isfile(CONFIG_FILE):
    print(f"Using configuration file: {CONFIG_FILE}")


class EdfGrabber:
    """Returns a list of .edf files in ../data/ directory"""
    def __init__(self, directory) -> None:
        self.directory = directory

    def grab_edfs(self):
        dir = os.path.join(self.directory, "*.edf")
        return glob.glob(dir)

class EdfToDataFrame:
    """Loads an .edf-file, determines seizure onset and saves all markers to a pandas DataFrame"""

    def __init__(self, edf) -> pd.DataFrame:
        self.edf = edf

    def _return_raw(self):
        return mne.io.read_raw(self.edf, preload = True)
    
    def _set_beginning(self, df):
        e_beginning = df[['e-beginn' in x for x in df['description'].str.lower()]]
        s_beginning = df[['s-beginn' in x for x in df['description'].str.lower()]]
        the_beginning = pd.concat([e_beginning, s_beginning], axis=0)
        if the_beginning.empty:
            print("Error: No marker containing \"Beginn\" found, cannot determine seizure onset for file: ", edf)
            print("Setting seizure onset to the beginning of the file")
            onset = "No seizure onset was marked"
            df.loc[-1] = [0, "_Beginn_(assumed)_"]
            df.index = df.index + 1
            df = df.sort_index()
            the_beginning.loc[1,:] = [0, "_Beginn-(assumed)_"]  
        samp_beginn = the_beginning.iloc[0,0].astype(float)
        onset = samp_beginn.astype(float)
        time_from_onset = df["onset"]
        time_from_onset = time_from_onset  - samp_beginn
        df["time_from_onset"] = time_from_onset
        return (df.drop(["onset"], axis = 1), onset)
    
    def _add_source_column(self, df):
        # Add source column to the left
        df["source"] = self.edf.split("/")[-1].split(".edf")[1]      # needs to be changed for windows still
        cols = list(df)
        cols.insert(0, cols.pop(cols.index('source')))
        return df.loc[:, cols]
        
    def raw_to_df(self):
        raw = self._return_raw()
        df = pd.DataFrame(raw.annotations)
        df = df.drop(["duration"], axis=1)
        df = df.drop(["orig_time"], axis=1)
        df, onset = self._set_beginning(df)
        df = self._add_source_column(df)
        return df, onset













def main():
    """Grab the .edf files"""
    Grabber = EdfGrabber(directory="../data")
    edfs = Grabber.grab_edfs()
    print(f"Found the following edfs:\n {edfs}\n\n")

    edf_framer = EdfToDataFrame(edfs[0])
    df = edf_framer.raw_to_df()
    print(df)

if __name__ == '__main__':
    main()




