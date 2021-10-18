# Author: Rudi Kreidenhuber
# License: BSD (3-clause)

import os
from os.path import join
import glob
import mne
import re
from mne import Report
import pandas as pd
import numpy as np
import platform
from shutil import copyfile
import AllHelpers as h
import EdfToDataFrame
import Grabber


# Configuration
ipynb = False



def main():
    win = True if platform.system().lower().startswith("win") else False
    folder_splitter = "\\" if win else "/"
    CONFIG_FILE = "/app/data/VEEG_config.xlsx"
    if not os.path.isfile(CONFIG_FILE):
        CONFIG_FILE = "../data/VEEG_config.xlsx"
    if not os.path.isfile(CONFIG_FILE):
        raise Exception("No VEEG_config.xlsx - file found")
    print("Using configuration file: ", CONFIG_FILE)

if not os.getcwd().endswith("src"):
    os.chdir(os.path.join(".", folder_splitter, "src"))
    print(f"Changed working directory to {os.getcwd()}")
    Grab = Grabber(directory="/app/data")
    edfs = Grab.grab_edfs()
    subj_name = Grab.grab_subject_name()
    h.create_results_folders(edfs)
    print(f"Subject/ Patient name is: {subj_name}")
    print(f"Found the following edfs:\n {edfs}\n\n")
    df = dict()             # all data
    e_events = dict()       # EEG events
    s_events = dict()       # Semiology events
    t_events = dict()       # Testing events

    for e in edfs:
        print(f"Now processing file: {e}")
        edf_framer = EdfToDataFrame(e, CONFIG_FILE)
        df[e], onset = edf_framer.raw_to_df()
        e_events[e], s_events[e], t_events[e] = h.extract_ordered_groups(df[e])

    #save
        csv_path = os.path.join("..", "results", e.split(folder_splitter)[-1].split(".")[0], "tables")
        e_file = e.split(folder_splitter)[-1].split(".")[0]
        tsv_name = "All_data_" + e_file + ".tsv"
        fname = os.path.join(csv_path, tsv_name)
        df[e].to_csv(fname, sep="\t")
        tsv_name = "EEG_data_" + e_file + ".tsv"
        fname = os.path.join(csv_path, tsv_name)
        e_events[e].to_csv(fname, sep="\t")
        tsv_name = "Semiology_data_" + e_file + ".tsv"
        fname = os.path.join(csv_path, tsv_name)
        s_events[e].to_csv(fname, sep="\t")
        tsv_name = "Testing_data_" + e_file + ".tsv"
        fname = os.path.join(csv_path, tsv_name)
        t_events[e].to_csv(fname, sep="\t")    
    
        for idx, val in enumerate(df.keys()):
            if idx == 0:
                # all data vertical
                vconcat = df[val]
                # all data horizontal
                concat = df[val]
                source = "source_" + str(idx)
                concat[source] = val
                cols = list(concat)
                cols.insert(0, cols.pop(cols.index(source)))
                concat = concat.loc[:, cols]
                concat = concat.sort_values(by=["time_from_onset"])
                if "source" in concat.keys():
                    concat.drop(columns=["source"], axis=1, inplace=True)
                concat["order_of_occurence"] = (1 + np.arange(len(concat.loc[:,"time_from_onset"])))
                # eeg, semio
                eeg_ga, semio_ga, test_ga = e_events[val], s_events[val], t_events[val]  # should be same keys as for e in edfs...
    
            if idx > 0:
                # all data vertical
                vnew_df = df[val]
                vconcat = pd.concat([vconcat, vnew_df], axis=0)
                # all data horizontal
                new_df = df[val]
                source = "source_" + str(idx)
                new_df[source] = val
                cols = list(new_df)
                cols.insert(0, cols.pop(cols.index(source)))
                new_df = new_df.loc[:, cols]
                if "source" in new_df.keys():
                    new_df.drop(columns=["source"], axis=1, inplace=True)
                new_df["order_of_occurence"] = (1 + np.arange(len(new_df.loc[:,"time_from_onset"]))).astype(int)
                concat = pd.merge(concat, new_df, how="outer", on="description", suffixes=(" ", "  "))
                # eeg, semio
                ne, ns, nt = e_events[val], s_events[val], t_events[val]
                eeg_ga = pd.merge(eeg_ga, ne, how="outer", on="description", suffixes=(" ", "  ")) 
                semio_ga = pd.merge(semio_ga, ns, how="outer", on="description", suffixes=(" ", "  "))
                test_ga = pd.merge(test_ga, nt, how="outer", on="description", suffixes=(" ", "  "))
    
            idx += 1
    
        if "source_0" in vconcat.keys():
            vconcat.drop(columns=["source_0"], axis=1, inplace=True)

    # save grand averages
    base_dir = os.path.join ("..", "results", "grand_average", "tables")
    eeg_ga.to_csv(os.path.join(base_dir, "EEG_data_grand_average.tsv"), sep="\t")
    semio_ga.to_csv(os.path.join(base_dir, "Semiology_data_grand_average.tsv"), sep="\t")
    test_ga.to_csv(os.path.join(base_dir, "Testing_data_grand_average.tsv"), sep="\t")
    concat.to_csv(os.path.join(base_dir, "All_data_grand_average_horizontal.tsv"), sep="\t")
    vconcat.to_csv(os.path.join(base_dir, "All_data_grand_average.tsv"), sep="\t")
    
    # write excel file
    h.write_excel_table(e_events, s_events, win=win) 

    # Plots/report for single seizures
    report_title = subj_name + " - Single seizure plots"
    report = Report(subject=subj_name, title=report_title)
    event_search = os.path.join("..", "results", "*")
    event_folders = glob.glob(event_search)
    data = dict()
    EEG = dict()
    Semio = dict()
    Test = dict()
    interactive_plots = dict()

    for e in event_folders:
        if e.endswith(".csv") or e.endswith(".xlsx"):
            pass
        else:
            source = e.split(folder_splitter)[-1].split(".")[0]
            sep = folder_splitter
            
            tsv_path = join(e, "tables")
            tsv_name = "All_data_" + source + ".tsv"
            tsv = os.path.join(tsv_path, tsv_name)
            data[source] = pd.read_csv(tsv, sep="\t")
            tsv_name = "EEG_data_" + source + ".tsv"
            tsv = os.path.join(tsv_path, tsv_name)
            EEG[source] = pd.read_csv(tsv, sep="\t")    
            tsv_name = "Semiology_data_" + source + ".tsv"
            tsv = os.path.join(tsv_path, tsv_name)
            Semio[source] = pd.read_csv(tsv, sep="\t")
            tsv_name = "Testing_data_" + source + ".tsv"
            tsv = os.path.join(tsv_path, tsv_name)
            Test[source] = pd.read_csv(tsv, sep="\t")
    
            if source == "grand_average":
                pass
            else:
                interactive_plots[source] = h.plot_interactive_subplot_with_table(data[source], EEG[source], 
                                                                            Semio[source], Test[source], title=source)
                save_name = join("..", "results", source, "viz", str(source + "_interactive_viz.html"))
                if not os.path.isfile(save_name):
                    h.save_plotly_to_html(interactive_plots[source], source=source)
                    cap = source + " VIZ --> seizure"
                    report.add_htmls_to_section(interactive_plots[source].to_html(full_html=False), 
                                                section=source, captions=cap)
    
    
                # event counts (plot.ly)
                event_counts = h.plot_interactive_eeg_and_semio(eeg=EEG[source], semio=Semio[source], source=source)
                cap = source + " VIZ --> event_conuts"
                sec = source
                report.add_htmls_to_section(event_counts.to_html(full_html=False), section=sec, captions=cap)
    
                # Testing
                cap = source + " VIZ --> Testing results"
                testing_viz = h.plot_interactive_testing_results(t_events=Test[source], title=cap)
                report.add_htmls_to_section(testing_viz.to_html(full_html=False), section=sec, captions=cap)

    # Save all
    report_save_name = os.path.join("..", "results", "Single_seizures_report.html")
    report.save(report_save_name, overwrite=True)

    # Plots/report for grand average
    # Grand average report - original markers
    ga_report = h.make_grand_average_report(df=data["grand_average"], name="grand_average")
    report_save_name = os.path.join("..", "results", "Grand_average_report.html")
    ga_report.save(report_save_name, overwrite=True)
    base_dir = os.path.join ("..", "results")
    data["grand_average"].to_csv(os.path.join(base_dir, "Data_grand_average.tsv"), sep="\t")

    # Lazy grand average report  
    lazy_df = data["grand_average"].copy()
    for idx, val in enumerate(lazy_df["description"]):
        lazy_df["description"][idx] = edf_framer._marker_to_text(val)
    base_dir = os.path.join ("..", "results")
    lazy_df.to_csv(os.path.join(base_dir, "Lazy_grand_average.tsv"), sep="\t")  
    lazy_ga_report = h.make_grand_average_report(df=lazy_df, name="readable_grand_average")
    report_save_name = os.path.join("..", "results", "Readable_grand_average_report.html")
    lazy_ga_report.save(report_save_name, overwrite=True)
    
    # Copy config file used to results directory
    if not ipynb:
        copyfile("/app/data/VEEG_config.xlsx", "/app/results/VEEG_config.xlsx")

if __name__ == '__main__':
    main()
