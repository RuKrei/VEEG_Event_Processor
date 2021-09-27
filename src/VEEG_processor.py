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
import platform
from utils import (get_parent_dir, extract_lab_sec, raw_to_df, extract_ordered_groups, save_plotly_to_html,
                        create_results_folders, plot_interactive_subplot_with_table,
                        plot_interactive_tables, plot_interactive_eeg_and_semio, plot_interactive_eventcount,
                        plot_interactive_testing_results, plot_interactive_EEG_results, plot_interactive_semio_results,
                        win_create_results_folders, write_excel_table)


# Configuration
win = True if platform.system().lower().startswith("win") else False
folder_splitter = "\\" if win else "/"
CONFIG_FILE = os.path.join(os.getcwd(), "VEEG_config.xlsx")
if not os.path.isfile(CONFIG_FILE):
    CONFIG_FILE = os.path.join(os.getcwd(), "src/VEEG_config.xlsx")
print("Using configuration file: ", CONFIG_FILE)

if not os.getcwd().endswith("src"):
    os.chdir("./src")
    print(f"Changed working directory to {os.getcwd()}")


class Grabber:
    """Returns a list of .edf files in ../data/ directory"""
    def __init__(self, directory) -> str:
        self.directory = directory

    def grab_edfs(self) -> list:
        pwd = os.getcwd()
        dir = os.path.join(pwd, self.directory, "*.edf")
        return glob.glob(dir)
    
    def grab_subject_name(self) -> str:
        return os.getcwd().split(folder_splitter)[-2].split("VEEG_Event_Processor-")[-1]
        

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
            print(f"Error: No marker containing \"Beginn\" found, cannot determine seizure onset for: {df}")
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
        df["source"] = self.edf.split(folder_splitter)[-1].split(".edf")[0]
        cols = list(df)
        cols.insert(0, cols.pop(cols.index('source')))
        return df.loc[:, cols], df["source"][0]
    
    def _read_config_file(self, config=CONFIG_FILE):
        mEEG = pd.read_excel(CONFIG_FILE, sheet_name="EEG")
        mEEG = mEEG[["mName", "mTranslation", "mSubstitution"]]
        mEEG.dropna(how="all", inplace=True)
        mEEG = mEEG.set_index("mName")

        mSemio = pd.read_excel(CONFIG_FILE, sheet_name="Semio")
        mSemio = mSemio[["mName", "mTranslation", "mSubstitution"]]
        mSemio.dropna(how="all", inplace=True)
        mSemio = mSemio.set_index("mName")

        mModifiers = pd.read_excel(CONFIG_FILE, sheet_name="Modifiers")
        mModifiers = mModifiers[["mName", "mTranslation", "mSubstitution"]]
        mModifiers.dropna(how="all", inplace=True)
        mModifiers = mModifiers.set_index("mName")

        mAnatomy = pd.read_excel(CONFIG_FILE, sheet_name="Anatomy")
        mAnatomy = mAnatomy[["mName", "mTranslation", "mSubstitution"]]
        mAnatomy.dropna(how="all", inplace=True)
        mAnatomy = mAnatomy.set_index("mName")

        return(mEEG, mSemio, mModifiers, mAnatomy)

    def _marker_to_text(self, string, substitute=True):
        """
        Splits the input string as needed
        Translates according to CONFIG_FILE
        returns:
          a string in human readable format
          type: EEG, Semio, Testing
          markers_code: e-"IAmTheBaseName"
        """
        mEEG, mSemio, mModifiers, mAnatomy = self._read_config_file()
        d = dict()
        readbable = str()
        # ignore the i- markers - not need to translate those
        if string.startswith("i-"):
            return "ignored"
        # the rest belongs to one of three groups
        elif string.startswith("e-"):
            d["type"] = "EEG"
        elif string.startswith("s-"):
            d["type"] = "Semiology"
        else:
            d["type"] = "Testing"
    
        # this returns a list of markers and modifiers
        rex = re.findall(r"[-|+]\w*", string)

        # First job is to define the base 
        try:
            # base comes first
            r = rex[0].strip("-")
            rr = rex[0]
            if r in mEEG.index:
                base = mEEG.loc[str(r)][0]
            else:
                base = str(r)
            # now we can drop it from the list
            rex.remove(rr)

        # This might not be a smart move :-(
        except Exception as e:
            print(f"Could not determine base: {e}, setting it to {string}")
            base = string
    
    
        # 2nd job: substitutions
        if substitute == True:
            for r in rex:
                #print(f"\n\nrex = {rex} \n\n")
                r = r.split("-")[-1].split("+")[-1] 
                if r in mEEG.index:
                    if mEEG.loc[str(r)][1] != None:
                        newitems = list()
                        try:
                            print(f"mEEG.loc[str(r)][1] --> {mEEG.loc[str(r)][1]}")
                            # split the substitution
                            subst = str(mEEG.loc[str(r)][1]).split("-")


                            #print(f"\n\nsubst = {subst} \n\n")


                            for s in subst:
                                if not s in rex:
                                    newitems.append(s)
                            for n in newitems:
                            #    if n == "" or n in rex or n in newitems:
                            #        pass
                            #    else:
                            #        rex.append(str("-" + n))
                                rex.append(str("-" + n))    
                            # delete r, as it has just been substituted
                            rex.remove(str("-" + r))
                        except Exception as e:
                            print(e)
                if r in mSemio.index:
                    pass
                if r in mModifiers.index:
                    pass
                if r in mAnatomy.index:
                    pass
        print(f"rex after substitution   -->   {rex}")      
        #print(f"rex without base: {rex}")
     #   define placeholder lists
        strEEG = []
        strSemio = []
        strAna = []
        strMod = []
        strNotRecognized = []
    
        # now we can go throug the modifiers etc.
        for r in rex:
            r = r.split("-")[-1] 
            r = r.split("+")[-1]
            r = r.strip("-")     
            if r in mEEG.index:
                strEEG.append(mEEG.loc[str(r)][0])
            elif r in mSemio.index:
                strSemio.append(mSemio.loc[str(r)][0])
            elif str("+" + r) in mModifiers.index:
                strMod.append(str(mModifiers.loc[str("+" + r)][0]))
            elif str(r) in mModifiers.index:
                strMod.append(str("with " + mModifiers.loc[str(r)][0]))
            elif r in mAnatomy.index:
                strAna.append(mAnatomy.loc[str(r)][0])
            else:
                strNotRecognized.append(r)

        # make sure output order is always the same + return 
        readable = ""
        if strEEG is not []:
            #strEEG = set(strEEG)
            for e in sorted(strEEG):
                readable += str(" " + e)
        if strSemio is not []:
            for m in sorted(strSemio):
                readable += str(" " + m)
        if strMod is not []:
            for m in sorted(strMod):
                readable += str(" " + m)
        if strAna is not []:
            for a in sorted(strAna):
                readable += str(" " + a)     
        if strNotRecognized is not []:
            for m in sorted(strNotRecognized):
                readable += str(" " + m)

        readable = base + " " + readable
        return readable
    
    def raw_to_df(self):
        raw = self._return_raw()
        df = pd.DataFrame(raw.annotations)
        df = df.drop(["duration"], axis=1)
        df = df.drop(["orig_time"], axis=1)
        df, onset = self._set_beginning(df)
        df, source = self._add_source_column(df)
#        for idx, val in enumerate(df["description"]):
#            df["description"][idx] = self._marker_to_text(val)
        print(f"source = {source}")
        return df, onset
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def translate_markers(self, df):
        """Takes a DataFrame as produced by raw_to_df and 
           changes Markers in Column description to human readable form.

        Args:
            df ([pandas.DataFrame]): Output of EdfToDataFrame.raw_to_df
        """

        pass



def main():
    #Grab the .edf files
    Grab = Grabber(directory="../data")
    edfs = Grab.grab_edfs()
    subj_name = Grab.grab_subject_name()
    if win:
        win_create_results_folders(edfs)
    else:
        create_results_folders(edfs)
    print(f"Subject/ Patient name is: {subj_name}")
    print(f"Found the following edfs:\n {edfs}\n\n")
    df = dict()             # all data
    e_events = dict()       # EEG events
    s_events = dict()       # Semiology events
    t_events = dict()       # Testing events

    for e in edfs:
        print(f"Now processing file: {e}")
        edf_framer = EdfToDataFrame(e)
        df[e], onset = edf_framer.raw_to_df()
        e_events[e], s_events[e], t_events[e] = extract_ordered_groups(df[e])


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
    write_excel_table(e_events, s_events, win=win) 


    # Plots/report for single seizures
    report_title = subj_name + " - Single seizure plots"
    report = Report(subject=subj_name, title=report_title)

    event_folders = glob.glob("../results/*")
    if win:
        event_folders = glob.glob("..\\results\\*")
    data = dict()
    EEG = dict()
    Semio = dict()
    Test = dict()
    interactive_plots = dict()

    for e in event_folders:
        source = e.split(folder_splitter)[-1].split(".")[0]
        sep = "\\" if win else "/"
        
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
            interactive_plots[source] = plot_interactive_subplot_with_table(data[source], EEG[source], 
                                                                        Semio[source], Test[source], title=source)
            save_name = join("..", "results", source, "viz", str(source + "_interactive_viz.html"))
            if not os.path.isfile(save_name):
                save_plotly_to_html(interactive_plots[source], source=source)
                cap = source + " VIZ --> seizure"
                report.add_htmls_to_section(interactive_plots[source].to_html(full_html=False), 
                                            section=source, captions=cap)


            # event counts (plot.ly)
            event_counts = plot_interactive_eeg_and_semio(eeg=EEG[source], semio=Semio[source], source=source)
            cap = source + " VIZ --> event_conuts"
            sec = source
            report.add_htmls_to_section(event_counts.to_html(full_html=False), section=sec, captions=cap)

            # Testing
            cap = source + " VIZ --> Testing results"
            testing_viz = plot_interactive_testing_results(t_events=Test[source], title=cap)
            report.add_htmls_to_section(testing_viz.to_html(full_html=False), section=sec, captions=cap)

    # Save all
    report_save_name = "../results/Single_seizures_report.html"
    if win:
        report_save_name = "..\\results\\Single_seizures_report.html"
    report.save(report_save_name, overwrite=True)


    # Plots/report for grand average

    ga_report_title = subj_name + " - All seizures"
    ga_report = Report(subject=subj_name, title=ga_report_title)

    source="grand_average"

    EEG["grand_average"], Semio["grand_average"], Test["grand_average"] = extract_ordered_groups(df=data["grand_average"])

    ga_fig = plot_interactive_subplot_with_table(df=data["grand_average"], eeg=EEG["grand_average"], 
                                                    semio=Semio["grand_average"], testing=Test["grand_average"], title=ga_report_title)

    save_name = join("..", "results", "grand_average", "viz", str("grand_average_interactive_viz.html"))
    if not os.path.isfile(save_name):
        save_plotly_to_html(ga_fig, source=source)
        cap = source + " VIZ --> All seizures"
        ga_report.add_htmls_to_section(ga_fig.to_html(full_html=False), 
                                    section=source, captions=cap)

    # event counts (plot.ly)
    event_counts = plot_interactive_eeg_and_semio(eeg=EEG[source], semio=Semio[source], source=source)
    cap = source + " VIZ --> All event_conuts"
    sec = source
    ga_report.add_htmls_to_section(event_counts.to_html(full_html=False), section=sec, captions=cap)
    # EEG
    cap = source + " VIZ --> All EEG results"
    print(EEG["grand_average"])
    eeg_viz = plot_interactive_EEG_results(e_events=EEG["grand_average"], title=cap)
    ga_report.add_htmls_to_section(eeg_viz.to_html(full_html=False), section=sec, captions=cap)
    # Semiology
    cap = source + " VIZ --> All Testing results"
    testing_viz = plot_interactive_testing_results(t_events=Test[source], title=cap)
    ga_report.add_htmls_to_section(testing_viz.to_html(full_html=False), section=sec, captions=cap)
    # Testing
    cap = source + " VIZ --> All Semiology results"
    semio_viz = plot_interactive_semio_results(s_events=Semio[source], title=cap)
    ga_report.add_htmls_to_section(semio_viz.to_html(full_html=False), section=sec, captions=cap)

    report_save_name = "../results/Grand_average_report.html"
    if win:
        report_save_name = "..\\results\\Grand_average_report.html"
    ga_report.save(report_save_name, overwrite=True)



    # Lazy grand average
    
    
    # to do --> also create a lazy_average DataFrame at the beginning to use 
    # here for another round of grand average visualization
    
    
    # load configuration from excel file:


"""
    testTags = ["e-BIRD-r-temp-ffluct",
                "e-ASD-FZ",
                "e-sw-FZ",
                "e-maf-l-par",
                "e-LPD+F-F7",
                "e-RDA-FZ-CZ",
                "e-oirda-l-temp-F7"]  #!!! --> left left temporal temporal

    for t in testTags:
        plain_text = list()
        trans = marker_to_text(t, substitute=True)
        trans = trans.split(" ")
        for t in trans:
            if t not in plain_text and not t == "":
                plain_text.append(t)
        plain_text = " ".join(plain_text)
        print(f"--> {plain_text}\n")
"""






if __name__ == '__main__':
    main()




