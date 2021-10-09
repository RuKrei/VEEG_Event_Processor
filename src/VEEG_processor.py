# Author: Rudi Kreidenhuber (Rudi.Kreidenhuber@gmail.com)
# License: BSD (3-clause)


import plotly as py
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# Configuration
win = True if platform.system().lower().startswith("win") else False
folder_splitter = "\\" if win else "/"
CONFIG_FILE = "/app/data/VEEG_config.xlsx"
if not os.path.isfile(CONFIG_FILE):
    raise Exception("No VEEG_config.xlsx - file found")
print("Using configuration file: ", CONFIG_FILE)

if not os.getcwd().endswith("src"):
    os.chdir("./src")
    print(f"Changed working directory to {os.getcwd()}")


# Helper functions
def get_parent_dir(d):
    return os.path.dirname(d)

def extract_lab_sec(df):
    times = df["time_from_onset"]
    labels = df["description"]
    return times, labels

def raw_to_df(raw, edf=None):
    df = pd.DataFrame(raw.annotations)
    df = df.drop(["duration"], axis=1)
    df = df.drop(["orig_time"], axis=1)

    # Find/set Beginn-Marker:
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
    df = df.drop(["onset"], axis = 1)
    
    # Add source column to the left
    df["source"] = edf.split("/")[-1].split(".edf")[0]      # needs to be changed for windows still
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('source')))
    df = df.loc[:, cols]
    return df, onset

def extract_ordered_groups(df=None):
    #df = df.drop_duplicates(subset=["description"], keep="first")   # not doing this, as e- and s- events might reuccur!
    e_events = df[df["description"].str.startswith("e-")]
    e_events["order_of_occurence"] = (np.arange(len(e_events.axes[0])) +1).astype(int)
    s_events = df[df["description"].str.startswith("s-")]
    s_events["order_of_occurence"] = (np.arange(len(s_events.axes[0])) +1).astype(int)
    t_events = df[~df["description"].str.startswith("s-")]
    t_events = t_events[~df["description"].str.startswith("e-")]
    t_events["order_of_occurence"] = (np.arange(len(t_events.axes[0])) +1).astype(int)
    return e_events, s_events, t_events

def write_excel_table(e_events=None, s_events=None, win=False):
    xlsx_file = "All_data_grand_average.xlsx"
    if win:
        xlsx_file = "..\\results\\" + xlsx_file
    else:
        xlsx_file = "../results/" + xlsx_file
    writer = pd.ExcelWriter (xlsx_file)

    # EEG-Events
    i = 1
    left = ["EEG", "", "File:", "Pattern 1:", "Pattern 2:", "Pattern 3:", "Pattern 4:", 
                "Pattern 5:", "Pattern 6:", "Pattern 7:",
                "Pattern 8:", "Pattern 9:", "Pattern 10:", "..."]
    for e in e_events.keys():
        try:
            if e_events[e].empty:
                print(f"Empty EEG-List --> {e_events[e]}, omitting")
            else:
                df_e = pd.DataFrame(e_events[e], columns=["description"])
                _, file = os.path.split(e)
                df_e = df_e.rename(columns={"description": file.split(".edf")[0]})
                df_e.to_excel(writer, sheet_name="EEG_1", startcol=(i+1), startrow=2, header=True, index=False)
                i += 1
            left_df = pd.DataFrame(left)
            left_df.to_excel(writer, sheet_name="EEG_1", startcol=0, startrow=0, header=False, index=False)
        except Exception as e:
            print(f"Excel-File: Something went wrong trying to parse EEG-Events for {e}")
    
    # Semiology events - pattern = index
    for e in e_events.keys():
        _, file = os.path.split(e)
        try:
            if e_events[e].empty:
                print(f"Empty EEG-List --> {e_events[e]}, omitting")
            else:
                try:
                    # merge 2 dataframes
                    new_df = pd.DataFrame(e_events[e], columns=["description", "order_of_occurence"])
                    new_df = new_df.rename(columns={"order_of_occurence": file.split(".edf")[0]})
                    df_e = pd.merge(df_e, new_df, how="outer", on="description", suffixes=(" ", "  "))
                except Exception as ex:
                    # there is no dataframe to start with, so create one
                    print(ex)
                    df_e = pd.DataFrame(e_events[e], columns=["description", "order_of_occurence"])
                    df_e = df_e.rename(columns={"order_of_occurence": file.split(".edf")[0]})
                
                # write to file
                df_e.to_excel(writer, sheet_name="EEG_2", startcol=1, startrow=3, header=True, index=False)
            sem_left = ["EEG", ""]
            sem_left_df = pd.DataFrame(sem_left)
            sem_left_df.to_excel(writer, sheet_name="EEG_2", startcol=0, startrow=0, header=False, index=False)
            writer.save()
        except Exception as ex:
            print(f"Excel-File: Something went wrong trying to parse Semiology-Events for {s}:")
            print(ex)
    
    # Semiology-Events - list
    sem_left = ["Semiology", "", "File:", "Pattern 1:", "Pattern 2:", "Pattern 3:", "Pattern 4:", 
                "Pattern 5:", "Pattern 6:", "Pattern 7:",
                "Pattern 8:", "Pattern 9:", "Pattern 10:", "..."]
    i = 1
    for s in s_events.keys():
        try:
            if s_events[s].empty:
                print(f"Empty Semiology-List --> {s_events[s]}, omitting")
            else:
                df_s = pd.DataFrame(s_events[s], columns=["description"])
                _, file = os.path.split(s)
                df_s = df_s.rename(columns={"description": file.split(".edf")[0]})
                df_s.to_excel(writer, sheet_name="Semiology_1", startcol=(i+1), startrow=2, header=True, index=False)
                #writer.save()
                i += 1
            sem_left_df = pd.DataFrame(sem_left)
            sem_left_df.to_excel(writer, sheet_name="Semiology_1", startcol=0, startrow=0, header=False, index=False)
        except Exception as e:
            print(f"Excel-File: Something went wrong trying to parse Semiology-Events for {s}")

    # Semiology events - pattern = index
    for s in s_events.keys():
        _, file = os.path.split(s)
        try:
            if s_events[s].empty:
                print(f"Empty Semiology-List --> {s_events[s]}, omitting")
            else:
                try:
                    # merge 2 dataframes
                    new_df = pd.DataFrame(s_events[s], columns=["description", "order_of_occurence"])
                    new_df = new_df.rename(columns={"order_of_occurence": file.split(".edf")[0]})
                    df_s = pd.merge(df_s, new_df, how="outer", on="description", suffixes=(" ", "  "))
                except Exception as e:
                    # there is no dataframe to start with, so create one
                    print(e)
                    df_s = pd.DataFrame(s_events[s], columns=["description", "order_of_occurence"])
                    df_s = df_s.rename(columns={"order_of_occurence": file.split(".edf")[0]})
                
                # write to file
                df_s.to_excel(writer, sheet_name="Semiology_2", startcol=1, startrow=3, header=True, index=False)
            sem_left = ["Semiology", ""]
            sem_left_df = pd.DataFrame(sem_left)
            sem_left_df.to_excel(writer, sheet_name="Semiology_2", startcol=0, startrow=0, header=False, index=False)
            writer.save()
        except Exception as e:
            print(f"Excel-File: Something went wrong trying to parse Semiology-Events for {s}:")
            print(e)

def make_folders(e):
    cwd = os.getcwd()
    parent = get_parent_dir(cwd)
    results_dir = os.path.join(parent, "results")
    res_folder_name = str(e).split("/")[-1].split(".")[-2]
    res = os.path.join(results_dir, res_folder_name)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    if not os.path.isdir(res):
        os.mkdir(res)

def plot_interactive(df=None, eeg=None, semio=None, testing=None, source=None):
    title = source + " - Interactive Visualization"
    xaxis_title="Time in seconds (from seizure onset)"
    fig = go.Figure()
    # Add traces
    x_axis = df["time_from_onset"]
    y_axis = np.ones_like(x_axis)
    # eeg
    times, labels = extract_lab_sec(eeg)
    fig.add_trace(go.Scatter(x=times, y=y_axis,
                        mode='markers',
                        name='EEG',
                        text=labels))
    # semio
    times, labels = extract_lab_sec(semio)
    fig.add_trace(go.Scatter(x=times, y=y_axis,
                        mode='markers',
                        name='Semiology',
                        text=labels))
    # testing
    times, labels = extract_lab_sec(testing)
    fig.add_trace(go.Scatter(x=times, y=y_axis,
                        mode='markers',
                        name='Testing',
                        text=labels))

    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title="")
    fig.show()
    return fig

def create_results_folders(edfs=None):
    for e in edfs:
        name = e.split("/")[-1].split(".")[0]
        directory = "../results/" + name
        viz = directory + "/viz"
        tables = directory + "/tables"
        for d in [directory, viz, tables]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
        if len(edfs) > 1:
            d = ("../results/grand_average/tables")
            os.makedirs(d, exist_ok=True)
            d = ("../results/grand_average/viz")
            os.makedirs(d, exist_ok=True)

def win_create_results_folders(edfs=None):
    for e in edfs:
        name = e.split("\\")[-1].split(".")[0]
        directory = "..\\results\\" + name
        viz = directory + "\\viz"
        tables = directory + "\\tables"
        for d in [directory, viz, tables]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
        if len(edfs) > 1:
            d = ("..\\results\\grand_average\\tables")
            os.makedirs(d, exist_ok=True)
            d = ("..\\results\\grand_average\\viz")
            os.makedirs(d, exist_ok=True)

def plot_interactive_subplot_with_table(df=None, eeg=None, semio=None, testing=None, title=None):
    xaxis_title="Time in seconds (from seizure onset)"
    fig = make_subplots(rows=5, cols=1, shared_xaxes="all", 
                        specs=[[{"type": "table"}],
                                [{"type": "scatter"}],
                                [{"type": "scatter"}],
                                [{"type": "scatter"}],
                                [{"type": "scatter"}]],
                        subplot_titles=("Events", "EEG", "Semiology", "Testing", "All events"),
                        row_width=[0.1, 0.1, 0.1, 0.1, 0.8])
    # Add traces
    # data
    fig.add_trace(go.Table(
                        header=dict(
                                values=df.columns[2:], font=dict(size=10)),
                        cells=dict(
                            values=[df[i].tolist() for i in df.columns[2:]],
                            align="left")
                        ),
                        row=1, col=1)
    # scatter plots
    x_axis = df["time_from_onset"]
    y_axis = np.ones_like(x_axis)
    # eeg
    times, labels = extract_lab_sec(eeg)
    fig.add_trace(go.Scatter(x=times, y=y_axis,
                        mode='markers',                 #mode="markers+text"
                        hoverinfo="name+x+text",
                        name='EEG',
                        text=labels,
                        marker_symbol="diamond"), row=2, col=1)
    # semio
    times, labels = extract_lab_sec(semio)
    fig.add_trace(go.Scatter(x=times, y=y_axis,
                        mode='markers',
                        name='Semiology',
                        text=labels,
                        marker_symbol="x"), row=3, col=1)
    # testing
    times, labels = extract_lab_sec(testing)
    fig.add_trace(go.Scatter(x=times, y=y_axis,
                        mode='markers',
                        name='Testing',
                        text=labels,
                        marker_symbol="circle"), row=4, col=1)
    # grand average
    times, labels = extract_lab_sec(df)
    fig.add_trace(go.Scatter(x=times, y=y_axis,
                        mode='markers',
                        name='All events',
                        text=labels,
                        marker_symbol="hexagon2-open-dot"), row=5, col=1)

    fig.update_layout(title=title, yaxis_title="")
    fig.update_xaxes(rangeslider={"visible":True}, title={"text":xaxis_title}, row=5)
    fig.update_yaxes(visible=False, showticklabels=False)

    fig.update_layout(width=1500, height=1200)
    return fig

def save_plotly_to_html(fig=None, source=None):
    save_dir = "../results/" + source + "/viz/"
    save_name = save_dir + source + "_interactive_viz.html"
    fig.write_html(save_name)

"""
def extract_parameters_from_raw(raw=None):
    highp = raw.info["highpass"]
    lowp = raw.info["lowpass"]
    sfreq = raw.info["sfreq"]
    aq = raw.info["meas_date"]
    channels = raw.info["ch_names"]
    nr_channels = raw.info["nchan"]
    return highp, lowp, sfreq, aq, channels, nr_channels
"""

def plot_interactive_tables(ga_h=None, EEG_ga=None, semio_ga=None, test_ga=None):
    fig = make_subplots(rows=4, cols=1, 
                        specs=[[{"type": "table"}],
                                [{"type": "table"}],
                                [{"type": "table"}],
                                [{"type": "table"}]],
                        subplot_titles=("All data horizontal", "EEG grand average", "Semiology grand average", "Testing grand average"),
                        vertical_spacing=0.05

                        )
    # Add traces
    # All data (horizontal view)
    fig.add_trace(go.Table(
                        header=dict(
                                values=ga_h.columns[:], font=dict(size=10)),
                        cells=dict(
                            values=[ga_h[i].values.tolist() for i in ga_h.columns[:]],
                            align="left")
                        ),
                        row=1, col=1)

    # EEG grand average
    fig.add_trace(go.Table(
                        header=dict(
                                values=EEG_ga.columns[:], font=dict(size=10)),
                        cells=dict(
                            values=[EEG_ga[i].values.tolist() for i in EEG_ga.columns[:]],
                            align="left")
                        ),
                        row=2, col=1)    
    
    # Semiology grand average
    fig.add_trace(go.Table(
                        header=dict(
                                values=semio_ga.columns[:], font=dict(size=10)),
                        cells=dict(
                            values=[semio_ga[i].values.tolist() for i in semio_ga.columns[:]],
                            align="left")
                        ),
                        row=3, col=1)    

    # Testing grand average
    fig.add_trace(go.Table(
                        header=dict(
                                values=test_ga.columns[:], font=dict(size=10)),
                        cells=dict(
                            values=[test_ga[i].values.tolist() for i in test_ga.columns[:]],
                            align="left")
                        ),
                        row=4, col=1)    

    fig.update_layout(title="All data grand average", yaxis_title="")
    fig.update_layout(width=1250, height=2400)
    return fig

def plot_interactive_eeg_and_semio(eeg=None, semio=None, source=None):
    fig = make_subplots(rows=1, cols=2, start_cell="top-left",
                        subplot_titles=("EEG events", "Semiology events"),
                        #row_width=[0.1, 0.1, 0.1],
                        horizontal_spacing=0.2
                        )
    # EEG
    fig.add_trace(go.Histogram(y=eeg["description"], 
                        histfunc="count",
                        orientation="h",
                        name="EEG"),
                    row=1, col=1
                    )

    # Semio
    fig.add_trace(go.Histogram(y=semio["description"], 
                        histfunc="count",
                        orientation="h",
                        name="Semiology"),
                    row=1, col=2
                    )    


    fig.update_yaxes(categoryorder="total descending")
    fig.update_layout(width=1100, height=800, title=source,
                        xaxis_title="Number of occurences",
                        yaxis_title="")
    return fig

def plot_interactive_eventcount(df=None, mode=None, source=None):
    fig = go.Figure(
        data=[go.Histogram(y=df["description"], 
                            histfunc="count",
                            orientation="h")]
                    )
    fig.update_yaxes(categoryorder="total descending")
    fig.update_layout(title=(source + " - " + mode + " - Eventcount"),
                        xaxis_title="Number of occurences",
                        yaxis_title="")
    return fig

def plot_interactive_testing_results(t_events=None, title="Testing results"):
    t_events_failed = t_events[t_events["description"].apply(lambda x: x.endswith("0"))]
    t_events_failed["description"] = t_events_failed.description.str.split("0").str[0]
    t_events_passed = t_events[t_events["description"].apply(lambda x: x.endswith("1"))]
    t_events_passed["description"] = t_events_passed.description.str.split("1").str[0]
    fig = go.Figure()

    # passed
    fig.add_trace(go.Scatter(x=t_events_passed["time_from_onset"], 
                        y=t_events_passed["description"],
                        name="passed",
                        mode="markers",
                        hovertext=t_events_passed["source"])
                    )

    # failed
    fig.add_trace(go.Scatter(x=t_events_failed["time_from_onset"], 
                        y=t_events_failed["description"],
                        name="failed",
                        mode="markers",
                        hovertext=t_events_passed["source"])
                    )  

    fig.update_layout(width=1100, height=800, title=title,
                    xaxis_title="Time in seconds from onset",
                    yaxis_title="")
    return fig

def plot_interactive_EEG_results(e_events=None, title="EEG results"):
    fig = px.scatter(e_events, y=e_events["description"], x=e_events["time_from_onset"],
                        color=e_events["source"])
    fig.update_layout(width=1100, height=800, title=title, xaxis_title="Time in seconds from onset")
    return fig

def plot_interactive_semio_results(s_events=None, title="Semiology results"):
    fig = px.scatter(s_events, y=s_events["description"], x=s_events["time_from_onset"],
                        color=s_events["source"])
    fig.update_layout(width=1100, height=800, title=title, xaxis_title="Time in seconds from onset")
    return fig

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
                            for s in subst:
                                if not s in rex:
                                    newitems.append(s)
                            for n in newitems:
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

        # bring back the prefix
        if string.startswith("e-"):
            prefix = "e-"
        elif string.startswith("s-"):
            prefix = "s-"
        else:
            prefix = ""

        readable = prefix + base + " " + readable
        if readable.startswith(" "):
            readable.lstrip(" ")
        return readable
    
    def raw_to_df(self):
        raw = self._return_raw()
        df = pd.DataFrame(raw.annotations)
        df = df.drop(["duration"], axis=1)
        df = df.drop(["orig_time"], axis=1)
        df, onset = self._set_beginning(df)
        df, source = self._add_source_column(df)
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
    Grab = Grabber(directory="/app/data")
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
        if e.endswith(".csv") or e.endswith(".xlsx"):
            pass
        else:
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
    def make_grand_average_report(df, name="grand_average"):
        ga_report_title = subj_name + " - All seizures"
        ga_report = Report(subject=subj_name, title=ga_report_title)
        EEG[name], Semio[name], Test[name] = extract_ordered_groups(df=df)
        
        # grand_average figure
        ga_fig = plot_interactive_subplot_with_table(df=df, eeg=EEG[name], 
                                                    semio=Semio[name], testing=Test[name], title=ga_report_title)
        cap = name + " VIZ --> All seizures"
        ga_report.add_htmls_to_section(ga_fig.to_html(full_html=False), 
                                    section=name, captions=cap)

        # EEG
        cap = name + " VIZ --> All EEG results"
        eeg_viz = plot_interactive_EEG_results(e_events=EEG[name], title=cap)
        ga_report.add_htmls_to_section(eeg_viz.to_html(full_html=False), section=name, captions=cap)
        
        # Testing
        if name == "grand_average":          # Testing-Markers are not renamed, no point in visualizing them twice
            cap = name + " VIZ --> All Testing results"
            testing_viz = plot_interactive_testing_results(t_events=Test[name], title=cap)
            ga_report.add_htmls_to_section(testing_viz.to_html(full_html=False), section=name, captions=cap)
        
        # Semiology
        cap = name + " VIZ --> All Semiology results"
        semio_viz = plot_interactive_semio_results(s_events=Semio[name], title=cap)
        ga_report.add_htmls_to_section(semio_viz.to_html(full_html=False), section=name, captions=cap)

        return ga_report

    # Grand average report - original markers
    ga_report = make_grand_average_report(df=data["grand_average"], name="grand_average")
    report_save_name = "../results/Grand_average_report.html"
    if win:
        report_save_name = "..\\results\\Grand_average_report.html"
    ga_report.save(report_save_name, overwrite=True)
    base_dir = os.path.join ("..", "results")
    data["grand_average"].to_csv(os.path.join(base_dir, "Data_grand_average.tsv"), sep="\t")

    # Lazy grand average report  
    lazy_df = data["grand_average"].copy()
    for idx, val in enumerate(lazy_df["description"]):
        lazy_df["description"][idx] = edf_framer._marker_to_text(val)
    base_dir = os.path.join ("..", "results")
    lazy_df.to_csv(os.path.join(base_dir, "Lazy_grand_average.tsv"), sep="\t")  
    lazy_ga_report = make_grand_average_report(df=lazy_df, name="readable_grand_average")
    report_save_name = "../results/Readable_grand_average_report.html"
    if win:
        report_save_name = "..\\results\\Readable_grand_average_report.html"
    lazy_ga_report.save(report_save_name, overwrite=True)
    
    # Copy config file used to results directory
    copyfile("/app/data/VEEG_config.xlsx", "/app/results/VEEG_config.xlsx")

if __name__ == '__main__':
    main()
