#!/usr/bin/env python
# Author: Rudi Kreidenhuber <Rudi.Kreidenhuber@gmail.com>
# License: BSD (3-clause)

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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

