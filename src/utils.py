#!/usr/bin/env python
# Author: Rudi Kreidenhuber <Rudi.Kreidenhuber@gmail.com>
# License: BSD (3-clause)

import os
import glob
import mne
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_parent_dir(d):
    return os.path.dirname(d)


def extract_lab_sec(df):
    times = df["time_from_onset"]
    labels = df["description"]
    return times, labels


def loc_of_sep_lines(df, width=10):
    first = df["time_from_onset"].values[0]
    last = df["time_from_onset"].values[-1]
    lines = list()
    x = 0
    while x > first:
        lines.append(x)
        x -= width
    x = width
    while x < last:
        lines.append(x)
        x += width
    return lines


def plot_seizure_horizontal(df=None, eeg=None, semio=None, tmin= 0, 
                                tmax=0, testing=None, source=None, name=None,
                                graph_sep_line_width=None):
    sep_lines = loc_of_sep_lines(df, graph_sep_line_width)
    fig, ax = plt.subplots(figsize=(15,12), sharex=True)
    plt.suptitle(str(source) + " - " + name)
    if tmax == 0 and tmin == 0:
        x_axis = df["time_from_onset"]
    if tmax != 0 or tmin !=0:
        x_axis = np.linspace(tmin,tmax, (tmax - tmin))
    y_axis = np.ones_like(x_axis)
    ax.set_yticks([])
    ax.set_axis_off()
    #eeg
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("EEG Events")
    ax1.set_yticks([])
    times, labels = extract_lab_sec(eeg)
    ax1.plot(0.1,0.1)
    ax1.plot(0.1,1.9)
    ax1.plot(x_axis, y_axis)
    sns.scatterplot(x=times, y=1, ax=ax1)
    for idx in range(len(times)):
        label = (str(labels.iloc[idx]))
        coords = ((times.iloc[idx]), 1)
        if tmax == 0:
            xx = np.linspace(x_axis.iloc[0], (x_axis.iloc[-1]*0.9), len(times))
        if tmax != 0 or tmin !=0:
            xx = np.linspace(x_axis[0], (x_axis[-1]*0.9), len(times))
        yy = 1.5 #np.linspace(1.05, 0.8, len(times))  
        xytext = (xx[idx], yy)
        ax1.annotate(label, coords, xytext=xytext, color="k",
                        arrowprops=dict(facecolor='k', 
                                        arrowstyle='-'))
    if tmin < 0 < tmax:
        plt.axvline(x=0, color="r") 
    for l in sep_lines:
        plt.axvline(x=l, color="lightblue", ls=":")
    #semio
    ax2 = fig.add_subplot(3,1,2)
    ax2.set_title("Semiology")
    ax2.set_yticks([])
    times, labels = extract_lab_sec(semio)
    ax2.plot(0.1,0.1)
    ax2.plot(0.1,1.9)
    ax2.plot(x_axis, y_axis)
    sns.scatterplot(x=times, y=1, ax=ax2)
    for idx in range(len(times)):
        label = (str(labels.iloc[idx]))
        coords = ((times.iloc[idx]), 1)
        if tmax == 0:
            xx = np.linspace(x_axis.iloc[0], (x_axis.iloc[-1]*0.9), len(times))
        if tmax != 0 or tmin !=0:
            xx = np.linspace(x_axis[0], (x_axis[-1]*0.9), len(times))
        yy = 1.5 #np.linspace(1.05, 0.8, len(times))  
        xytext = (xx[idx], yy)
        ax2.annotate(label, coords, xytext=xytext, color="k",
                        arrowprops=dict(facecolor='k', 
                                        arrowstyle='-'))
    if tmin < 0 < tmax:
        plt.axvline(x=0, color="r")
    for l in sep_lines:
        plt.axvline(x=l, color="lightblue", ls=":")
    #testing
    ax3 = fig.add_subplot(3,1,3)
    ax3.set_title("Testing")
    ax3.set_yticks([])
    times, labels = extract_lab_sec(testing)
    ax3.plot(0.1,0.1)
    ax3.plot(0.1,1.9)
    ax3.plot(x_axis, y_axis)
    sns.scatterplot(x=times, y=1, ax=ax3)
    for idx in range(len(times)):
        label = (str(labels.iloc[idx]))
        coords = ((times.iloc[idx]), 1)
        if tmax == 0:
            xx = np.linspace(x_axis.iloc[0], (x_axis.iloc[-1]*0.9), len(times))
        if tmax != 0 or tmin !=0:
            xx = np.linspace(x_axis[0], (x_axis[-1]*0.9), len(times))
        yy = [1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 
                1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 
                1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 
                1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 
                1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 
                1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7,
                1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 
                1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 
                1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7, 1.9, 1.7, 1.5, 1.3, 0.1, 0.3, 0.5, 0.7] 
        xytext = (xx[idx], yy[idx])
        ax3.annotate(label, coords, xytext=xytext, color="k",
                        arrowprops=dict(facecolor='k', 
                                        arrowstyle='-'))
    if tmin < 0 < tmax:
        plt.axvline(x=0, color="r")
    for l in sep_lines:
        plt.axvline(x=l, color="lightblue", ls=":")
    fig.subplots_adjust(hspace=0.8)
    return fig


def plot_seizure_vertical(df=None, eeg=None, semio=None, testing=None, 
                            tmin= 0, tmax=0, source=None, name=None,
                            graph_sep_line_width=None):
    fig, ax = plt.subplots(figsize=(15,25), sharey=True)
    plt.suptitle(str(source) + " - " + name)
        
    if tmax == 0 and tmin == 0:
        y_axis = df["time_from_onset"]
    if tmax != 0 or tmin !=0:
        y_axis = np.linspace(tmin,tmax,tmax)

    x_axis = np.ones_like(y_axis) + 1
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    #eeg
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("EEG Events")
    times, labels = extract_lab_sec(eeg)
    ax1.plot(x_axis, y_axis)
    sns.scatterplot(y=times, x=0.5, ax=ax1)
    for idx in range(len(times)):
        label = (str(labels.iloc[idx]))
        coords = (1, (times.iloc[idx])) 
        xytext = (0.53, (times.iloc[idx]))
        ax1.annotate(label, coords, xytext=xytext, color="k")
    if tmin < 0 < tmax:
        plt.axhline(y=0, color="r", ls="-")
    ax1.set_xticks([])
    sep_lines = loc_of_sep_lines(df, graph_sep_line_width)
    for l in sep_lines:
        plt.axhline(y=l, color="lightblue", ls=":")
    #semio
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title("Semiology")
    ax2.set_xticks([])
    #ax2.set_yticks([])
    times, labels = extract_lab_sec(semio)
    ax2.plot(x_axis, y_axis)
    sns.scatterplot(y=times, x=0.5, ax=ax2)
    for idx in range(len(times)):
        label = (str(labels.iloc[idx]))
        coords = (1, (times.iloc[idx]))
        xytext = (0.53, (times.iloc[idx]))
        ax2.annotate(label, coords, xytext=xytext, color="k")
    
    if tmin < 0 < tmax:
        plt.axhline(y=0, color="r", ls="-")
    sep_lines = loc_of_sep_lines(df, graph_sep_line_width)
    for l in sep_lines:
        plt.axhline(y=l, color="lightblue", ls=":")
    #testing
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title("Testing")
    ax3.set_xticks([])
    #ax3.set_yticks([])
    times, labels = extract_lab_sec(testing)
    ax3.plot(x_axis, y_axis)
    sns.scatterplot(y=times, x=0.5, ax=ax3)
    for idx in range(len(times)):
        label = (str(labels.iloc[idx]))
        coords = (1, (times.iloc[idx]))
        xytext = (0.53, (times.iloc[idx]))
        ax3.annotate(label, coords, xytext=xytext, color="k")
    if tmin < 0 < tmax:
        plt.axhline(y=0, color="r", ls="-")
    sep_lines = loc_of_sep_lines(df, graph_sep_line_width)
    for l in sep_lines:
        plt.axhline(y=l, color="lightblue", ls=":")
    fig.subplots_adjust(hspace=0.2)
    return fig


def raw_to_df(raw, edf=None):
    df = pd.DataFrame(raw.annotations)
    to_drop = ["duration"]
    df = df.drop(to_drop, axis=1)
    if "Beginn" in str(df["description"]):
        samp_beginn = df[df["description"].str.contains("Beginn")]["onset"]
        print("samp_beginn = ", samp_beginn)
        onset = samp_beginn.astype(int)
        if isinstance(samp_beginn, pd.core.series.Series):
            print(f"There are multiple markers for seizure onset in this file --> taking first one.")
            samp_beginn = samp_beginn.iloc[0].astype(int)
            onset = samp_beginn
    else:
        print("Error: No marker containing \"Beginn\" found, cannot determine seizure onset for file: ", edf)
        print("Setting seizure onset to the beginning of the file")
        samp_beginn = int(0)
        onset = "No seizure onset was marked"
    df["time_from_onset"] = df["onset"] - float(samp_beginn)
    df = df.drop(["orig_time"], axis=1)
    df["source"] = edf.split("/")[-1]
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('source')))
    df = df.loc[:, cols]
    return df, onset


def extract_groups(df, edf=None):
    e_events = df[df["description"].str.startswith("e-")]
    e_events["source"] = edf
    s_events = df[df["description"].str.startswith("s-")]
    s_events["source"] = edf
    # All other flags should be testing or ignored, so:
    t_events = df[~df["description"].str.startswith("s-")]
    t_events = t_events[~t_events["description"].str.startswith("e-")]
    t_events = t_events[~t_events["description"].str.startswith("i-")]
    t_events["source"] = edf
    return e_events, s_events, t_events


def extract_ordered_groups(df=None, source=None):
    df = df.drop_duplicates(subset=["description"], keep="first")
    e_events = df[df["description"].str.startswith("e-")]
    e_events["order_of_occurence"] = (np.arange(len(e_events.axes[0])) +1).astype(int)
    e_events["source"] = source.split("/")[-1]
    cols = list(e_events)
    cols.insert(0, cols.pop(cols.index('source')))
    e_events = e_events.loc[:, cols]
    s_events = df[df["description"].str.startswith("s-")]
    s_events["order_of_occurence"] = (np.arange(len(s_events.axes[0])) +1).astype(int)
    s_events["source"] = source.split("/")[-1]
    cols = list(s_events)
    cols.insert(0, cols.pop(cols.index('source')))
    s_events = s_events.loc[:, cols]
    t_events = df[~df["description"].str.startswith("s-")]
    t_events = t_events[~df["description"].str.startswith("e-")]
    t_events["order_of_occurence"] = (np.arange(len(t_events.axes[0])) +1).astype(int)
    t_events["source"] = source.split("/")[-1]
    cols = list(t_events)
    cols.insert(0, cols.pop(cols.index('source')))
    t_events = t_events.loc[:, cols]
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


def shrink_df_to_tmax(df=None, tmax=None, tmin=None):
    shrink_df = df[df["time_from_onset"] < tmax]
    shrink_df = shrink_df[shrink_df["time_from_onset"] > tmin]
    return shrink_df


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


def save_fig_to_disc(fig=None, source=None, name=None):
    source = source.split("/")[-1].split(".")[0]
    name = name + ".png"
    save_path = ("../results/" + source + "/viz/" + name)
    fig.savefig(save_path)


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


def plot_eventcounts(df=None, eeg=None, semio=None, source=None):
    fig, ax = plt.subplots(figsize=(15,26))
    plt.suptitle(str(source) + " - Event counts")
    #EEG
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title("EEG Events")
    if len(eeg['description'].value_counts()) > 0:
        sns.countplot(y="description", data=eeg, orient="h", ax=ax1, order = eeg['description'].value_counts().index)
    else:
        #plt.plot(x=1, y=1)
        ax1.set_title("No EEG events found in file...")
    #Semiology
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title("Semiology Events")
    if len(semio['description'].value_counts()) > 0:
        sns.countplot(y="description", data=semio, orient="h", ax=ax2, order = semio['description'].value_counts().index)
    else:
        #plt.plot(x=1, y=1)
        ax2.set_title("No Semiology events found in file...")
    ax.set_yticks([])
    ax.set_axis_off()
    fig.subplots_adjust(hspace=0.2)
    return fig


def save_plotly_to_html(fig=None, source=None):
    save_dir = "../results/" + source + "/viz/"
    save_name = save_dir + source + "_interactive_viz.html"
    fig.write_html(save_name)


def extract_parameters_from_raw(raw=None):
    highp = raw.info["highpass"]
    lowp = raw.info["lowpass"]
    sfreq = raw.info["sfreq"]
    aq = raw.info["meas_date"]
    channels = raw.info["ch_names"]
    nr_channels = raw.info["nchan"]
    return highp, lowp, sfreq, aq, channels, nr_channels


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