import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from os.path import join
from mne import Report
import pandas as pd
import numpy as np

class AllHelpers:
    def __init__():
        pass
    
    def extract_lab_sec(df):
        times = df["time_from_onset"]
        labels = df["description"]
        return times, labels

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
        xlsx_file = os.path.join("..", "results", xlsx_file)
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

    def create_results_folders(edfs=None):
        for e in edfs:
            name = e.split(folder_splitter)[-1].split(".")[0]
            directory = os.path.join("..", "results", name)
            viz = os.path.join(directory, "viz")
            tables = os.path.join(directory, "tables")
            for d in [directory, viz, tables]:
                if not os.path.exists(d):
                    os.makedirs(d, exist_ok=True)
            if len(edfs) > 1:
                d = os.path.join("..", "results", "grand_average", "tables")
                os.makedirs(d, exist_ok=True)
                d = os.path.join("..", "results", "grand_average", "viz")
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
        save_dir = os.path.join("..", "results", source, "viz")
        save_name = os.path.join(save_dir, (source + "_interactive_viz.html"))
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
        fig.update_yaxes(categoryorder="category ascending")
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