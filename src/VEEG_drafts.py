#!/usr/bin/env python
# Author: Rudi Kreidenhuber <Rudi.Kreidenhuber@gmail.com>
# License: BSD (3-clause)



def plot_superimposed(df=None, eeg=None, semio=None, tmin= 0, tmax=0, 
                        testing=None, source=None):
    sep_lines = loc_of_sep_lines(df, graph_sep_line_width)
    fig_si = plt.Figure(figsize=(15.,2.), frameon=False)
    plt.title((str(source) + " - Events on common timeline"))
    if tmax == 0 and tmin == 0:
        x_axis = df["time_from_onset"]
    if tmax != 0 or tmin !=0:
        x_axis = np.linspace(tmin,tmax, (tmax - tmin))
    y_axis = np.ones_like(x_axis)
    #eeg
    times, labels = extract_lab_sec(eeg)
    plt.plot(x_axis, y_axis)
    plt.plot(x_axis[0],0.1)
    plt.plot(x_axis[0],1.9)
    sns.scatterplot(x=times, y=1)
    for idx in range(len(times)):
        label = (str(labels.iloc[idx]))
        coords = ((times.iloc[idx]), 1)
        if tmax == 0 and tmin == 0:
            xx = np.linspace(x_axis.iloc[0], (x_axis.iloc[-1]*0.9), len(times))
        else:
            xx = np.linspace(x_axis[0], (x_axis[-1]*0.9), len(times))
        yy = [1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 
                1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 
                1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 
                1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3, 1.9, 1.7, 1.5, 1.3] 
        xytext = ((times.iloc[idx]), yy[idx])
        plt.annotate(label, coords, xytext=xytext, color="k",
                        arrowprops=dict(facecolor='k', 
                                        arrowstyle='-'))
    for l in sep_lines:
        plt.axvline(x=l, color="lightblue", ls=":")
    # semiology
    times, labels = extract_lab_sec(semio)
    sns.scatterplot(x=times, y=1)
    for idx in range(len(times)):
        label = (str(labels.iloc[idx]))
        coords = ((times.iloc[idx]), 1)
        if tmax == 0:
            xx = np.linspace(x_axis.iloc[0], (x_axis.iloc[-1]*0.9), len(times))
        if tmax != 0 or tmin !=0:
            xx = np.linspace(x_axis[0], (x_axis[-1]*0.9), len(times))
        yy = [0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 
                0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 
                0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 
                0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7] 
        xytext = ((times.iloc[idx]), yy[idx])
        plt.annotate(label, coords, xytext=xytext, color="k",
                        arrowprops=dict(facecolor='k', 
                                        arrowstyle='-'))
    for l in sep_lines:
        plt.axvline(x=l, color="lightblue", ls=":")
    return fig_si


