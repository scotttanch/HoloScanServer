import os
import pandas as pd
import matplotlib.pyplot as plt

# get results files
files = os.scandir()

fig, ax = plt.subplots(1, 1)
legend = []

marker_line = [("^", "--"), ("o", "-.")]


data_files = []
for file in files:
    if file.path.__contains__("ParallelResults"):
        data_files.append(file)

for file in data_files:
    frame = pd.read_csv(file, skipfooter=1, engine='python')
    vals = []
    stds = []
    for core_count in frame.cores.unique():
        vals.append(frame.loc[frame.cores == core_count].time.mean())
        stds.append(frame.loc[frame.cores == core_count].time.std())

    with open(file, 'r') as f:
        line = f.readlines()[-1]
    system_info = line.split(',')
    comp = system_info[0]
    proc = system_info[1]
    legend.append(proc)

    #ax.plot(list(frame.cores.unique()), vals, marker=marker_line[len(legend)-1][0], linestyle=marker_line[len(legend)-1][1])
    ax.errorbar(list(frame.cores.unique()), vals, yerr=stds,
                ecolor='black', barsabove=False, capsize=5,
                marker=marker_line[len(legend)-1][0], linestyle=marker_line[len(legend)-1][1])

ax.legend(legend)
ax.set_xlabel("Number of Cores")
ax.set_ylabel("Computation Time (min)")
ax.set_title("Comparison of Mean Back-Projection Time \n Over Multi-Processor Systems")
plt.show()
