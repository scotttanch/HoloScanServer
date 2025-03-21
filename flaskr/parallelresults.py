import os
import pandas as pd
import matplotlib.pyplot as plt

# get results files
files = os.scandir()

legend = []

marker_line = {"Lenovo": ("^", "--"), "Pi 5": ("o", "-."), "VACC": ("x", ":")}

resolutions = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)',]
data_files = []
for file in files:
    if file.path.__contains__("ParallelResults"):
        data_files.append(file)

fig, axs = plt.subplots(2, 3)
axs = axs.reshape(-1)
fig.tight_layout()
for file in data_files:
    frame = pd.read_csv(file, skipfooter=1, engine='python')
    # for each frame I want to plot time (y) cores (x) for each resolution
    with open(file, 'r') as f:
        lines = f.readlines()
    sys_name, proc_info = lines[-1].split(',')

    for index, res in enumerate(resolutions):
        legend.append(proc_info+f" {res}")
        subframe = frame.loc[frame.resolution == res]
        t = []
        std = []
        cores = []
        t_0 = subframe.loc[subframe.cores == 1].time.mean()
        for core in subframe.cores.unique():

            t.append(subframe.loc[subframe.cores == core].time.mean()/t_0)
            std.append(subframe.loc[subframe.cores == core].time.std())
            cores.append(core)

        axs[index].plot(cores, t, marker=marker_line[sys_name][0], linestyle=marker_line[sys_name][1])
        axs[index].set_ylabel("Time (min)")
        axs[index].set_xlabel(labels[index])
        axs[index].set_title(f"Resolution: {res*100} cm")
        axs[index].legend(["Lenovo", "Pi 5", "VACC"])

plt.show()



# Demonstration of increased processing time for single core
resolutions = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
lenovo_t = [36.282763333333335, 8.86012999999999, 2.651513333333314, 0.44679666666668105, 0.22184833333331572, 0.11132833333331893]
pi_t = [61.603359593465065, 14.955855604471532, 4.500249354785046, 0.7620869520266813, 0.3805801828316665, 0.19146609531662154]
vacc_t = [15.230688599026957, 7.520051410583159, 3.774699691890079, 1.8577183686433514, 0.9228830658681302, 0.4648272364953301]
cells = [501750, 122250, 36465, 6138, 3036, 1518]

#for i in range(6):
#    lenovo_t[i] = lenovo_t[i]/0.11132833333331893
#    pi_t[i] = pi_t[i]/0.19146609531662154
#    vacc_t[i] = vacc_t[i]/0.4648272364953301
#    cells[i] = cells[i]/1518

#resolutions.reverse()
#lenovo_t.reverse()
#pi_t.reverse()
#vacc_t.reverse()
#cells.reverse()


print("Resolution \t #Cells \t VACC \t Lenovo \t Pi")
for i in range(len(resolutions)):
    print(resolutions[i], cells[i], vacc_t[i], lenovo_t[i], pi_t[i])


plt.loglog(cells, lenovo_t, marker='x')
plt.loglog(cells, pi_t, marker='o')
plt.loglog(cells, vacc_t, marker='^')
plt.xlabel("Number of Pixels")
plt.ylabel("Computational Time")
plt.legend(["Lenovo", "Pi 5", "VACC"])
plt.show()