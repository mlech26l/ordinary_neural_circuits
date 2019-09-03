import numpy as np
import os

csv_files = sorted([f for f in os.listdir("logs_csv") if os.path.isfile(os.path.join("logs_csv", f)) and f.endswith('.log')])

tw = []
random = []

for f in csv_files:
    arr = np.loadtxt(os.path.join("logs_csv",f),delimiter=";")
    lst = tw
    if("csvlog_rnd" in f):
        lst = random

    result = arr[-1,1]
    lst.append(result)

print("TW circuit: {:0.2f} +- {:0.2f}".format(np.mean(tw),np.std(tw)))
print("Random    : {:0.2f} +- {:0.2f}".format(np.mean(random),np.std(random)))
