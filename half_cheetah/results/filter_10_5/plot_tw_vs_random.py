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

def count_success(lst):
    succ = 0
    for r in lst:
        if(r > 2000):
            succ += 1
    return 100*float(succ)/len(lst)

print("TW circuit: {:0.2f} +- {:0.2f} ({:0.2f}% success rate)".format(np.mean(tw),np.std(tw),count_success(tw)))
print("Random    : {:0.2f} +- {:0.2f} ({:0.2f}% success rate)".format(np.mean(random),np.std(random),count_success(random)))

