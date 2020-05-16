# -*- coding: utf-8 -*-
import numpy as np

fread = open("../data/iris_string.txt","r")
clabel_list = []
for row in fread:
    row = row.replace("\n","")
    rowsp = row.split(",")
    clabel_list.append(rowsp[4])

clabel_list = list(np.unique(clabel_list))

fread.close();

fread = open("../data/iris_string.txt","r")
fwrite = open("../data/iris.txt","w")

for row in fread:
    row = row.replace("\n","")
    rowsp = row.split(",")
    line = ""
    for i in range(len(rowsp)-1):
        line = line + rowsp[i] + ","
    line = line + str(clabel_list.index(rowsp[-1])) + "\n"
    fwrite.write(line)
fwrite.close()
fread.close()
