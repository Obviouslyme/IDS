import csv
import pandas as pd

with open('./data/thursday.csv', 'r', newline='', encoding='utf-8') as cicids:
    cicids_lines = list(csv.reader(cicids, delimiter=','))

with open('./data/tpot.csv', 'r', newline='', encoding='utf-8') as tpot:
    tpot_lines = list(csv.reader(tpot, delimiter=','))

lines = cicids_lines+tpot_lines[1:len(tpot_lines)]

for pos, value in enumerate(lines):
    if pos != 0:
        if lines[pos][78] != 'BENIGN':
            lines[pos][78] = 'MALICIOUS'

with open('./data/dataset.csv', 'w', newline='', encoding='utf-8') as dataset:
    writer = csv.writer(dataset)
    writer.writerows(lines)
