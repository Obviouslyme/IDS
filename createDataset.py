import csv
    
cicids_lines = list(csv.reader(open('./data/thursday.csv')))
tpot_lines = list(csv.reader(open('./data/tpot.csv')))
tpot_lines_without_label = list()

for pos, value in enumerate(tpot_lines):
    if pos != 0:
        tpot_lines_without_label.append(value)

lines = cicids_lines+tpot_lines_without_label

for pos, value in enumerate(lines):
    if pos != 0:
        if lines[pos][78] != 'BENIGN':
            lines[pos][78] = 'MALICIOUS'

with open('./data/dataset.csv', 'w', newline='', encoding='utf-8') as dataset:
    writer = csv.writer(dataset)
    writer.writerows(lines)



