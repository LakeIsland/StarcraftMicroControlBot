import random
import math
import datetime
import re

def clamp(x, x_min, x_max):
    return x_min if x < x_min else (x_max if x > x_max else x)

def argmax(state_action):
    max_index_list = []
    max_value = state_action[0]
    for index, value in enumerate(state_action):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
    return random.choice(max_index_list)

def importTable(fileName):
    table = []
    with open(fileName) as f:
        for line in f:
            line_splited = re.split(' |\t|\n',line)
            table.append([float(line_splited[0]), float(line_splited[1])])
    return table

def exportTable(table, algorithm, iterate, fileName = ''):
    now=datetime.datetime.now()
    nowDate = now.strftime('%Y_%m_%d_%H_%M')
    if(fileName is ''):
        fileName = "../tableData/%s_%dtimes_%s.txt" %(algorithm, iterate, nowDate)

    with open(fileName, mode='w') as f:
        for i in range(len(table)):
            for j in range(len(table[0])):
                if j is len(table[0])-1:
                    f.write("%f\n" % (table[i][j]))
                else:
                    f.write("%f\t" % (table[i][j]))

    print("export finished")