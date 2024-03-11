import json
import os
import datetime

def add_list(l1, l2):
    return [l1[i]+l2[i] for i in range(len(l1))]

def divide_list(l1, c):
    return [i/c for i in l1]

def min_list(l1, l2):
    return [min(l1[i], l2[i]) for i in range(len(l1))]

# directory_name = 'rl5915_results'
directory_name = input()
paths = [directory_name + '/' + fname for fname in os.listdir(directory_name)]
results = []
start = datetime.datetime.strptime('0:00:00.000000', "%H:%M:%S.%f")
for path in paths:
    f = open(path)
    data = json.load(f)
    for k, v in data.items():
        time = datetime.datetime.strptime(v[1], "%H:%M:%S.%f") - start
        data[k][1] = time.total_seconds()
    results.append(data)

avg_result = {k: [0.0, 0.0, 0] for k, _ in results[0].items()}
best_result = results[0]
for k, v in avg_result.items():
    for j in range(len(results)):
        avg_result[k] = add_list(avg_result[k], results[j][k])
        best_result[k] = min_list(best_result[k], results[j][k])
    avg_result[k] = divide_list(avg_result[k], len(results))

# print(avg_result)
# print('------------------')
# print(best_result)

json_object = json.dumps(avg_result, indent=4)

with open(directory_name + "/average_result.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(best_result, indent=4)

with open(directory_name + "/best_result.json", "w") as outfile:
    outfile.write(json_object)
    
