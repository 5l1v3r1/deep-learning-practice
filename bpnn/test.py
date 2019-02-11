# import csv
# data_matrix_list = []
# with open("qiye.csv", "r") as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         row = [int(x) for x in row]
#         data_matrix_list.append(row)
# print(data_matrix_list)

a = list(range(10))
import random
random.shuffle(a)
print(a)