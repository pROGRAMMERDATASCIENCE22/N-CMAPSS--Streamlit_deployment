import pandas as pd


t_u = pd.read_csv('File_DevUnits_TestUnits.csv')
print(t_u.info())
print(t_u['Dev Units'].dtype)
#t_u['Dev Units'] = t_u['Dev Units'].astype(float)
t_u['D_U'] = t_u['Dev Units'].apply(lambda x:x[1:-1].split(',')).apply(lambda x:[float(i) for i in x])

x = 7
print(type(t_u['Dev Units'][0]))
print(t_u['Dev Units'][0])
if x in t_u['D_U']:
    print(t_u['File'])


# t_u['Dev Units'] = t_u.apply(lambda x: list([x['D_u 1'],
#                                         x['D_u 2'],
#                                         x['D_u 3'],
#                                         x['D_u 4'],
#                                         x['D_u 5'],
#                                         x['D_u 6'],
#                                         x['D_u 7'],
#                                         x['D_u 8'],
#                                         x['D_u 9']]),axis=1)

# x = 7.
# t_u_t = list(t_u.columns[10:])
# print(t_u_t)
# t_u['Test Units'] = [[e for e in row if e==e] for row in t_u[t_u_t].values.tolist()]
# #print(t_u)
# t_u = t_u.drop(columns = t_u[t_u_t])
# t_u_d = list(t_u.columns[1:10])
#
# t_u['Dev Units'] = [[e for e in row if e==e] for row in t_u[t_u_d].values.tolist()]
# print(t_u)
# t_u = t_u.drop(columns = t_u[t_u_d])
# print(t_u.info())
# columns_titles = ["File","Dev Units","Test Units"]
# t_u=t_u.reindex(columns=columns_titles)
# print(t_u)
# t_u.to_csv("File_DevUnits_TestUnits_f.csv")
#print(t_u)
# if x in t_u['Dev Units']:
#     print(t_u['File'])
