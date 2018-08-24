import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
limit_rows   = 1700
limit_people = 1615
#read file
df0 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 0)
df1 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 1)
df2 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 2)
df3 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 3)
df4 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 4)
#
# unique_ids   = pd.Series(df1["Client"].unique())
# unique_id    = unique_ids.sample(n=limit_people)
# df1           = df1[df1.ncodpers.isin(unique_id)]
# df.describe()
# Change datatype

#
# df1["Age"]   = pd.to_numeric(df1["Age"], errors="coerce")
# df1["Tenure"]   = pd.to_numeric(df1["Tenure"], errors="coerce")
# df1.isnull().sum()
# pd.DataFrame(df2)
dic = {}
profile = df1.columns.values
act_bal = df2.columns.values
in_out = df3.columns.values
revenue =  df4.columns.values
#create dic
id,sex,age,tenure = df1[profile]
# act_bal = df2[act_bal]
# id,sex,age,tenure = df3[in_out]
# id,sex,age,tenure = df4[revenue]
print "start"
#create dic with ID
for i in range(df1[id].count()):
# for i in range(3):
    dic[df1[id][i]] = df1[sex][i],df1[age][i],df1[tenure][i]
print(dic)
for i in df2[act_bal][0]:
    dic[i]
    # for i, j in df2[act_bal][0], df3[in_out][0], df4[revenue][0]
    #     for i, j in df2[act_bal][0], df3[in_out][0], df4[revenue][0]
#
#
# for die in dat_info['die']:
#     fileDict[die]={}
#     for temper in dat_info['temper']:
#         fileDict[die][temper]={}
#         for mosType in dat_info['mosType']:
#             fileDict[die][temper][mosType]={}
#             for vt in dat_info['vt']:
#                 fileDict[die][temper][mosType][vt]={}
#                 for width in dat_info['width']:
#                     fileDict[die][temper][mosType][vt][width]={}
#                     for length in dat_info['length']:
#                         fileDict[die][temper][mosType][vt][width]
#
# array([u'Client', u'Count_CA', u'Count_SA', u'Count_MF', u'Count_OVD',
#        u'Count_CC', u'Count_CL', u'ActBal_CA', u'ActBal_SA', u'ActBal_MF',
#        u'ActBal_OVD', u'ActBal_CC', u'ActBal_CL'], dtype=object)
#
# dicts = {}
# keys = range(4)
# values = ["Hi", "I", "am", "John"]
# for i in keys:
#         dicts[i] = values[i]

