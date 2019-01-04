# -*- coding:utf-8 -*-

import pandas as pd

stu_dic = {'Age': [14, 13, 13, 14, 14, 12, 12, 15, 13, 12, 11, 14, 12, 15, 16, 12, 15, 11, 15],
           'Height': [69, 56.5, 65.3, 62.8, 63.5, 57.3, 59.8, 62.5, 62.5, 59, 51.3, 64.3, 56.3, 66.5, 72, 64.8, 67,
                      57.5, 66.5],
           'Name': ['Alfred', 'Alice', 'Barbara', 'Carol', 'Henry', 'James', 'Jane', 'Janet', 'Jeffrey', 'John',
                    'Joyce', 'Judy', 'Louise', 'Marry', 'Philip', 'Robert', 'Ronald', 'Thomas', 'Willam'],
           'Sex': ['M', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M'],
           'Weight': [112.5, 84, 98, 102.5, 102.5, 83, 84.5, 112.5, 84, 99.5, 50.5, 90, 77, 112, 150, 128, 133, 85,
                      112]}
student = pd.DataFrame(stu_dic)
print(student)
print(student.head())
print(student.tail())
# 查询指定的行
print(student.loc[[0, 2, 4, 5, 7]])
# 查询指定列如果多个列的话，必须使用双重中括号
print(student[['Name','Height','Weight']].head())
# 查询出所有12岁以上的女生信息
print(student[(student['Sex']=='F') & (student['Age']>12)][['Name','Height','Weight']])
print(student.loc[0]['Name'])
