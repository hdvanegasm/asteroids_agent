import pandas
import numpy

data = numpy.array([['Col1','Col2'], [1, 2], [3, 4]])

dataframe = pandas.DataFrame(columns=data[0, 0:], data=data[1:, 0:])

print(dataframe)

def me(a):
    a.append(1)

a = []
me(a)
print(a)

