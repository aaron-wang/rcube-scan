

bob = [((302, 209), 'yellow'), ((231, 209), 'yellow'), ((372, 211), 'yellow'), ((299, 272), 'blue'), ((229, 275), 'orange'), ((375, 276), 'green'), ((378, 345), 'white'), ((303, 347), 'white'), ((227, 347), 'white')]


bob = [((302, 209), 'yellow'), ((231, 211), 'yellow'), ((372, 209),'yellow')]

bob.sort(key=lambda x: x[0][1])

print(bob)

bob[0:3].sort(key=lambda x: x[0][0])
# for i in range(3):
#     bob[i*3:i*3+3].sort(key=lambda x: x[0][0])
#     print(bob[3*i:i*3+3])
#     print(3*i,i*3 + 3)
print(bob)