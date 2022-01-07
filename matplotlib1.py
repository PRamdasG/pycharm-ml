from matplotlib import pyplot as plt
x = [6, 1, 3, 2, 5, 4, 7];
y = [1, 13, 14, 36, 75, 66, 77];
plt.plot(x,y,color = "green",label = "first")


x1 = [1, 2, 3, 4, 5, 6, 7];
y1 = [1, 12, 15, 34, 25, 66, 77];
plt.plot(x1,y1,color= "red",label = "second")

plt.legend()
plt.grid()
plt.show()
