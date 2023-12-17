import matplotlib.pyplot as plt

X = [25, 50, 75, 100]
ep = [38.0, 43.0, 56.0, 67.0]
local = [30.12, 57.75, 60.67, 61.33]

plt.plot(X, ep, '-o',label="Hetero_NN")
plt.plot(X, local,'-o', label="Local")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
