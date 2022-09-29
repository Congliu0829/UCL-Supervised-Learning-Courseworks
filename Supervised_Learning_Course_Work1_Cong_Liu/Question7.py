from Question6 import gen_error_list
from Question6 import gen_data
import matplotlib.pyplot as plt
# calculate the averaged generalization error for each k and plot the curve
k_list = [i for i in range(1, 50)]
X_h, y_h = gen_data(100)
error_list = gen_error_list(X_h, y_h, 4000, k_list)
# plot and save the figure
plt.plot(k_list, error_list)
plt.xlabel("K")
plt.ylabel("estimated generalization error")
plt.grid()
plt.savefig("2-7", dpi=500)
plt.show()
