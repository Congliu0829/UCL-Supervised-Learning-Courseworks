import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Reload data for task 1.4
    with open('X_zipcombo.npy', 'rb') as f:
        X = np.load(f)
    with open('y_zipcombo.npy', 'rb') as f:
        y = np.load(f)
    with open('record_box.npy', 'rb') as f:
        record_box = np.load(f)

    label = np.argsort(abs(record_box)[:,0])[-5:]
    print('The index of five hardest label' + '\n')
    print(label)
    print('record box of these five labels' + '\n')
    print('[0] represents the wrong times, [1] represents the wrong confidence value' + '\n')
    print(record_box[label[:]])
    print('Five hardest label' + '\n')
    fig, axis = plt.subplots(1, 5, sharey = True, figsize = (15, 10))
    for i in range(len(label)):
        reshape_img = np.reshape(X[label[i]], (16, 16))
        axis[i].imshow(reshape_img, cmap='gray')
        axis[i].title.set_text(("Truth label = " + str(y[label[i]])))
    plt.savefig('Five hardest label', dpi=500)
    plt.show()