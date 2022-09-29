import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Reload data for task 1.3
    with open('confusion_matrix.npy', 'rb') as f:
        confusion_mat = np.load(f)

    print('Mean value of confusion matrix (%)' + '\n')
    print(np.mean(confusion_mat,axis=0) * 100)
    print('Standard deviation of confusion matrix (%)"' + '\n')
    print(np.std(confusion_mat,axis=0) * 100)

    # Plot heat map
    truth_label = ["Truth label" + str(i) for i in range(10)]
    pred_label = ["Predict label" + str(i) for i in range(10)]

    h_map = np.mean(confusion_mat,axis=0) * 100

    fig, ax = plt.subplots()
    im = ax.imshow(h_map)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(truth_label)), labels=truth_label)
    ax.set_xticks(np.arange(len(pred_label)), labels=pred_label)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(truth_label)):
        for j in range(len(pred_label)):
            text = ax.text(j, i, round(h_map[i, j],2),
                        ha="center", va="center", color="w")

    ax.set_title("Heatmap of mean of confusion matrix")
    # fig.tight_layout()
    plt.savefig('1-3', dpi=800)
    plt.show()