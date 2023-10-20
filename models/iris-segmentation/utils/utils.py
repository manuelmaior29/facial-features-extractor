import matplotlib.pyplot as plt

def show_example_pair(ipt_image, tgt_image):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(ipt_image, cmap='gray')
    axs[1].imshow(tgt_image, cmap='gray')
    fig.show()
    plt.waitforbuttonpress(1)