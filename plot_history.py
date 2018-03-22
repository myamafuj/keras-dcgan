import os
import json

import matplotlib.pyplot as plt


def main():
    path = f'var/log/dcgan_cat_face_mod_history.json'
    with open(path, 'r', encoding='UTF-8') as f:
        history = json.load(f)

    x = range(len(history[0]))
    plt.figure(figsize=(6, 4))
    plt.subplot(211)
    plt.plot(x, history[0], 'b-', label='d_loss', linewidth=.5)
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(x, history[1], 'r-', label='g_loss', linewidth=.5)
    plt.legend(loc='upper right')
    plt.subplots_adjust(left=.1, right=.95, bottom=.1, top=.95)
    path = f'figure/dcgan_cat_face_mod_history.png'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    main()
