import numpy as np
from matplotlib import pyplot as plt
import random
import cv2 as cv
import csv


def make_parabola(a, b, c, x_margin, y_margin, i, parabola_dir, intercept_dir):
    '''
    y=ax^2 + bx + c
    The graph is shifted randomly according to x_margin and y_margin
    This could be better; it's hard to know how the outputs will be distributed
    '''
    x = np.linspace(-6, 6, 1000)
    y = a*x**2 + b*x + c

    fig = plt.figure(figsize=(1.6, 1.2))
    axes = fig.add_subplot(111)
    parabola, = axes.plot(x, y, c='k', linewidth=.8)
    # center the axes on the origin
    axes.spines['left'].set_position(('data', 0))
    axes.spines['bottom'].set_position(('data', 0))
    # set a random window which includes the y-intercept
    plt.xlim([-1*np.random.uniform(.1, x_margin), np.random.uniform(.1, x_margin)])
    plt.ylim([min(0, c) - np.random.uniform(.1, y_margin), max(0, c) + np.random.uniform(.1, y_margin)])
    for edge_i in ['top', 'right']:
        axes.spines[edge_i].set_edgecolor("white")
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False)
    axes.xaxis.set_ticklabels([])
    axes.yaxis.set_ticklabels([])
    plt.savefig(parabola_dir + '/parabola_{}'.format(i) + '.png')
    parabola.remove()
    axes.axis('off')
    axes.plot(0, c, 'k.', markersize=4)
    plt.savefig(intercept_dir + '/parabola_{}'.format(i) + '.png')
    plt.close()
    # does some image preprocessing to make the probability sum to one
    # this could be better
    label_img = cv.imread(intercept_dir + '/parabola_{}'.format(i) + '.png', cv.IMREAD_GRAYSCALE)
    label_img[label_img > 5] = 255
    label_img[label_img < 5] = 226
    cv.imwrite(intercept_dir + '/parabola_{}'.format(i) + '.png', label_img)


def make_parabolas(count, parabola_dir, intercept_dir, csv_dir):
    random.seed()
    parabola_names = [['filename']]
    for i in range(count):
        parabola_names.append(['parabola_{}'.format(i) + '.png'])
        a = random.uniform(0.5, 1.0)*(-1)**random.randint(0, 1)
        b = random.uniform(0.0, 0.3)*(-1)**random.randint(0, 1)
        c = random.uniform(-1.0, 1.0)
        x_margin = random.uniform(2.0, 5.0)
        y_margin = random.uniform(2.0, 5.0)
        make_parabola(a, b, c, x_margin, y_margin, i, parabola_dir, intercept_dir)
    with open(csv_dir+'/filenames.csv', 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(parabola_names)


if __name__ == '__main__':
    make_parabolas(5000, 'parabolas', 'intercepts', '.')
