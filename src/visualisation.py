import numpy as np
import matplotlib.pyplot as plt
from skimage import color

def plot_image(ax, img, title=""):
    ax.imshow(img)
    ax.axis('off')
    if title != "":
        ax.set_title(title)

def plot_percentages(ax, counts, centers, title=""):
    idx = np.argsort(counts)[::-1]
    
    # Set bar labels
    bar_labels = [f"{center}" for center in centers[idx]]
    
    percentages = counts[idx] / counts.sum() * 100
    
    # Build plot
    ax.bar(bar_labels, percentages, color=centers[idx]/255)
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, rotation=90)
    ax.set_xlabel("Colour (R,G,B)")
    ax.set_ylabel("Percentage (%)")
    if title != "":
        ax.set_title(title)

def plot_hues(ax, centers, title=""):
    # Draw the full colour wheel as a background
    theta = np.linspace(0, 2 * np.pi, 360)
    for t in theta:
        ax.bar(t, 1, width=2*np.pi/360, color=plt.cm.hsv(t / (2 * np.pi)), linewidth=0)

    hues = color.rgb2hsv(centers)[:,0]
    
    # Plot each cluster as a marker on the wheel, sized by percentage
    for i in range(len(hues)):
        angle = hues[i] * 2 * np.pi
        ax.scatter(angle, 1, s=100, color="white",
                   edgecolors='black', linewidth=1.5, zorder=5)
        ax.plot([angle, angle], [0, 1], color='black', linewidth=1.5, zorder=4)
    
    ax.set_yticklabels([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels([f"{int(h)}°" for h in np.linspace(0, 360, 12, endpoint=False)])
    if title != "":
        ax.set_title(title)