import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.cm as cm
from pykrige.ok3d import OrdinaryKriging3D
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

'''
draw_result: 根据不同参数有不同功能
(之后用Python装饰器改一改)
'''
def draw_result(data,
                model,
                model_cpu = None,
                source={"x": 63, "y": 58, "z": 29},
                plotDim=2,
                axe="x"):

    # 3D plot prediction
    if plotDim == 3:
        draw_result3D(data, 
                      model, 
                      model_cpu, 
                      source=source
                      )
        return 0

    # 2D plot according to the axis
    if plotDim == 2:
        # Set up the plot area
        if source[axe] < inner_radius_max:
            print("The partition range exceeds the limit of the coordinate axis, drawing forward")
            x = [i if axe == "x" else source["x"] for i in range(source[axe], source[axe] + inner_radius_max)]
            y = [i if axe == "y" else source["y"] for i in range(source[axe], source[axe] + inner_radius_max)]
            z = [i if axe == "z" else source["z"] for i in range(source[axe], source[axe] + inner_radius_max)]
        else:
            x = [i if axe == "x" else source["x"] for i in range(source[axe] - inner_radius_max, source[axe] + inner_radius_max)]
            y = [i if axe == "y" else source["y"] for i in range(source[axe] - inner_radius_max, source[axe] + inner_radius_max)]
            z = [i if axe == "z" else source["z"] for i in range(source[axe] - inner_radius_max, source[axe] + inner_radius_max)]
        true_values = [data[z[i]][y[i]][x[i]] for i in range(len(x))]
        predict_values, ss = model.execute('points', np.array(x).astype(float), np.array(y).astype(float), np.array(z).astype(float))
        if source[axe] < inner_radius_max:
            plt.plot([i for i in range(source[axe], source[axe] + inner_radius_max)], predict_values, color='b', linestyle="--", label="predict")
            ## True graph
            plt.plot([i for i in range(source[axe], source[axe] + inner_radius_max)], true_values, color='r', linestyle="-", label='true')

            plt.xlabel(axe, fontsize=25)
            plt.ylabel("Dose", fontsize=25)
            plt.title(axe + " axis direction", fontsize=30)
            fig = plt.gcf()
            fig.set_size_inches(15, 12)
            plt.legend(fontsize=20)
            plt.tick_params(labelsize=20)
            plt.savefig(axe + ".png", bbox_inches='tight', dpi=35, transparent=True)
        else:
            ## Predicted graph
            plt.plot([i for i in range(source[axe] - inner_radius_max, source[axe] + inner_radius_max)], predict_values, color='b', linestyle="--", label="predict")
            ## True graph
            plt.plot([i for i in range(source[axe] - inner_radius_max, source[axe] + inner_radius_max)], true_values, color='r', linestyle="-", label='true')

            plt.xlabel(axe, fontsize=25)
            plt.ylabel("Dose", fontsize=25)
            plt.title(axe + " axis direction", fontsize=30)
            fig = plt.gcf()
            fig.set_size_inches(15, 12)
            plt.legend(fontsize=20)
            plt.tick_params(labelsize=20)
            plt.savefig(axe + ".png", bbox_inches='tight', dpi=35, transparent=True)
        return 0

'''
draw_result3D: Use nested functions to clarify functional division.
'''
#3D prediction graph
def draw_result3D(data,
                  model,
                  model_cpu,
                  draw_source=False,
                  vol_size=1,
                  source={"x": 63, "y": 58, "z": 29}
                  ):

    # Define color values and corresponding positions
    colors = ['none', 'blue', 'green','yellow' , 'red', 'black']
    if draw_source:
        alpha = [0.0, 0.4, 0.6, 0.9, 1.0]
    else:
        alpha = [0.0, 0.0001, 0.01, 0.1, 1.0]

    # Create a color mapping object
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', list(zip(alpha, colors)), N=256)

    # Parameters needed for plotting
        
    x = np.array([i for i in data["x"]])
    y = np.array([i for i in data["y"]])
    z = np.array([i for i in data["z"]])

    target_values_test = np.array([data["target"][i] for i in range(len(x))])
    predict_values_test, ss = model.execute('points', x.astype(float), y.astype(float), z.astype(float))
    predict_values_test_cpu, ss_cpu = model_cpu.execute('points', x.astype(float), y.astype(float), z.astype(float))

    draw_unit(x,
              y,
              z,
              target_values_test,
              predict_values_test,
              predict_values_test_cpu,
              cmap,
              draw_source,
              source,
              vol_size,
              )
    return 0

def draw_unit(x,
              y,
              z,
              target_values_test,
              predict_values_test,
              predict_values_test_cpu,
              cmap,
              draw_source,
              source,
              vol_size,
              ):
    fig = plt.figure(figsize=(9, 3))
    plt.subplots_adjust(wspace=0, bottom=0.2)
    # Subplot 1 (true value)
    # 创建颜色映射对象
    cmap_projection = cmap
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(x, y, z, c=target_values_test, s=2, cmap=cmap, alpha=0.5)
    ax1.set_xlim(x.min()-1, x.max()+1)
    ax1.set_ylim(y.min()-1, y.max()+1)
    ax1.set_zlim(z.min()-1, z.max()+1)
    ax1.set_xlabel('x', fontsize=10, labelpad=1.2)
    ax1.set_ylabel('y', fontsize=10, labelpad=1.2)
    ax1.set_zlabel('z', fontsize=10, labelpad=1.2)
    ax1.set_title('Simulated', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=8, pad=0)

    add_projection(ax1, x, y, z, target_values_test, cmap_projection)

    ## Find the index of the maximum target
    max_target_index = np.argmax(target_values_test)
    ## Mark the maximum target point
    ax1.scatter(x[max_target_index], y[max_target_index], z[max_target_index], c='black', edgecolors='black', s=30)

    if draw_source:
        axes = [70, 70, 70]
        data = np.zeros(axes, dtype=bool)
        if vol_size == 0:
            data[source["x"]][source["y"]][source["z"]] = 1
        else:
            data[source["x"]-vol_size:source["x"]+vol_size, source["y"]-vol_size:source["y"]+vol_size, source["z"]-vol_size:source["z"]+vol_size] = 1

        alpha = 0.2
        colors = np.array([0, 1, 0, alpha])
        ax1.voxels(data, facecolors=colors)
        ax1.set_xlim(source["x"]-2, source["x"]+2)
        ax1.set_ylim(source["y"]-2, source["y"]+2)
        ax1.set_zlim(source["z"]-2, source["z"]+2)
        plt.savefig(r"c.png", bbox_inches='tight', dpi=100, transparent=True)
        return 0

    # Subplot 2 (predicted value)
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(x, y, z, c=predict_values_test, s=2, cmap=cmap, alpha=0.5)
    ax2.set_xlim(x.min() - 1, x.max() + 1)
    ax2.set_ylim(y.min() - 1, y.max() + 1)
    ax2.set_zlim(z.min() - 1, z.max() + 1)
    ax2.set_xlabel('x', fontsize=10, labelpad=1.2)
    ax2.set_ylabel('y', fontsize=10, labelpad=1.2)
    ax2.set_zlabel('z', fontsize=10, labelpad=1.2)
    ax2.set_title('GPU-Krige', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=8, pad=0)

    add_projection(ax2, x, y, z, predict_values_test, cmap_projection)

    ## Find the index of the maximum target
    max_target_index = np.argmax(predict_values_test)
    ## Mark the maximum target point
    ax2.scatter(x[max_target_index], y[max_target_index], z[max_target_index], c='black', edgecolors='black', s=30)

    # Subplot 3 (predicted value cpu)
    ax3 = fig.add_subplot(133, projection='3d')
    scatter = ax3.scatter(x, y, z, c=predict_values_test_cpu, s=2, cmap=cmap, alpha=0.5)
    ax3.set_xlim(x.min() - 1, x.max() + 1)
    ax3.set_ylim(y.min() - 1, y.max() + 1)
    ax3.set_zlim(z.min() - 1, z.max() + 1)
    ax3.set_xlabel('x', fontsize=10, labelpad=1.2)
    ax3.set_ylabel('y', fontsize=10, labelpad=1.2)
    ax3.set_zlabel('z', fontsize=10, labelpad=1.2)
    ax3.set_title('CPU-Krige', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=8, pad=0)

    add_projection(ax3, x, y, z, predict_values_test_cpu, cmap_projection)

    ## Find the index of the maximum target
    max_target_index = np.argmax(predict_values_test_cpu)
    ## Mark the maximum target point
    ax3.scatter(x[max_target_index], y[max_target_index], z[max_target_index], c='black', edgecolors='black', s=30)

    # Add unified color bar
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(predict_values_test_cpu), vmax=max(predict_values_test_cpu)))
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.05])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax, aspect=10, orientation='horizontal')
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.05, 0.17, r"Dose Rate/(mGy$\cdot h^{-1}$)", ha='left', va='center', fontsize=8)

    plt.savefig(r"precision.pdf", format="pdf", bbox_inches='tight', dpi=300, transparent=True)
    plt.show()

# Function to add projection on y-z plane
def add_projection(ax, x, y, z, values, cmap):
    # Create grid for projection
    yi, zi = np.meshgrid(np.linspace(y.min(), y.max(), 100), np.linspace(z.min(), z.max(), 100))
    xi = np.ones_like(yi) * (x.max() + 1)
    
    # Interpolate values for the grid
    from scipy.interpolate import griddata
    projection = griddata((y, z), values, (yi, zi), method='linear')
    
    # Plot the surface projection
    ax.plot_surface(xi, yi, zi, facecolors=cmap((projection - projection.min()) / (projection.max() - projection.min())), rstride=1, cstride=1, antialiased=False, alpha=0.8)