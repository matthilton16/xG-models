import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class XGDistributionPlots:
    def __init__(self):
        """Initializes the xG Distribution plotting utility."""
        pass

    def plot_xg_distribution(self, xg_mean, xg_var, 
                             mean_bins=100, var_bins=50, 
                             var_filter_threshold=0.5, 
                             show_plot=True, save_path=None):
        """
        Plots the distribution of xG mean and variance.

        Args:
            xg_mean (np.array): Array of xG mean values.
            xg_var (np.array): Array of xG variance values.
            mean_bins (int): Number of bins for the xG mean histogram.
            var_bins (int): Number of bins for the xG variance histogram.
            var_filter_threshold (float): Upper threshold to filter xG variance for plotting.
            show_plot (bool): If True, displays the plot.
            save_path (str, optional): Path to save the plot. If None, plot is not saved.
        """
        plt.figure(figsize=(10, 7))

        plt.hist(xg_mean, bins=mean_bins, color='purple', edgecolor='black', alpha=0.6, label='xG Mean')

        if xg_var is not None:
            xg_var_filtered = xg_var[xg_var < var_filter_threshold]
            plt.hist(xg_var_filtered, bins=var_bins, color='orange', edgecolor='black', alpha=0.9, label=f'xG Variance')

        plt.xlabel('xG Value', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of xG Mean and Variance', fontsize=16)
        plt.xlim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Distribution plot saved to {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close() # Close plot if not shown to free memory

class XGPitchMapPlots:
    def __init__(self, pitch_image_path='img/pitch_white.png'):
        """
        Initializes the xG Pitch Map plotting utility.

        Args:
            pitch_image_path (str): Path to the pitch background image.
        """
        self.pitch_image_path = pitch_image_path
        self.pitch_img = None
        if os.path.exists(self.pitch_image_path):
            try:
                self.pitch_img = mpimg.imread(self.pitch_image_path)
                print(f"Pitch image loaded from {self.pitch_image_path}")
            except Exception as e:
                print(f"Warning: Could not load pitch image from {self.pitch_image_path}. Error: {e}")
        else:
            print(f"Warning: Pitch image not found at {self.pitch_image_path}. Pitch maps will be plotted without background image.")

    def plot_pitch_map(self, x_coords, y_coords, intensity_values,
                       map_title='xG Pitch Map',
                       colorbar_label='xG',
                       cmap='viridis',
                       size_scaling_factor=250,
                       plot_buffer=None,
                       plot_xlim=(53,100), # e.g. (0,53) to focus on one half 
                       show_plot=True, save_path=None):
        """
        Plots intensity values (like xG mean or variance) on a 2D pitch map.

        Args:
            x_coords (np.array): X coordinates of the shots/events.
            y_coords (np.array): Y coordinates of the shots/events.
            intensity_values (np.array): Values to represent by color and/or size.
            map_title (str): Title of the plot.
            colorbar_label (str): Label for the colorbar.
            cmap (str): Colormap for the scatter plot.
            size_scaling_factor (float): Factor to scale point sizes by intensity.
            plot_buffer (int, optional): Number of points to plot (randomly sampled if more available).
            plot_xlim (tuple, optional): Tuple (min, max) for x-axis limits of the plot.
            show_plot (bool): If True, displays the plot.
            save_path (str, optional): Path to save the plot. If None, plot is not saved.
        """
        
        if plot_buffer and len(x_coords) > plot_buffer:
            indices = np.random.choice(len(x_coords), plot_buffer, replace=False)
            x_plot = x_coords[indices]
            y_plot = y_coords[indices]
            intensity_plot = intensity_values[indices]
        else:
            x_plot = x_coords
            y_plot = y_coords
            intensity_plot = intensity_values

        pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black')
        fig, ax = pitch.draw(figsize=(16, 10))
        
        scatter = ax.scatter(x_plot, y_plot,
                           s=intensity_plot * size_scaling_factor,
                           c=intensity_plot,
                           cmap=cmap,
                           alpha=0.7,
                           edgecolors='black')
        
        plt.colorbar(scatter, label=colorbar_label)
        ax.set_title(map_title, fontsize=16)

        if plot_xlim:
            ax.set_xlim(plot_xlim)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Pitch map plot saved to {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close() # Close plot if not shown to free memory
