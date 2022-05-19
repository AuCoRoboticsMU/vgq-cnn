import numpy as np
from path_settings import DATA_PATH
import os
import matplotlib.pyplot as plt
import argparse

"""
Script to analyse the dataset statistics of a VGQ-dset. It plots various graphs for positivity rate and
number of grasps over beta, phi, psi and d. In addition, .npy files are saved for undersampling strategies,
needed to run tools/undersample_beta.py, tools/undersample_beta_positivity.py and tools/undersample_phi.py.
"""

parser = argparse.ArgumentParser(description="Analyse a VGQ-dset.")
parser.add_argument("dset_name",
                    type=str,
                    default=None,
                    help="Name of dataset")
parser.add_argument("--collisions",
                    type=bool,
                    default=False,
                    help="analyse the collision rate")

args = parser.parse_args()
dset_name = args.dset_name
COLLISIONS = args.collisions

data_dir = '{}{}/tensors/'.format(DATA_PATH, dset_name)
graph_name = dset_name
save_dir = data_dir + '../'

BACKGROUND = True
hist_color = (0.2, 0.6, 1)
tp = False


# Unicode for letter strings
psi_angle = '\u03C8'
beta_angle = '\u03B2'
phi_angle = '\u03C6'

COLLISION = False

rad_step = 0.05
radius_bins = np.arange(0.4, 1.15, rad_step)
phi_step = 5
phi_bins = np.arange(0, 75, phi_step)
psi_step = 5
psi_bins = np.arange(0, 95, psi_step)
beta_step = 5
beta_bins = np.arange(0, 95, beta_step)

def plot_heatmap(data, x_axis, y_axis, title, save_name, unit, color="Greens", v_max=None):
    """
    Plots and saves a heatmap with matplotlib.

    Parameters
    ----------
    data (numpy.ndarray): data to be plotted in the heatmap
    x_axis (str):  x_axis variable, can be 'psi', 'd', 'beta'
    y_axis (str): y_axis variable, can be 'psi', 'phi', 'beta'
    title (str): Title for heatmap.
    save_name (str): Where to save the image (full path)
    unit (str): Unit to label the color bar with.
    color (str): Color of the color bar, defaults to "Greens".
    v_max (float): Maximum value of the heatmap color bar, defaults to None (== maximum value in data).
    """
    if x_axis == 'psi':
        sz_x = len(psi_bins) - 1
        x_label = 'Grasp angle psi \u03C8 [\u00b0]'
        grid_x = np.arange(0, psi_bins[-1] + psi_step, psi_step)
    elif x_axis == 'd':
        sz_x = len(radius_bins) - 1
        x_label = 'Grasp distance d'
        grid_x = np.arange(0, radius_bins[-1] + rad_step, rad_step)
    elif x_axis == 'beta':
        sz_x = len(beta_bins) - 1
        x_label = 'Grasp-table angle beta {} [\u00b0]'.format(beta_angle)
        grid_x = np.arange(0, beta_bins[-1] + beta_step, beta_step)
    else:
        raise AttributeError("x_axis '{}' cannot be plotted in mesh".format(x_axis))
    if y_axis == 'psi':
        sz_y = len(psi_bins) - 1
        y_label = 'Grasp angle psi \u03C8 [\u00b0]'
        grid_y = np.arange(0, psi_bins[-1] + psi_step, psi_step)
    elif y_axis == 'phi':
        sz_y = len(phi_bins) - 1
        y_label = 'Elevation angle \u03C6 [\u00b0]'
        grid_y = np.arange(0, phi_bins[-1] + phi_step, phi_step)
    elif y_axis == 'beta':
        sz_y = len(beta_bins) - 1
        y_label = 'Grasp-table angle beta {} [\u00b0]'.format(beta_angle)
        grid_y = np.arange(0, beta_bins[-1] + beta_step, beta_step)
    else:
        raise AttributeError("y_axis '{}' cannot be plotted in mesh".format(y_axis))

    combined_accuracy_map = np.array(data).reshape((sz_y, sz_x))

    x, y = np.meshgrid(grid_x,
                       grid_y)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, combined_accuracy_map.reshape((sz_y, sz_x)),
                      cmap=color,
                      vmin=0,
                      vmax=v_max)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax, label=unit)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title('{} {}'.format(title, graph_name))
    plt.savefig('{}/{}_{}.png'.format(save_dir, dset_name, save_name))
    plt.close()

def plot_histogram(data, x_axis, title, save_name):
    """
    Plots and saves a histogram with matplotlib.

    Parameters
    ----------
    data (numpy.ndarray): data to be plotted in the histograp
    x_axis (str):  x_axis variable, can be 'phi', 'psi', 'd', 'beta'
    title (str): Title for histogram.
    save_name (str): Where to save the image (full path)
    """
    if x_axis == 'phi':
        x_label = '{} [\u00b0]'.format(phi_angle)
        bins = phi_bins
    elif x_axis == 'psi':
        x_label = '{} [\u00b0]'.format(psi_angle)
        bins = psi_bins
    elif x_axis == 'beta':
        x_label = '{} [\u00b0]'.format(beta_angle)
        bins = beta_bins
    elif x_axis == 'd':
        x_label = 'Camera distance d'
        bins = radius_bins
    else:
        raise AttributeError("Cannot plot histogram for {}".format(x_axis))

    plt.hist(data, bins=bins, color=hist_color)
    plt.title('{} in {}'.format(title, dset_name))
    plt.ylabel('Number of grasps')
    plt.xlabel(x_label)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    plt.savefig('{}{}.png'.format(save_dir, save_name), transparent=tp)
    plt.close()

def plot_line(x_data, y_data, title, x_axis, y_axis, save_name, y_lim=None):
    """
    Plots and saves a line plot with matplotlib.

    Parameters
    ----------
    x_data (numpy.ndarray): values of the x-variable.
    y_data (numpy.ndarray): values of the y-variable.
    title (str): Title for line plot.
    x_axis (str):  x_axis variable, can be 'phi', 'psi', 'd', 'beta'
    y_axis (str): y_axis variable, can be positivity rate 'pos' or collision rate 'coll'
    save_name (str): Where to save the image (full path)
    y_lim (float): Maximum value of the y-axis, defaults to None - y limits will be set to (0, 100).
    """
    if x_axis == 'phi':
        x_label = '{} [\u00b0]'.format(phi_angle)
    elif x_axis == 'psi':
        x_label = '{} [\u00b0]'.format(psi_angle)
    elif x_axis == 'beta':
        x_label = '{} [\u00b0]'.format(beta_angle)
    elif x_axis == 'd':
        x_label = 'Camera distance d'
    else:
        raise AttributeError("Cannot plot line for x - {}".format(x_axis))
    if y_axis == 'pos':
        y_label = 'Positivity rate [%]'
    elif y_axis == 'coll':
        y_label = 'Collision rate [%]'
    else:
        raise AttributeError("Cannot plot line for y - {}".format(y_axis))
    plt.plot(x_data[:-1], y_data, 'x-', color=hist_color)
    plt.title('{} in {}'.format(title, dset_name))
    if y_lim is not None:
        plt.ylim((0, y_lim))
    else:
        plt.ylim((0, 100))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('{}{}.png'.format(save_dir, save_name), transparent=tp)
    plt.close()

def plot_scatter(x_data, y_data, title, x_axis, y_axis, save_name):
    """
    Plots and saves a scatter plot with matplotlib.

    Parameters
    ----------
    x_data (numpy.ndarray): values of the x-variable.
    y_data (numpy.ndarray): values of the y-variable.
    title (str): Title for line plot.
    x_axis (str):  x_axis variable, can be 'phi', 'psi', 'd', 'beta'
    y_axis (str): y_axis variable, can be 'phi', 'psi', 'd', 'beta'
    save_name (str): Where to save the image (full path)
    """
    if x_axis == 'phi':
        x_label = '{} [\u00b0]'.format(phi_angle)
    elif x_axis == 'psi':
        x_label = '{} [\u00b0]'.format(psi_angle)
    elif x_axis == 'beta':
        x_label = '{} [\u00b0]'.format(beta_angle)
    elif x_axis == 'd':
        x_label = 'Camera distance d'
    else:
        raise AttributeError("Cannot plot line for x - {}".format(x_axis))
    if y_axis == 'phi':
        y_label = '{} [\u00b0]'.format(phi_angle)
    elif y_axis == 'psi':
        y_label = '{} [\u00b0]'.format(psi_angle)
    elif y_axis == 'beta':
        y_label = '{} [\u00b0]'.format(beta_angle)
    elif y_axis == 'd':
        y_label = 'Camera distance d'
    else:
        raise AttributeError("Cannot plot line for y - {}".format(y_axis))
    plt.scatter(x_data, y_data)
    plt.title('{} in {}'.format(title, dset_name))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('{}{}.png'.format(save_dir, save_name), transparent=tp)
    plt.close()

def generate_phi_psi_overview(phi, psi, labels, collision=False):
    """
    Calculates and plots an overview over phi (camera elevation angle) and psi (grasp-camera angle).

    Parameters
    ----------
    phi (numpy.array): Phi angles in degree.
    psi (numpy.array):  Psi angles in degree.
    labels (numpy.array): Binary array indicating ground-truth class of grasps or
                            collisions of a grasp, if collision = True.
    collision (bool): Indicator, if collision plots should be created.
    """
    phi_psi_pos = []
    phi_psi_num_pos = []
    phi_psi_num = []
    for _phi in phi_bins[:-1]:
        for _psi in psi_bins[:-1]:
            mask = (phi.squeeze() >= _phi) & \
                   (phi.squeeze() < _phi + phi_step) & \
                   (psi.squeeze() >= _psi) & \
                   (psi.squeeze() < _psi + psi_step)
            aligned_grasps_in_bin = labels[mask]
            try:
                phi_psi_pos.append(sum(aligned_grasps_in_bin) / len(aligned_grasps_in_bin) * 100)
            except ZeroDivisionError:
                phi_psi_pos.append(0.0)
            phi_psi_num_pos.append(sum(aligned_grasps_in_bin))
            phi_psi_num.append(len(aligned_grasps_in_bin))

    if collision:
        np.save('{}/phi_psi_collision_rate.npy'.format(save_dir), 100 - np.array(phi_psi_pos))
        plot_heatmap(data=100 - np.array(phi_psi_pos), x_axis='psi', y_axis='phi', title='Collision rate grasps',
                     save_name='collision_psi_phi', unit='Collision rate', v_max=100)
    else:
        np.save('{}/phi_psi_positivity_rate.npy'.format(save_dir), np.array(phi_psi_pos))
        np.save('{}/phi_steps.npy'.format(save_dir), np.repeat(phi_bins, len(psi_bins)))
        np.save('{}/psi_steps.npy'.format(save_dir), np.tile(psi_bins, len(phi_bins)))

        plot_heatmap(data=phi_psi_pos, x_axis='psi', y_axis='phi', title='Positivity rate grasps',
                     save_name='positivity_psi_phi', unit='Positivity rate [%]', v_max=25)

        plot_heatmap(data=phi_psi_num_pos, x_axis='psi', y_axis='phi', title='Number of positive grasps',
                     save_name='num_pos_psi_phi', unit='Number positive grasps')

        plot_heatmap(data=phi_psi_num, x_axis='psi', y_axis='phi', title='Number grasps', save_name='num_psi_phi',
                     unit='Number of grasps')

def generate_phi_beta_overview(phi, beta, labels, collision=False):
    """
    Calculates and plots an overview over phi (camera elevation angle) and beta (grasp-table angle).

    Parameters
    ----------
    phi (numpy.array): Phi angles in degree.
    beta (numpy.array):  beta angles in degree.
    labels (numpy.array): Binary array indicating ground-truth class of grasps or
                            collisions of grasps, if collision = True.
    collision (bool): Indicator, if collision plots should be created.
    """
    phi_beta_pos = []
    phi_beta_num_pos = []
    phi_beta_num = []
    for _phi in phi_bins[:-1]:
        for _beta in beta_bins[:-1]:
            mask = (phi.squeeze() >= _phi) & \
                   (phi.squeeze() < _phi + phi_step) & \
                   (beta.squeeze() >= _beta) & \
                   (beta.squeeze() < _beta + beta_step)
            aligned_grasps_in_bin = labels[mask]
            try:
                phi_beta_pos.append(sum(aligned_grasps_in_bin) / len(aligned_grasps_in_bin) * 100)
            except ZeroDivisionError:
                phi_beta_pos.append(0.0)
            phi_beta_num_pos.append(sum(aligned_grasps_in_bin))
            phi_beta_num.append(len(aligned_grasps_in_bin))

    if collision:
        np.save('{}/phi_beta_collision_rate.npy'.format(save_dir), 100 - np.array(phi_beta_pos))
        plot_heatmap(data=100 - np.array(phi_beta_pos), x_axis='beta', y_axis='phi',
                     title='Collision rate grasps', save_name='collision_beta_phi', unit='Collision rate', v_max=100.0)
    else:
        np.save('{}/phi_beta_positivity_rate.npy'.format(save_dir), np.array(phi_beta_pos))
        np.save('{}/beta_steps.npy'.format(save_dir), np.tile(beta_bins, len(phi_bins)))

        plot_heatmap(data=phi_beta_pos, x_axis='beta', y_axis='phi', title='Positivity rate grasps',
                     save_name='positivity_beta_phi', unit='Positivity rate [%]', v_max=25)

        plot_heatmap(data=phi_beta_num_pos, x_axis='beta', y_axis='phi', title='Number of positive grasps',
                     save_name='num_pos_beta_phi', unit='Number positive grasps')

        plot_heatmap(data=phi_beta_num, x_axis='beta', y_axis='phi', title='Number grasps',
                     save_name='num_beta_phi', unit='Number of grasps')

def generate_phi_dist_overview(phi, rad, labels):
    """
    Calculates and plots an overview over phi (camera elevation angle) and d (grasp-camera distance).

    Parameters
    ----------
    phi (numpy.array): Phi angles in degree.
    d (numpy.array):  Grasp-camera distance (T^g - T^c).
    labels (numpy.array): Binary array indicating ground-truth class of grasps.
    collision (bool): Indicator, if collision plots should be created.
    """
    pos_rate = []
    num_pos = []
    num_all = []
    for _phi in phi_bins[:-1]:
        for _rad in radius_bins[:-1]:
            mask = (phi.squeeze() >= _phi) & \
                   (phi.squeeze() < _phi + phi_step) & \
                   (rad.squeeze() >= _rad) & \
                   (rad.squeeze() < _rad + rad_step)
            aligned_grasps_in_bin = labels[mask]
            try:
                pos_rate.append(sum(aligned_grasps_in_bin) / len(aligned_grasps_in_bin) * 100)
            except ZeroDivisionError:
                pos_rate.append(0.0)
            num_pos.append(sum(aligned_grasps_in_bin))
            num_all.append(len(aligned_grasps_in_bin))

    plot_heatmap(data=pos_rate, x_axis='d', y_axis='phi', title='Positivity rate grasps', save_name='positivity_d_phi',
                 unit='Positivity rate [%]', v_max=25)

    plot_heatmap(data=num_pos, x_axis='d', y_axis='phi', title='Number of positive grasps', save_name='num_pos_d_phi',
                 unit='Number positive grasps')

    plot_heatmap(data=num_all, x_axis='d', y_axis='phi', title='Number grasps', save_name='num_d_phi',
                 unit='Number of grasps')

def generate_phi_overview(phi, labels, cf):
    """
    Calculates and plots an overview over the parameter phi (camera elevation angle).

    Parameters
    ----------
    phi (numpy.array): Phi angles in degree.
    labels (numpy.array): Binary array indicating ground-truth class of grasps.
    cf (numpy.array): Binary array indicating if grasps are in collision.
    """
    plot_histogram(phi,
                   x_axis='phi',
                   title='Grasps per \u03C6',
                   save_name='phi_num_grasps')

    phi_positivity_rate = []
    collided_grasp_ratio = []
    grasps_in_bin = []
    for _phi in phi_bins[:-1]:
        angles = (phi.squeeze() > _phi) & (phi.squeeze() <= _phi + phi_step)
        included_grasps = labels[angles]
        grasps_in_bin.append(len(included_grasps) * step)
        positive_grasps = (included_grasps == 1)
        if COLLISION:
            collision_free_in_phi = cf[angles]
            collided_grasp_ratio.append((len(collision_free_in_phi) - sum(collision_free_in_phi))
                                        / len(collision_free_in_phi) * 100)
        positive = len(included_grasps[positive_grasps])
        try:
            phi_positivity_rate.append(positive / len(included_grasps) * 100)
        except ZeroDivisionError:
            phi_positivity_rate.append(0)

    # Plot positivity rate over elevation (phi) angles
    plot_line(x_data=phi_bins,
              y_data=phi_positivity_rate,
              title="Positivity rate per \u03C6",
              x_axis='phi',
              y_axis='pos',
              y_lim=25,
              save_name='phi_positivity_rate')

    if COLLISION:
        plot_line(x_data=phi_bins,
                  y_data=collided_grasp_ratio,
                  title="Collision rate per \u03C6",
                  x_axis='phi',
                  y_axis='coll',
                  save_name='phi_collision_rate')

    np.save(save_dir + 'phi_grasps.npy', grasps_in_bin)

def generate_psi_overview(psi, labels, cf):
    """
    Calculates and plots an overview over the parameter psi (grasp-camera angle).

    Parameters
    ----------
    psi (numpy.array): Psi angles in degree.
    labels (numpy.array): Binary array indicating ground-truth class of grasps.
    cf (numpy.array): Binary array indicating if grasps are in collision.
    """
    plot_histogram(psi_angles,
                   x_axis='psi',
                   title='Grasps per {}'.format(psi_angle),
                   save_name='psi_num_grasps')

    positivity_rate = []
    collided_grasp_ratio = []
    for _psi in psi_bins[:-1]:
        angles = (psi.squeeze() > _psi) & (psi.squeeze() <= _psi + psi_step)
        grasps_in_phi = labels[angles]
        positive_grasps = (grasps_in_phi == 1)
        if COLLISION:
            collision_free_in_phi = cf[angles]
            try:
                collided_grasp_ratio.append((len(collision_free_in_phi) - sum(collision_free_in_phi))
                                            / len(collision_free_in_phi) * 100)
            except ZeroDivisionError:
                collided_grasp_ratio.append(0.0)
        positive = len(grasps_in_phi[positive_grasps])
        try:
            positivity_rate.append(positive / len(grasps_in_phi) * 100)
        except ZeroDivisionError:
            positivity_rate.append(0)

    # Plot positivity rate over elevation (phi) angles
    plot_line(x_data=psi_bins,
              y_data=positivity_rate,
              title="Positivity rate per {}".format(psi_angle),
              x_axis='psi',
              y_axis='pos',
              save_name='psi_positivity_rate')

    if COLLISION:
        plot_line(x_data=psi_bins,
                  y_data=collided_grasp_ratio,
                  title="Collision rate per {}".format(psi_angle),
                  x_axis='psi',
                  y_axis='coll',
                  save_name='psi_collision_rate')

def generate_beta_overview(beta, labels, cf):
    """
    Calculates and plots an overview over the parameter beta (grasp-table angle).

    Parameters
    ----------
    beta (numpy.array): Beta angles in degree.
    labels (numpy.array): Binary array indicating ground-truth class of grasps.
    cf (numpy.array): Binary array indicating if grasps are in collision.
    """
    plot_histogram(beta,
                   x_axis='beta',
                   title='Grasps per {}'.format(beta_angle),
                   save_name='beta_num_grasps')

    all_grasps = len(labels)
    positivity_rate = []
    collided_grasp_ratio = []
    grasps_ratio = []
    grasps_in_bin = []
    for _beta in beta_bins[:-1]:
        angles = (beta.squeeze() > _beta) & (beta.squeeze() <= _beta + beta_step)
        included_grasps = labels[angles]
        grasps_ratio.append(len(included_grasps) / all_grasps)
        grasps_in_bin.append(len(included_grasps) * step)
        positive_grasps = (included_grasps == 1)
        if COLLISION:
            no_collision = cf[angles]
            try:
                collided_grasp_ratio.append((len(no_collision) - sum(no_collision))
                                            / len(no_collision) * 100)
            except ZeroDivisionError:
                collided_grasp_ratio.append(0.0)
        positive = len(included_grasps[positive_grasps])
        try:
            positivity_rate.append(positive / len(included_grasps) * 100)
        except ZeroDivisionError:
            positivity_rate.append(0)

    # Plot positivity rate over elevation (phi) angles
    plot_line(x_data=beta_bins,
              y_data=positivity_rate,
              title="Positivity rate per {}".format(beta_angle),
              x_axis='beta',
              y_axis='pos',
              y_lim=50,
              save_name='beta_positivity_rate')

    if COLLISION:
        print("Collision rate per {}:".format(beta_angle))
        print(collided_grasp_ratio)
        plot_line(x_data=beta_bins,
                  y_data=collided_grasp_ratio,
                  title="Collision rate per {}".format(beta_angle),
                  x_axis='beta',
                  y_axis='coll',
                  save_name='beta_collision_rate')

    np.save(save_dir + 'beta_positivity_rate.npy', positivity_rate)
    np.save(save_dir + 'beta_grasps.npy', grasps_in_bin)


print("Read data")
# gather data
total_tensors = len([1 for x in os.listdir(data_dir) if 'image_labels' in x])

# If the dataset is too big, only read every n-th file to reduce memory overload
step = total_tensors // 10000

if step == 0:
    step = 1
print("Tensors: {}, steps {}".format(total_tensors, step))

try:
    _ = np.load(data_dir + 'camera_poses_{:06d}.npz'.format(0))['arr_0']
    file_format = '{:06d}.npz'
except FileNotFoundError:
    file_format = '{:05d}.npz'
# Load first tensors
num = file_format.format(0)
cam_poses = np.load(data_dir + 'camera_poses_{}'.format(num))['arr_0']
phi_angles = list(cam_poses[:, 1].squeeze())
dist = list(cam_poses[:, 0].squeeze())

grasp_labels = list(np.where(np.load(data_dir + 'robust_ferrari_canny_{}'.format(num))['arr_0']
                             >= 0.002, 1, 0))
collision_free = list(np.load(data_dir + 'collision_free_{}'.format(num))['arr_0'])
poses = np.load(data_dir + 'hand_poses_{}'.format(num))['arr_0']
q = list(poses[:, 4:8])
psi_angles = list(poses[:, -2])
beta_angles = list(poses[:, -1])

# Read in other tensors
for cnt in range(0, total_tensors, step):
    num = file_format.format(cnt)
    poses = np.load(data_dir + 'hand_poses_{}'.format(num))['arr_0']
    grasp_label = np.where(np.load(data_dir + 'robust_ferrari_canny_{}'.format(num))['arr_0']
                           >= 0.002, 1, 0)

    collision_free.extend(np.load(data_dir + 'collision_free_{}'.format(num))['arr_0'])
    q.extend(list(poses[:, 4:8]))
    psi_angles.extend(list(poses[:, -2]))
    beta_angles.extend(list(poses[:, -1]))
    cam_poses = np.load(data_dir + 'camera_poses_{}'.format(num))['arr_0']
    phi_angles.extend(list(cam_poses[:, 1].squeeze()))
    dist.extend(list(cam_poses[:, 0].squeeze()))
    grasp_labels.extend(grasp_label)

# Convert list into numpy arrays and angles from radians to degree
collision_free = np.array(collision_free)
phi_angles = np.rad2deg(np.array(phi_angles))
beta_angles = np.rad2deg(np.array(beta_angles))
psi_angles = np.rad2deg(np.array(psi_angles))
dist = np.array(dist)
grasp_labels = np.array(grasp_labels)


#########################################################
print("Calculate dataset statistics")

# Overall positivity rate
p_rate = sum(grasp_labels) / len(grasp_labels) * 100
print("Overall positivity rate is {:02f} %".format(p_rate))

# Calculate + plot number of grasp per elevation (phi) angle
generate_phi_overview(phi_angles, grasp_labels, collision_free)
generate_phi_psi_overview(phi_angles, psi_angles, grasp_labels)
generate_phi_beta_overview(phi_angles, beta_angles, grasp_labels)
if COLLISION:
    generate_phi_psi_overview(phi_angles, psi_angles, collision_free, collision=True)
    generate_phi_beta_overview(phi_angles, beta_angles, collision_free, collision=True)

generate_psi_overview(psi_angles, grasp_labels, collision_free)
generate_beta_overview(beta_angles, grasp_labels, collision_free)
