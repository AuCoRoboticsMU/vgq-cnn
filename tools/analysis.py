import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
from autolab_core import Logger, BinaryClassificationResult, YamlConfig
import argparse

from gqcnn.model import get_gqcnn_model
from gqcnn.utils import read_pose_data
from tools.path_settings import EXPER_PATH

"""
Analyse a vgq-cnn network. Will output an analysis over the camera poses by default. Use the flags
to produce extra analysis over psi, beta or objects.
"""

BACKGROUND = True
DEBUG = False

if not BACKGROUND:
    line_color = 'w'
    hist_color = (0.616, 0.773, 0.730)
    transparent = True
else:
    line_color = 'k'
    hist_color = (0.2, 0.6, 1)
    transparent = False

# Matplotlib style setting
plt.rc('axes', edgecolor=line_color)
plt.rc('text', color=line_color)
plt.rc(('xtick', 'ytick'), c=line_color)
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=20)


class VgqcnnAnalyse:
    def __init__(self, camera_analysis, beta_analysis, psi_analysis, object_analysis, verbose=True):
        """
        Initiates the analysis of a VGQ-CNN model.

        Parameters
        ----------
        data_dir (str): Path to dataset on which the model should be evaluated.
        verbose (bool): Whether or not to log initialization output to `stdout`.
        """
        self.metric_thresh = 0.002  # Change metric threshold here if needed!
        self.analyse_checkpoints = False
        self.verbose = verbose
        self.files = None
        self.logger = None
        self.pose_input = None
        self.gripper_mode = None
        self.psi_analysis = psi_analysis
        self.phi_analysis = camera_analysis
        self.d_analysis = camera_analysis
        self.beta_analysis = beta_analysis
        self.store_in_text = True
        self.plot_grasp_num = False
        self.analyse_objects = object_analysis

        self.dist_step = 0.05
        self.dist_bins = np.arange(0.4, 1.1, self.dist_step)
        self.phi_step = 5
        self.phi_bins = np.arange(0, 70, self.phi_step)
        self.psi_step = 5
        self.psi_bins = np.arange(0, 90, self.psi_step)
        self.beta_step = 5
        self.beta_bins = np.arange(0, 90, self.beta_step)

        # Create output dir if it doesn't exist yet
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Set up logger
        self.logger = Logger.get_logger(self.__class__.__name__,
                                        log_file=os.path.join(
                                            output_dir, "analysis.log"),
                                        silence=(not self.verbose),
                                        global_log_file=self.verbose)
        self.logger.info("Saving output to %s" % output_dir)
        self.files = self._get_identifiers()

    def _get_identifiers(self):
        """
        Reads the identifiers of all tensors in one dataset.

        Returns
        -------
        numbers (list): List of strings containing the tensor identifiers of the dataset
        """
        self.logger.info("Loading data from {}".format(data_dir))
        files = os.listdir(data_dir)

        numbers = list(set([string.split('_')[-1][:-4] for string in files if '.npz' in string]))
        numbers.sort()
        return numbers

    def _read_model(self, checkpoint):
        """
        Reads in the model parameters from model dir.

        Parameters
        ----------
        checkpoint (str): Checkpoint to be loaded

        Returns
        -------
        models (gqcnn.model): loaded GQCNN model
        """
        # Determine model name
        model_root, model_n = os.path.split(model_dir)
        self.logger.info("Analyzing model %s" % model_n)
        # Load model.

        if checkpoint == 'final':
            model = get_gqcnn_model(verbose=self.verbose).load(
                model_dir, verbose=self.verbose)
        else:
            model = get_gqcnn_model(verbose=self.verbose).load(
                model_dir, verbose=self.verbose, checkpoint_step=checkpoint)
        model.open_session()

        return model

    @staticmethod
    def _plot_heatmap(data, x_bins, x_step, y_bins, y_step, x_axis,
                      y_axis, title, save_name, unit, color="Greens", v_max=None):
        """
        Plots and saves a heatmap with matplotlib.

        Parameters
        ----------
        data (numpy.ndarray): data to be plotted in the heatmap
        x_bins (numpy.array): data bins for the x-axis, e.g. the camera distance d.
        x_step (str): Step size for the x-axis, e.g. 0.05m.
        y_bins (numpy.array): data bins for the y-axis, e.g. the camera elevation angle phi.
        y_step (str): Step size for the y-axis, e.g. 5degree.
        x_axis (str):  x_axis variable, can be 'psi', 'phi', 'beta'
        y_axis (str): y_axis variable, can be 'psi', 'd', 'beta'
        title (str): Title for heatmap.
        save_name (str): Where to save the image (full path)
        unit (str): Unit to label the color bar with.
        color (str): Color of the color bar, defaults to "Greens".
        v_max (float): Maximum value of the heatmap color bar, defaults to None (== maximum value in data).
        """
        sz_x = len(x_bins)
        grid_x = np.arange(0, x_bins[-1] + 2 * x_step, x_step)
        sz_y = len(y_bins)
        grid_y = np.arange(0, y_bins[-1] + 2 * y_step, y_step)
        if x_axis == 'psi':
            x_label = 'Grasp angle psi \u03C8 [\u00b0]'
        elif x_axis == 'd':
            x_label = 'Grasp distance d'
            grid_x = np.arange(0.4, x_bins[-1] + 2 * x_step, x_step)
        elif x_axis == 'beta':
            x_label = 'Grasp angle beta {} [\u00b0]'.format('\u03B2')
        else:
            raise AttributeError("x_axis '{}' cannot be plotted in mesh".format(x_axis))
        if y_axis == 'psi':
            y_label = 'Grasp angle psi \u03C8 [\u00b0]'
        elif y_axis == 'phi':
            y_label = 'Elevation angle \u03C6 [\u00b0]'
        elif y_axis == 'beta':
            y_label = 'Grasp angle beta {} [\u00b0]'.format('\u03B2')
        else:
            raise AttributeError("y_axis '{}' cannot be plotted in mesh".format(y_axis))

        combined_accuracy_map = np.array(data).reshape((sz_y, sz_x))
        x, y = np.meshgrid(grid_x,
                           grid_y)
        fig, ax = plt.subplots(figsize=(5, 6))
        c = ax.pcolormesh(x, y, combined_accuracy_map,
                          cmap=color,
                          vmin=0,
                          vmax=v_max)
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax, label=unit, orientation="horizontal")
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        plt.savefig(save_name)
        plt.close()

    @staticmethod
    def _plot_line(y_data, x_data, title, x_axis, y_axis, save_name, y_lim=None):
        """
        Plots and saves a line with matplotlib.

        Parameters
        ----------
        y_data (list/numpy.array): data points in y-direction.
        x_data (list/np.array): data points in x-direction. If None, the n y-points are plotted along 1, 2, 3, ... n
        title (str): Title for line plot.
        x_axis (str):  can be 'psi', 'phi', 'beta', 'd'; if not one of these, x_axis will be used as x label
        y_axis (str): can be 'pos', 'coll', 'TPR', 'TNR'; if not one of these, y_axis will be used as y label
        save_name (str): Where to save the image (full path)
        y_lim (tuple): y ranges for line plot, can be overwritten by yaxis for TPR and TNR. If None, matplotlib adjusts
                      it automatically.
        """
        if x_axis == 'phi':
            x_label = '{} [\u00b0]'.format('\u03C6')
        elif x_axis == 'psi':
            x_label = '{} [\u00b0]'.format('\u03C8')
        elif x_axis == 'beta':
            x_label = '{} [\u00b0]'.format('\u03B2')
        elif x_axis == 'd':
            x_label = 'Camera distance d'
        else:
            x_label = x_axis
        if y_axis == 'pos':
            y_label = 'Positivity rate [%]'
        elif y_axis == 'coll':
            y_label = 'Collision rate [%]'
        elif y_axis == 'TPR':
            y_label = 'TPR'
            y_lim = (0, 1)
        elif y_axis == 'TNR':
            y_label = 'TNR'
            y_lim = (0, 1)
        else:
            y_label = y_axis

        if x_data is None:
            plt.plot(y_data, '-', color=hist_color)
        else:
            plt.plot(x_data, y_data, 'x-', color=hist_color)
        plt.title(title)
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.xlabel(x_label, color=line_color)
        plt.ylabel(y_label, color=line_color)
        plt.savefig(save_name, transparent=transparent)
        plt.close()

    @staticmethod
    def _balanced_acc(tpr, tnr):
        """
        Calculates the balanced accuracy as (TPR + TNR) / 2 for tpr and tnr values.

        Parameters
        ----------
        tpr (float/list/np.array): True Positive Rate (TP / NUM_P)
        tnr (float/list/np.array): True Negative Rate (TN / NUM_N)

        Returns
        --------
        bal_acc (np.array): Balanced accuracy
        """
        bal_acc = (np.array(tpr) + np.array(tnr)) / 2.0
        return bal_acc

    def _plot_validation_performance(self):
        """
        Plots a range of validation performance plots over the training iterations. The plots include validation loss,
        validation error, validation TPR, validation TNR, validation balanced accuracy and learning rate. Plots are
        stored according to output_dir as defined in the outer scope of this file.
        """
        val_error = []
        val_loss = []
        val_tpr = []
        val_tnr = []
        val_acc = []
        lr = None
        lr_x = [0]
        x = 0
        with open(model_dir + '/training.log', 'r') as reader:
            lines = reader.read().splitlines()
            for line in lines:
                if 'learning rate' in line:
                    rate = float(line.split(':')[-1])
                    if lr is None:
                        lr = [rate]
                    if lr[-1] == rate:
                        x += 1
                    else:
                        lr.append(lr[-1])
                        lr_x.append(x / 100)
                        lr.append(rate)
                        lr_x.append(x / 100)
                if 'alidation error' in line:
                    val_error.append(float(line.split(':')[-1]))
                if 'alidation loss' in line:
                    val_loss.append(float(line.split(':')[-1]))
                if 'alidation tpr' in line:
                    val_tpr.append(float(line.split(':')[-1]) * 100)
                if 'alidation tnr' in line:
                    val_tnr.append(float(line.split(':')[-1]) * 100)
                if 'alidation acc' in line:
                    val_acc.append(float(line.split(':')[-1]) * 100)

        balanced_acc = self._balanced_acc(tpr=val_tpr, tnr=val_tnr)

        self._plot_line(val_error, None, "Validation error {}".format(model_name),
                        "Training iterations [million]", "Validation error",
                        output_dir + '/val_errors.png')

        self._plot_line(val_loss, None, "Validation loss {}".format(model_name),
                        "Training iterations [million]", "Validation loss",
                        output_dir + '/val_loss.png')

        self._plot_line(balanced_acc, None, "Validation balanced accuracy {}".format(model_name),
                        "Training iterations [million]", "Balanced accuracy [%]",
                        output_dir + '/val_bal_acc.png', y_lim=(0, 100))

        self._plot_line(val_tpr, None, "Validation balanced accuracy {}".format(model_name),
                        "Training iterations [million]", "TPR [%]",
                        output_dir + '/val_tpr.png', y_lim=(0, 100))

        self._plot_line(val_tnr, None, "Validation TNR {}".format(model_name),
                        "Training iterations [million]", "TNR [%]",
                        output_dir + '/val_tnr.png', y_lim=(0, 100))

        self._plot_line(val_acc, None, "Validation accuracy {}".format(model_name),
                        "Training iterations [million]", "Accuracy [%]",
                        output_dir + '/val_acc.png', y_lim=(0, 100))

        self._plot_line(lr_x, lr, "Learning rate {}".format(model_name),
                        "Training iterations [million]", "Learning rate",
                        output_dir + '/learning_rate.png', y_lim=(0, lr[0]*1.05))

    @staticmethod
    def _get_accuracies(results, param, bin_1, bin_2=None):
        """
        Calculates the TPR, TNR, balanced accuracy and accuracy based on the prediction results

        Parameters
        ----------
        results (dict): Dictoonary of the prediction results after binary classification
        param (str): defining which parameter bin_1 (and bin_2) relates to. Can be 'phi', 'beta', 'phi_dist', etc.
        bin_1 (float): bin for param, e.g. 5[degree] for 'beta'
        bin_2 (float): 2nd bin for param, e.g. 0.05[m] for the distance in 'phi_dist'

        Returns
        --------
        tpr (np.array): True Positive Rate, calculated as TP / NUM_P
        tnr (np.array): True Negative Rate, calculated as TN / NUM_N
        bal_acc (np.array): Balanced accuracy, calculated as (TPR + TNR) / 2
        acc (np.array): Accuracy, calculated as (TP + TN) / (NUM_P + NUM_N)
        """
        if bin_2 is None:
            tp = sum(results['tp'][param][bin_1])
            tn = sum(results['tn'][param][bin_1])
            num_n = sum(results['num_n'][param][bin_1])
            num_p = sum(results['num_p'][param][bin_1])
        else:
            tp = sum(results['tp'][param][bin_1][bin_2])
            tn = sum(results['tn'][param][bin_1][bin_2])
            num_n = sum(results['num_n'][param][bin_1][bin_2])
            num_p = sum(results['num_p'][param][bin_1][bin_2])

        tpr = tp / num_p
        tnr = tn / num_n
        bal_acc = (tpr + tnr) / 2
        acc = (tn + tp) / (num_p + num_n)
        return tpr, tnr, bal_acc, acc

    def _get_acc_range(self, results, param, bins, bins_2=None):
        """
        Generates an array of accuracy metrics over parameter ranges, e.g. over the grasp-table angle beta.

        Parameters
        ----------
        results (dict): Dictionary of the prediction results after binary classification
        param (str): defining which parameter bins to calculate. Can be 'phi', 'beta', 'psi', 'phi_dist', etc.
        bins (np.array): All bins for the (first) parameter, e.g. 'beta' with [0, 5, 10, 15, ...] degree
        bins_2 (np.array): All bins for the second parameter, e.g. distance d for 'phi_dist'. Defaults to None
                           for single parameter evaluations.

        Returns
        --------
        arr_tpr (np.array): True Positive Rate, calculated as TP / NUM_P
        arr_tnr (np.array): True Negative Rate, calculated as TN / NUM_N
        arr_bal_acc (np.array): Balanced accuracy, calculated as (TPR + TNR) / 2
        arr_acc (np.array): Accuracy, calculated as (TP + TN) / (NUM_P + NUM_N)
        """
        arr_tpr = []
        arr_tnr = []
        arr_bal_acc = []
        arr_acc = []
        for bin1 in bins:
            if bins_2 is None:
                try:
                    tpr, tnr, bal_acc, acc = self._get_accuracies(results, param, bin1)
                    arr_tpr.append(tpr)
                    arr_tnr.append(tnr)
                    arr_bal_acc.append(bal_acc)
                    arr_acc.append(acc)
                except ZeroDivisionError:
                    arr_tpr.append(0.0)
                    arr_tnr.append(0.0)
                    arr_bal_acc.append(0.0)
                    arr_acc.append(0.0)
            else:
                for bin2 in bins_2:
                    try:
                        tpr, tnr, bal_acc, acc = self._get_accuracies(results, param, bin1, bin2)
                        arr_tpr.append(tpr)
                        arr_tnr.append(tnr)
                        arr_bal_acc.append(bal_acc)
                        arr_acc.append(acc)
                    except ZeroDivisionError:
                        arr_tpr.append(0.0)
                        arr_tnr.append(0.0)
                        arr_bal_acc.append(0.0)
                        arr_acc.append(0.0)
        return arr_tpr, arr_tnr, arr_bal_acc, arr_acc

    @staticmethod
    def _run_binary_classification(predictions, labels):
        """
        Runs binary classification on a set of predictions and labels. Returns 0.0 for all metrics
        [tp, tn, num_p, num_n] if predictions is None.

        Parameters
        ----------
        predictions (numpy.array): Predictions of the classification model, can be a vector or a [n, 2] matrix.
                                   If [n, 2] matrix, the second column is assumed to be the prediction, alongside
                                   the definition during GQ-CNN and VGQ-CNN training.
        labels (numpy.array): Array of binary class labels, with 1 for positive grasps and 0 for negative grasps.

        Returns
        --------
        tp (int): Number of true positive grasps, e.g. ground-truth positive grasps that were classified as positive
        tn (int): Number of true negative grasps, e.g. ground-truth negative grasps that were classified as negative
        num_p (int): Number of ground-truth positive grasps
        num_n (int): Number of ground-truth negative grasps
        """
        if predictions is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        num_p = sum(labels)
        num_n = len(labels) - sum(labels)
        try:
            results = BinaryClassificationResult(predictions[:, 1], labels)
            tp = len(results.true_positive_indices)
            tn = len(results.true_negative_indices)
        except IndexError:
            results = BinaryClassificationResult(predictions, labels)
            tp = sum(results.predictions[labels == 1])
            tn = len(results.predictions[labels == 0]) - sum(results.predictions[labels == 0])

            if type(tp) is not int:
                tp = tp.squeeze()
            if type(tn) is not int:
                tn = tn.squeeze()

        return tp, tn, num_p, num_n

    @staticmethod
    def _save_results(save_dir, data):
        """
        Generates a text file from an array of results in percentage.

        Parameters
        ----------
        save_dir (str): Full path of the text file.
        data (numpy.array): results stored in the format of an array. Will be converted to [%].
        """
        with open(save_dir, 'w') as f:
            np.savetxt(f, data * 100, '%.1f')

    def _run_prediction(self, model):
        """
        Predicts the data with the loaded model and plots a box-plot to show TPR and TNR performance over the objects.

        Parameters
        ----------
        model (gqcnn.model): Model to be used for prediction.

        Returns
        ----------
        results (dict):             Dictionary with True Positives, True Negatives, Ground-truth Negatives and
                                    Ground-truth Positives after prediction and binary classification
        object_predictions (dict):  Dictionary with the object labels, ground truth grasp labels and model
                                    predictions for object-wise analysis. If self.analyse_objects = False, an empty
                                    dictionary will be returned.
        """
        results = {'tp': {'all': [], 'phi': {}, 'psi': {}, 'beta': {}, 'dist': {}, 'phi_dist': {}},
                   'tn': {'all': [], 'phi': {}, 'psi': {}, 'beta': {}, 'dist': {}, 'phi_dist': {}},
                   'num_n': {'all': [], 'phi': {}, 'psi': {}, 'beta': {}, 'dist': {}, 'phi_dist': {}},
                   'num_p': {'all': [], 'phi': {}, 'psi': {}, 'beta': {}, 'dist': {}, 'phi_dist': {}}}

        res = ['tp', 'tn', 'num_p', 'num_n']
        for result in res:
            results[result]['all'] = []
            for _phi in self.phi_bins:
                results[result]['phi'][_phi] = []
            for _psi in self.psi_bins:
                results[result]['psi'][_psi] = []
            for _beta in self.beta_bins:
                results[result]['beta'][_beta] = []
            for _dist in self.dist_bins:
                results[result]['dist'][_dist] = []
            for _phi in self.phi_bins:
                results[result]['phi_dist'][_phi] = {}
                for _dist in self.dist_bins:
                    results[result]['phi_dist'][_phi][_dist] = []

        object_predictions = {'predictions': [], 'labels': [], 'objects': []}

        step_size = 25
        num_files = len(self.files)
        if DEBUG:
            num_files = 1
            step_size = 1

        # Read and predict data
        for steps in range(0, num_files, step_size):
            self.logger.info("Read in tensors %d to %d of %d" % (steps, steps + step_size, num_files))
            images, poses, all_labels, elev_arr, radii, psi_arr, beta_arr, objects = self._read_data(steps, step_size)
            predictions = model.predict(images, poses)

            tp, tn, num_p, num_n = self._run_binary_classification(predictions, all_labels)
            results['tp']['all'].append(tp)
            results['tn']['all'].append(tn)
            results['num_p']['all'].append(num_p)
            results['num_n']['all'].append(num_n)

            if self.analyse_objects:
                object_predictions['predictions'].extend(predictions)
                object_predictions['labels'].extend(all_labels)
                object_predictions['objects'].extend(list(objects))

            if self.phi_analysis:
                for _phi in self.phi_bins:
                    mask = (elev_arr.squeeze() >= _phi) & (elev_arr.squeeze() < _phi + self.phi_step)
                    phi_predictions = predictions[mask]
                    phi_labels = all_labels[mask]
                    tp, tn, num_p, num_n = self._run_binary_classification(phi_predictions, phi_labels)
                    results['tp']['phi'][_phi].append(tp)
                    results['tn']['phi'][_phi].append(tn)
                    results['num_p']['phi'][_phi].append(num_p)
                    results['num_n']['phi'][_phi].append(num_n)

            if self.beta_analysis:
                for _beta in self.beta_bins:
                    mask = (beta_arr.squeeze() >= _beta) & (beta_arr.squeeze() < _beta + self.beta_step)
                    beta_predictions = predictions[mask]
                    beta_labels = all_labels[mask]
                    tp, tn, num_p, num_n = self._run_binary_classification(beta_predictions, beta_labels)
                    results['tp']['beta'][_beta].append(tp)
                    results['tn']['beta'][_beta].append(tn)
                    results['num_p']['beta'][_beta].append(num_p)
                    results['num_n']['beta'][_beta].append(num_n)

            if self.psi_analysis:
                for _psi in self.psi_bins:
                    mask = (psi_arr.squeeze() >= _psi) & (psi_arr.squeeze() < _psi + self.psi_step)
                    psi_predictions = predictions[mask]
                    psi_labels = all_labels[mask]

                    tp, tn, num_p, num_n = self._run_binary_classification(psi_predictions, psi_labels)
                    results['tp']['psi'][_psi].append(tp)
                    results['tn']['psi'][_psi].append(tn)
                    results['num_p']['psi'][_psi].append(num_p)
                    results['num_n']['psi'][_psi].append(num_n)

            if self.d_analysis:
                for _dist in self.dist_bins:
                    mask = (radii.squeeze() >= _dist) & (radii.squeeze() < _dist + self.dist_step)
                    dist_predictions = predictions[mask]
                    dist_labels = all_labels[mask]

                    tp, tn, num_p, num_n = self._run_binary_classification(dist_predictions, dist_labels)
                    results['tp']['dist'][_dist].append(tp)
                    results['tn']['dist'][_dist].append(tn)
                    results['num_p']['dist'][_dist].append(num_p)
                    results['num_n']['dist'][_dist].append(num_n)

            if self.d_analysis and self.phi_analysis:
                for _phi in self.phi_bins:
                    for _dist in self.dist_bins:
                        mask = (radii.squeeze() >= _dist) & \
                               (radii.squeeze() < _dist + self.dist_step) & \
                               (elev_arr.squeeze() >= _phi) & \
                               (elev_arr.squeeze() < _phi + self.phi_step)
                        phi_dist_predictions = predictions[mask]
                        phi_dist_labels = all_labels[mask]
                        tp, tn, num_p, num_n = self._run_binary_classification(phi_dist_predictions, phi_dist_labels)
                        results['tp']['phi_dist'][_phi][_dist].append(tp)
                        results['tn']['phi_dist'][_phi][_dist].append(tn)
                        results['num_p']['phi_dist'][_phi][_dist].append(num_p)
                        results['num_n']['phi_dist'][_phi][_dist].append(num_n)

        return results, object_predictions

    @staticmethod
    def _analyse_performance_per_object(object_predictions):
        """
        Runs binary classification and plots a box-plot to show TPR and TNR performance over the objects.

        Parameters
        ----------
        object_predictions (dict): Object labels, grasp ground-truth labels and model predictions of model
        """
        object_labels = list(set(list(object_predictions['objects'])))
        object_predictions['objects'] = np.array(object_predictions['objects'])
        object_predictions['labels'] = np.array(object_predictions['labels'])
        object_predictions['predictions'] = np.array(object_predictions['predictions'])[:, 1]
        tpr = []
        tnr = []
        for object_label in object_labels:
            in_bin = (list(object_predictions['objects']) == object_label)
            labels = object_predictions['labels'][in_bin]
            obj_predictions = np.where(object_predictions['predictions'][in_bin] >= 0.5, 1, 0)

            positives = (labels == 1)
            pos_predictions = obj_predictions[positives]
            if len(pos_predictions) != 0:
                tpr.append(sum(pos_predictions) / len(pos_predictions))

            negatives = (labels == 0)
            neg_predictions = obj_predictions[negatives]
            if len(neg_predictions) != 0:
                tnr.append((len(neg_predictions) - sum(neg_predictions)) / len(neg_predictions))

        plt.boxplot([tpr, tnr], labels=["TPR", "TNR"])
        plt.ylim((0.0, 1.0))
        plt.title("TPR and TNR per object")
        plt.savefig(output_dir + '/object_tpr_tnr.png')
        plt.close()

    def _load_model_and_plot_training_stats(self):
        """
        Loads the model and plots the training stats, including learning rate, validation loss, validation errors, ...

        Returns
        --------
        model (gqcnn.model): Loaded model with the highest balanced accuracy on the validation set during training.
        """
        model_config = YamlConfig(model_dir + '/config.json')
        self.gripper_mode = model_config['gqcnn']['gripper_mode']
        if 'pose_input' in model_config['gqcnn']:
            self.pose_input = model_config['gqcnn']['pose_input']
            if self.pose_input['quaternion'] == 0 and self.psi_analysis:
                self.psi_analysis = False

        self._plot_validation_performance()
        model = self._read_model('final')
        return model

    def visualise(self):
        """
        Evaluates the model on the dataset in self.datadir. Plots and saves the resulting classification accuracies.

        """
        model = self._load_model_and_plot_training_stats()
        results, object_predictions = self._run_prediction(model)

        # Calculate prediction accuracy for all models and all elevation (phi) angles

        if sum(results['num_p']['all']) > 0:
            tpr = sum(results['tp']['all']) / sum(results['num_p']['all']) * 100
        else:
            tpr = 0.0
        if sum(results['num_n']['all']) > 0:
            tnr = sum(results['tn']['all']) / sum(results['num_n']['all']) * 100
        else:
            tnr = 0.0
        self.logger.info("TPR: %.1f %%" % tpr)
        self.logger.info("TNR: %.1f %%" % tnr)
        self.logger.info("Balanced accuracy: %.1f %%" % ((tpr + tnr) / 2))

        if self.analyse_objects:
            self._analyse_performance_per_object(object_predictions)

        if self.phi_analysis:
            phi_tpr, phi_tnr, _, _ = self._get_acc_range(results, 'phi', self.phi_bins)
            self._plot_line(x_data=self.phi_bins + self.phi_step / 2, y_data=phi_tpr,
                            title="TPR over varying elevation angle",
                            x_axis='phi',
                            y_axis='TPR',
                            save_name=output_dir + '/phi_tpr.png')

            self._plot_line(x_data=self.phi_bins + self.phi_step / 2, y_data=phi_tnr,
                            title="TNR over varying elevation angle",
                            x_axis='phi',
                            y_axis='TNR',
                            save_name=output_dir + '/phi_tnr.png')
            if self.store_in_text:
                self._save_results("{}/phi_tpr.txt".format(output_dir), phi_tpr)
                self._save_results("{}/phi_tnr.txt".format(output_dir), phi_tnr)

        if self.beta_analysis:
            beta_tpr, beta_tnr, _, _ = self._get_acc_range(results, 'beta', self.beta_bins)
            self._plot_line(x_data=self.beta_bins + self.beta_step / 2, y_data=beta_tpr,
                            title="TPR over varying grasp/table angle",
                            x_axis='beta',
                            y_axis='TPR',
                            save_name=output_dir + '/beta_tpr.png')

            self._plot_line(x_data=self.beta_bins + self.beta_step / 2, y_data=beta_tnr,
                            title="TNR over varying grasp_table angle",
                            x_axis='beta',
                            y_axis='TNR',
                            save_name=output_dir + '/beta_tnr.png')
            if self.store_in_text:
                self._save_results("{}/beta_tpr.txt".format(output_dir), beta_tpr)
                self._save_results("{}/beta_tnr.txt".format(output_dir), beta_tnr)

        if self.d_analysis:
            dist_tpr, dist_tnr, _, _ = self._get_acc_range(results, 'dist', self.dist_bins)
            self._plot_line(x_data=self.dist_bins + self.dist_step / 2, y_data=dist_tpr,
                            title="TPR over varying camera radii",
                            x_axis='d',
                            y_axis='TPR',
                            save_name=output_dir + '/rad_tpr.png')

            self._plot_line(x_data=self.dist_bins + self.dist_step / 2, y_data=dist_tnr,
                            title="TNR over varying camera radii",
                            x_axis='d',
                            y_axis='TNR',
                            save_name=output_dir + '/rad_tnr.png')
            if self.store_in_text:
                self._save_results("{}/d_tpr.txt".format(output_dir), dist_tpr)
                self._save_results("{}/d_tnr.txt".format(output_dir), dist_tnr)

        if self.psi_analysis:
            psi_tpr, psi_tnr, _, _ = self._get_acc_range(results, 'psi', self.psi_bins)
            self._plot_line(x_data=self.psi_bins + self.psi_step / 2, y_data=psi_tpr,
                            title="TPR over varying grasp angle",
                            x_axis='psi',
                            y_axis='TPR',
                            save_name=output_dir + '/psi_tpr.png')

            self._plot_line(x_data=self.psi_bins + self.psi_step / 2, y_data=psi_tnr,
                            title="TNR over varying grasp angle",
                            x_axis='psi',
                            y_axis='TNR',
                            save_name=output_dir + '/psi_tnr.png')
            if self.store_in_text:
                self._save_results("{}/psi_tpr.txt".format(output_dir), psi_tpr)
                self._save_results("{}/psi_tnr.txt".format(output_dir), psi_tnr)

        if self.phi_analysis and self.d_analysis:
            phi_d_tpr, phi_d_tnr, _, _ = self._get_acc_range(results, 'phi_dist', self.phi_bins, self.dist_bins)

            if self.store_in_text:
                self._save_results("{}/phi_d_tpr.txt".format(output_dir), phi_d_tpr)
                self._save_results("{}/phi_d_tnr.txt".format(output_dir), phi_d_tnr)

            tnr = np.array(phi_d_tnr) * 100
            tpr = np.array(phi_d_tpr) * 100
            self.logger.info("Phi-d TNR min: {:.1f}%".format(min(tnr)))
            self.logger.info("Phi-d TNR max: {:.1f}%".format(max(tnr)))
            self.logger.info("Phi-d TNR mean: {:.1f}% +/- {:.1f}%".format(tnr.mean(), tnr.std()))
            self.logger.info("Phi-d TPR min: {:.1f}%".format(min(tpr)))
            self.logger.info("Phi-d TPR max: {:.1f}%".format(max(tpr)))
            self.logger.info("Phi-d TPR mean: {:.1f}% +/- {:.1f}%".format(tpr.mean(), tpr.std()))

            dset_name = data_dir.split('/')[-3]
            title = 'TPR {} on {}'.format(model_name, dset_name)
            save_name = output_dir + '/phi_d_tpr_{}.png'.format(model_name)
            self._plot_heatmap(data=phi_d_tpr, x_bins=self.dist_bins, x_step=self.dist_step,
                               y_bins=self.phi_bins, y_step=self.phi_step, x_axis='d', y_axis='phi', title=title,
                               save_name=save_name, unit="TPR", v_max=1)

            title = 'TNR {} on {}'.format(model_name, dset_name)
            save_name = output_dir + '/phi_d_tnr_{}.png'.format(model_name)
            self._plot_heatmap(data=phi_d_tnr, x_bins=self.dist_bins, x_step=self.dist_step,
                               y_bins=self.phi_bins, y_step=self.phi_step, x_axis='d', y_axis='phi', title=title,
                               save_name=save_name, unit="TNR", v_max=1, color="Blues")

    def _read_data(self, steps, step_size):
        """Reads data from a tensor dataset.

        Parameters
        ----------
            steps (int): Tensor to start reading data from.
            step_size (int): Amount of tensors to read in one batch. All tensors from steps:steps+step_size are read.

        Returns
        ----------
            images (np.array): Depth images.
            poses (np.array): Pose information in format for network-input.
            labels (np.array): Binary ground-truth labels of grasps [0/1].
            phi (np.array): Elevation angle phi of camera for images.
            dist (np.array): Camera distance to T^w origin for images.
            psi (np.array): Angle between grasp approach axis (z) and camera principal ray, psi for all grasps.
            beta (np.array): Angle between grasp approach axis (z) and table normal (T^w z), beta, for all grasps.
            object_labels (np.array): Counter for object identifier for all grasps.
        """
        cam_str = 'camera_poses_'
        depth_str = 'depth_ims_tf_table_'
        pose_str = 'hand_poses_'
        metric_str = 'robust_ferrari_canny_'
        object_str = 'object_labels_'

        phi = None
        object_labels = None
        dist = None
        psi = None
        beta = None

        files = self.files[steps:steps+step_size]

        images = list(np.load(data_dir + depth_str + files[0] + ".npz")['arr_0'])
        pose_arr = list(np.load(data_dir + pose_str + files[0] + ".npz")['arr_0'])
        metric_arr = list(np.load(data_dir + metric_str + files[0] + ".npz")['arr_0'])
        if self.analyse_objects:
            object_labels = list(np.load("{}{}{}.npz".format(data_dir, object_str, files[0]))['arr_0'])
        if self.d_analysis or self.phi_analysis:
            cameras = np.load(data_dir + cam_str + files[0] + ".npz")['arr_0']
            dist = list(cameras[:, 0])
            phi = list(cameras[:, 1])

        for next_file in files[1:]:
            images.extend(list(np.load(data_dir + depth_str + next_file + ".npz")['arr_0']))
            pose_arr.extend(list(np.load(data_dir + pose_str + next_file + ".npz")['arr_0']))
            metric_arr.extend(list(np.load(data_dir + metric_str + next_file + ".npz")['arr_0']))
            if self.analyse_objects:
                object_labels.extend(list(np.load("{}{}{}.npz".format(data_dir, object_str, next_file))['arr_0']))

            if self.d_analysis or self.phi_analysis:
                cameras = np.load(data_dir + cam_str + next_file + ".npz")['arr_0']
                dist.extend(list(cameras[:, 0]))
                phi.extend(list(cameras[:, 1]))

        labels = 1 * (np.array(metric_arr) > self.metric_thresh)
        images = np.array(images)
        pose_arr = np.array(pose_arr)
        poses = read_pose_data(pose_arr, self.gripper_mode, self.pose_input)

        if self.analyse_objects:
            object_labels = np.array(object_labels)
        if self.psi_analysis:
            psi = pose_arr[:, -2].squeeze()
            psi = np.rad2deg(psi)
        if self.beta_analysis:
            beta = np.rad2deg(pose_arr[:, -1].squeeze())
        if self.d_analysis or self.phi_analysis:
            phi = np.rad2deg(np.array(phi))
            dist = np.array(dist)

        return images, poses, labels, phi, dist, psi, beta, object_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a GQCNN with Tensorflow on single data")
    parser.add_argument("model_name",
                        type=str,
                        default=None,
                        help="name of model to analyse")
    parser.add_argument("data_dir",
                        type=str,
                        default=None,
                        help="path to where the data is stored")
    parser.add_argument("--camera_pose",
                        type=bool,
                        default=True,
                        help="analyse performance over camera pose")
    parser.add_argument("--beta",
                        type=bool,
                        default=False,
                        help="analyse performance over beta")
    parser.add_argument("--psi",
                        type=bool,
                        default=False,
                        help="analyse performance over psi")
    parser.add_argument("--objects",
                        type=bool,
                        default=False,
                        help="analyse performance over objects")

    args = parser.parse_args()
    model_name = args.model_name
    data_dir = args.data_dir
    analyse_pose = args.camera_pose
    analyse_beta = args.beta
    analyse_psi = args.psi
    analyse_objects = args.objects

    # Create model dir.
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models")
    model_dir = os.path.join(model_dir, model_name)

    # Make the output dir.
    try:
        output_dir = EXPER_PATH + model_name + '_on_{}/'.format(data_dir.split('/')[2])
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    except FileNotFoundError:
        output_dir = '/results/' + model_name + '_on_{}/'.format(data_dir.split('/')[2])
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # Initialise analyser and run analysis.
    analyser = VgqcnnAnalyse(analyse_pose, analyse_beta, analyse_psi, analyse_objects)
    analyser.visualise()
