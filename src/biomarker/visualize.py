import os
import uuid
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from nibabel.affines import apply_affine
from nilearn.plotting import plot_connectome, plot_stat_map, plot_roi


__dir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tempdata")


class PowerCrossleyVisualizer:
    def __init__(self, img_file, bg_file, roi_file, power_path, crossley_path):
        self._node_coords = self._get_coords(power_path)
        self._map_power_to_crossley_atlas(self._node_coords, crossley_path)

        self._K = len(self._node_coords)
        self._img_file = img_file
        self._bg_file = bg_file
        self._roi_file = roi_file

    def _get_coords(self, node_coords_path):
        node_coords_df = pd.read_csv(node_coords_path)
        node_coords = np.array(node_coords_df[["X", "Y", "Z"]])
        return node_coords

    def _map_power_to_crossley_atlas(self, node_coords, crossley_path):
        crossley_df = pd.read_csv(crossley_path)
        crossley_xyz = np.array(crossley_df[["X", "Y", "Z"]])
        crossley_labels = np.array(crossley_df["crossley_labels"])

        similarity = cdist(node_coords, crossley_xyz, "euclidean")
        closest_index = np.argmin(similarity, axis=1)
        node_labels = crossley_labels[closest_index]

        # to preserve the order as in original code
        module_labels, idx, module_frequencies = np.unique(
            node_labels, return_counts=True, return_index=True
        )
        idx = sorted(range(len(idx)), key=lambda i: idx[i])
        module_labels = module_labels[idx]
        module_frequencies = module_frequencies[idx]
        # # alternative:
        # module_labels, module_frequencies = np.unique(
        #     node_labels, return_counts=True
        # )

        sorted_modules_index = np.argsort(-module_frequencies, kind="mergesort")
        sorted_module_labels = module_labels[sorted_modules_index]
        self._module_bounds = np.cumsum(module_frequencies[sorted_modules_index]) - 1

        node_to_module_index = np.zeros(len(node_labels), dtype=int)
        for i in range(len(sorted_module_labels)):
            is_cur_module = node_labels == sorted_module_labels[i]
            node_to_module_index[is_cur_module] = i

        # to preserve the order as in original code
        sorted_node_index = np.array(
            sorted(
                range(len(node_to_module_index)),
                key=lambda i: (node_to_module_index[i], i),
            )
        )
        # # alternative:
        # sorted_node_index = np.argsort(node_to_module_index)

        sorted_node_labels = node_labels[sorted_node_index]
        self._module_labels = sorted_module_labels
        self._rarrange_node_labels = sorted_node_labels
        self._node_labels = node_labels
        self._rearrange_idx = sorted_node_index

    def plot_connectome(self, score_matrix, output_file):
        if score_matrix.shape != (self._K, self._K):
            raise TypeError(
                "expected score_matrix of shape {}, but got shape {} instead".format(
                    (self._K, self._K), score_matrix.shape
                )
            )
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return plot_connectome(
            score_matrix ** 2,
            self._node_coords,
            node_size=0,
            edge_cmap="YlOrBr",
            output_file=output_file,
            colorbar=False,
            edge_threshold="99.9%",
        )

    def _remove_symmetry(self, score_matrix):
        if score_matrix.shape != (self._K, self._K):
            raise TypeError(
                "expected score_matrix of shape {}, but got shape {} instead".format(
                    (self._K, self._K), score_matrix.shape
                )
            )
        new_matrix = np.zeros_like(score_matrix)
        idx = np.triu_indices(self._K, 1)
        new_matrix[idx[0], idx[1]] = score_matrix[idx[0], idx[1]]
        return new_matrix

    def _get_nodal_sensitivity(self, score_matrix, axis=0):
        """
        score_matrix: np.ndarray (K, K)
        axis: int
            0 means column-wise importance, 1 means row-wise importance
        nodal_sensitivity: np.ndarray (K,)
        """
        # score_matrix = self._remove_symmetry(score_matrix)
        nodal_sensitivity = np.sum(np.abs(score_matrix), axis=axis)
        return nodal_sensitivity

    @staticmethod
    def _get_threshold(data, threshold):
        sorted_graph_data = np.sort(data[data > 0], axis=None)
        boundary_element_index = np.floor(sorted_graph_data.size * (1 - threshold))
        threshold = sorted_graph_data[int(boundary_element_index)]
        return threshold

    @staticmethod
    def _convert_mni_to_ijk(affine, mni_coords):
        ijk = apply_affine(np.linalg.inv(affine), mni_coords)
        ijk = np.round(ijk).astype(int)
        return ijk

    @staticmethod
    def _is_valid_voxel(matrix_template, v):
        if np.count_nonzero((v < 0) | (v >= matrix_template.shape[:3])) > 0:
            return False
        if np.count_nonzero(matrix_template[tuple(v)]) == 0:
            return False
        return True

    @classmethod
    def _get_valid_coords_in_box(cls, matrix_template, ijk_coord, dv_in_box):
        voxels_coords = dv_in_box + ijk_coord
        valid_coords_in_box = [
            v for v in voxels_coords if cls._is_valid_voxel(matrix_template, v)
        ]
        return np.array(valid_coords_in_box)

    def _power_sphere(self, nodal_sensitivity, filename):
        if not filename.endswith(".nii"):
            filename = filename + ".nii"

        img = nib.load(self._img_file)
        affine = img.affine
        matrix_template = img.get_data()

        var_range = [-1, 0, 1]
        dv_in_box = np.stack(
            np.meshgrid(var_range, var_range, var_range), axis=3
        ).reshape(-1, 3)[:, [1, 0, 2]]

        sensitivity_matrix = np.zeros(matrix_template.shape[:3])
        power_coords_ijk = self._convert_mni_to_ijk(affine, self._node_coords)
        for k in range(self._K):
            voxel_coords = self._get_valid_coords_in_box(
                matrix_template, power_coords_ijk[k], dv_in_box
            )
            for v1, v2, v3 in voxel_coords:
                try:
                    sensitivity_matrix[v1, v2, v3] = nodal_sensitivity[k]
                except:
                    continue

        array_img = nib.Nifti1Image(sensitivity_matrix, affine)
        nib.save(array_img, filename)

    def _prepare_sensitivity_matrix(self, score_matrix, sen_file_path, threshold=0.1):
        nodal_sensitivity = self._get_nodal_sensitivity(score_matrix, axis=0)
        t = self._get_threshold(nodal_sensitivity, threshold)
        final_sensitivity = nodal_sensitivity * (nodal_sensitivity > t)
        self._power_sphere(final_sensitivity, sen_file_path)

    def plot_stat_map(
        self, score_matrix, output_dir, output_prefix="", threshold=0.1, vmax=None
    ):
        if not os.path.exists("/tmp"):
            os.makedirs("/tmp")
        sen_file_path = "/tmp/temp_{}.nii".format(uuid.uuid4().hex)

        try:
            self._prepare_sensitivity_matrix(score_matrix, sen_file_path, threshold)

            output_dir = os.path.abspath(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            prefix = output_prefix + "_" if output_prefix else ""

            x = plot_stat_map(
                stat_map_img=sen_file_path,
                bg_img=self._bg_file,
                colorbar=True,
                output_file=os.path.join(
                    output_dir, "{}sagittal_t_{}.png".format(prefix, threshold)
                ),
                display_mode="x",
                cut_coords=7,
                draw_cross=False,
                black_bg=False,
                vmax=vmax,
            )
            y = plot_stat_map(
                stat_map_img=sen_file_path,
                bg_img=self._bg_file,
                colorbar=True,
                output_file=os.path.join(
                    output_dir, "{}coronal_t_{}.png".format(prefix, threshold)
                ),
                display_mode="y",
                cut_coords=7,
                draw_cross=False,
                black_bg=False,
                vmax=vmax,
            )
            z = plot_stat_map(
                stat_map_img=sen_file_path,
                bg_img=self._bg_file,
                colorbar=True,
                output_file=os.path.join(
                    output_dir, "{}axial_t_{}.png".format(prefix, threshold)
                ),
                display_mode="z",
                cut_coords=7,
                draw_cross=False,
                black_bg=False,
                vmax=vmax,
            )
            os.remove(sen_file_path)
            return x, y, z

        except Exception as e:
            if os.path.exists(sen_file_path):
                os.remove(sen_file_path)
            raise e

    def plot_roi(self, output_file):
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return plot_roi(
            roi_img=self._roi_file,
            bg_img=self._bg_file,
            cmap="Paired",
            output_file=output_file,
        )

    @staticmethod
    def _mpl_savefig(f, output_file):
        if output_file is None:
            return
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        f.savefig(output_file, bbox_inches="tight", dpi=200)

    def plot_module_importance_boxplot(
        self, score_matrix, output_file, percentage_modules_to_plot=1.0
    ):
        num_modules = len(self._module_labels)
        num_modules_to_plot = int(
            round(percentage_modules_to_plot * len(self._module_labels))
        )
        nodal_sen = self._get_nodal_sensitivity(score_matrix, axis=1)

        module_sen = dict()
        module_average = np.zeros(num_modules)
        for i, module in enumerate(self._module_labels):
            module_sen[module] = nodal_sen[self._node_labels == module]
            module_average[i] = np.mean(module_sen[module])

        sort_indices = np.argsort(-module_average)
        sorted_modules = self._module_labels[sort_indices]
        sorted_modules = sorted_modules[:num_modules_to_plot]
        box_plot_array = [module_sen[module] for module in sorted_modules]

        f = plt.figure()
        red_square = dict(markerfacecolor="r", marker="s")
        plt.boxplot(
            box_plot_array, meanline=True, showmeans=True, flierprops=red_square
        )
        plt.ylabel("Importance Score", fontsize=12)
        plt.xticks(
            np.arange(num_modules_to_plot) + 1, sorted_modules, fontsize=11, rotation=90
        )
        self._mpl_savefig(f, output_file)
        return f

    def _rearrange_score_matrix(self, matrix):
        new_matrix = matrix.copy()
        new_matrix = new_matrix[self._rearrange_idx, :]
        new_matrix = new_matrix[:, self._rearrange_idx]
        return new_matrix

    def plot_complete_score_matrix(self, score_matrix, output_file):
        new_matrix = self._rearrange_score_matrix(score_matrix)
        f = plt.figure()
        plt.imshow(new_matrix.astype(float), cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.xticks(self._module_bounds, self._module_labels, rotation=90, fontsize=8)
        plt.yticks(self._module_bounds, self._module_labels, rotation=0, fontsize=8)
        self._mpl_savefig(f, output_file)
        return f

    def _get_module_sensitivity_map(self, score_matrix):
        score_matrix = np.abs(score_matrix.astype(float))
        num_clusters = len(self._module_labels)
        msm = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(i, num_clusters):
                module_i, module_j = self._module_labels[i], self._module_labels[j]
                nodes_i = np.argwhere(self._node_labels == module_i).flatten()
                nodes_j = np.argwhere(self._node_labels == module_j).flatten()
                connections = score_matrix[nodes_i, :][:, nodes_j]
                msm[i, j] = msm[j, i] = np.mean(connections)
        return msm

    def plot_module_sensitivity_map(self, score_matrix, output_file, vmax=None):
        msm = self._get_module_sensitivity_map(score_matrix)
        f = plt.figure()
        plt.imshow(msm.astype(float), cmap="hot", interpolation="nearest", vmax=vmax)
        plt.colorbar()
        ticks = np.arange(len(self._module_labels))
        plt.xticks(ticks, self._module_labels, rotation=90, fontsize=9, weight="bold")
        plt.yticks(ticks, self._module_labels, rotation=0, fontsize=9, weight="bold")
        self._mpl_savefig(f, output_file)
        return f


class ABIDEBiomarkersVisualizer(PowerCrossleyVisualizer):
    def __init__(self):
        img_file = os.path.join(__dir__, "ABIDE_img_file.nii.gz")
        bg_file = os.path.join(__dir__, "ABIDE_bg_file.nii.gz")
        roi_file = os.path.join(__dir__, "ABIDE_roi_file.nii.gz")
        power_path = os.path.join(__dir__, "power_264_xyz.csv")
        crossley_path = os.path.join(__dir__, "crossley_638_xyz.csv")
        super().__init__(img_file, bg_file, roi_file, power_path, crossley_path)


class ADHDBiomarkersVisualizer(PowerCrossleyVisualizer):
    def __init__(self):
        img_file = os.path.join(__dir__, "ADHD_img_file.nii.gz")
        bg_file = os.path.join(__dir__, "ADHD_bg_file.nii.gz")
        roi_file = os.path.join(__dir__, "ADHD_roi_file.nii.gz")
        power_path = os.path.join(__dir__, "power_264_xyz.csv")
        crossley_path = os.path.join(__dir__, "crossley_638_xyz.csv")
        super().__init__(img_file, bg_file, roi_file, power_path, crossley_path)


if __name__ == "__main__":
    matrix = np.load(
        "/data/yeww0006/FYP-SSL/ref/biomarkers/group_mean_score_matrix.npy",
        allow_pickle=True,
    )
    viz = ADHDBiomarkersVisualizer()
    viz.plot_connectome(matrix, "connectome.png")
    viz.plot_stat_map(matrix, "./")
    viz.plot_roi("roi.png")
    viz.plot_module_importance_boxplot(matrix, "boxplot.png")
    viz.plot_complete_score_matrix(matrix, "conn_mat.png")
    viz.plot_module_sensitivity_map(matrix, "msm.png", vmax=0.0002)
