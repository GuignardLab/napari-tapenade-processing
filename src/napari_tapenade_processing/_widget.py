import glob
import json
import os
import time
import warnings
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import napari.utils
import numpy as np
import tifffile
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    EmptyWidget,
    Label,
    PushButton,
    create_widget,
)
from napari.layers import Image
from natsort import natsorted
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from tapenade.preprocessing import (
    align_array_major_axis,
    align_array_major_axis_from_files,
    change_array_pixelsize,
    compute_mask,
    crop_array_using_mask,
    crop_array_using_mask_from_files,
    global_contrast_enhancement,
    local_contrast_enhancement,
    masked_gaussian_smoothing,
    normalize_intensity,
    reorganize_array_dimensions,
    reorganize_array_dimensions_from_files,
    segment_stardist,
    segment_stardist_from_files,
)
from tapenade.preprocessing.segmentation_postprocessing import (
    remove_labels_outside_of_mask,
)
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from napari_tapenade_processing._custom_widgets import HoverTooltipButton
from napari_tapenade_processing._macro_recorder import MacroRecorder
from napari_tapenade_processing._processing_graph import ProcessingGraph

if TYPE_CHECKING:
    import napari

"""
! TODO:
- Use adjective_dict in functions instead of manually specifying the name of the output layers
- Add manual rotation of principal axis
- Add Napari progress bars https://napari.org/stable/api/napari.utils.progress.html
- Block function calls when a function is already running ?
    -> Instead, at the start of each function, disable the run button
        -> does not work as the viewer state is not updated before all functions are finished
- Add tracks processing functions
- Replace sliders with spinboxes for integer values
- Consider replacing sliders with spinboxes for float values too

- Add dask support for macro running
- Populate labels postprocessing
    - labels smoothing
    - remove labels that touch the border
    - close holes

"""


class TapenadeProcessingWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self._viewer = viewer

        self._array_layer_combo = create_widget(
            widget_type="ComboBox",
            label="Array",
            options={"nullable": False},
        )

        self._image_layer_combo = create_widget(
            widget_type="ComboBox",
            label="Image",
            options={"nullable": True},
        )

        self._ref_image_layer_combo = create_widget(
            widget_type="ComboBox",
            label="Image (ref)",
            options={"nullable": True},
        )

        self._mask_layer_combo = create_widget(
            widget_type="ComboBox",
            label="Mask",
            options={"nullable": True},
        )

        self._mask_for_volume_layer_combo = create_widget(
            widget_type="ComboBox",
            label="Mask (volume)",
            options={"nullable": True},
        )

        self._labels_layer_combo = create_widget(
            widget_type="ComboBox",
            label="Labels",
            options={"nullable": True},
        )

        layer_combos_container = Container(
            widgets=[
                self._array_layer_combo,
                self._image_layer_combo,
                self._ref_image_layer_combo,
                self._mask_layer_combo,
                self._mask_for_volume_layer_combo,
                self._labels_layer_combo,
            ],
            layout="vertical",
            labels=True,
        )

        self._update_layer_combos()

        # self._image_layer_combo.native.currentIndexChanged.connect(self._populate_layer_combos_values)
        self._array_layer_combo.bind(self._bind_layer_combo)
        self._image_layer_combo.bind(self._bind_layer_combo)
        self._ref_image_layer_combo.bind(self._bind_layer_combo)
        self._mask_layer_combo.bind(self._bind_layer_combo)
        self._mask_for_volume_layer_combo.bind(self._bind_layer_combo)
        self._labels_layer_combo.bind(self._bind_layer_combo)

        if True:
            # Making array isotropic
            if True:
                self._rescale_interp_order_combo = create_widget(
                    label="Interp order",
                    options={
                        "choices": ["Nearest", "Linear", "Cubic"],
                        "value": "Linear",
                    },
                )
                self._rescale_interp_order_combo.bind(
                    self._bind_combo_interpolation_order
                )
                tooltip_rescale = (
                    "Interpolation order.\n"
                    "0: Nearest, 1: Linear, 3: Cubic\n"
                    "Bigger means slower but smoother."
                )

                rescale_interp_order_label = Label(value="Images interp order")

                rescale_interp_order_container = Container(
                    widgets=[
                        rescale_interp_order_label,
                        self._rescale_interp_order_combo,
                    ],
                    layout="horizontal",
                    labels=False,
                )

                self._add_tooltip_button_to_container(
                    rescale_interp_order_container, tooltip_rescale
                )

                self._rescale_input_pixelsize = create_widget(
                    widget_type="TupleEdit",
                    label="In",
                    options={
                        "value": (1.0, 1.0, 1.0),
                        "layout": "vertical",
                        "options": {"min": 0},
                    },
                )

                self._rescale_output_pixelsize = create_widget(
                    widget_type="TupleEdit",
                    label="Out",
                    options={
                        "value": (1.0, 1.0, 1.0),
                        "layout": "vertical",
                        "options": {"min": 0},
                    },
                )

                pixelsizes_container = Container(
                    widgets=[
                        self._rescale_input_pixelsize,
                        self._rescale_output_pixelsize,
                    ],
                    layout="horizontal",
                    labels=True,
                )

                self._rescale_container = Container(
                    widgets=[
                        rescale_interp_order_container,
                        Label(value="Voxelsizes (ZYX, e.g in µm/pix):"),
                        pixelsizes_container,
                    ],
                    labels=False,
                )

            # Reorganize array dimensions
            if True:
                self._refresh_dims_button = create_widget(
                    widget_type="PushButton",
                    label="Refresh displayed array dimensions",
                )

                self._refresh_dims_button.clicked.connect(
                    self._update_array_reorganization_comboboxes
                )

                refresh_dims_tooltip = (
                    "Click refresh if the dimensions in the boxes\n"
                    "below don't match the expected dimensions of\n"
                    "the currently selected array."
                )

                refresh_dims_container = self._add_tooltip_button_to_container(
                    self._refresh_dims_button, refresh_dims_tooltip
                )

                self._reorganize_dims_nb_timepoints_combobox = create_widget(
                    widget_type="ComboBox", label="T:"
                )

                select_nb_timepoints_tooltip = "Number of timepoints."
                select_nb_timepoints_container = (
                    self._add_tooltip_button_to_container(
                        self._reorganize_dims_nb_timepoints_combobox,
                        select_nb_timepoints_tooltip,
                    )
                )

                self._reorganize_dims_nb_channels_combobox = create_widget(
                    widget_type="ComboBox", label="C:"
                )
                select_nb_channels_tooltip = "Number of channels."
                select_nb_channels_container = (
                    self._add_tooltip_button_to_container(
                        self._reorganize_dims_nb_channels_combobox,
                        select_nb_channels_tooltip,
                    )
                )

                self._reorganize_dims_depth_combobox = create_widget(
                    widget_type="ComboBox", label="Z:"
                )
                select_depth_tooltip = "Number of planes in the Z dimension."
                select_depth_container = self._add_tooltip_button_to_container(
                    self._reorganize_dims_depth_combobox,
                    select_depth_tooltip,
                )

                self._reorganize_dims_Y_combobox = create_widget(
                    widget_type="ComboBox", label="Y:"
                )
                select_Y_tooltip = "Number of pixels in the Y dimension."
                select_Y_container = self._add_tooltip_button_to_container(
                    self._reorganize_dims_Y_combobox,
                    select_Y_tooltip,
                )

                self._reorganize_dims_X_combobox = create_widget(
                    widget_type="ComboBox", label="X:"
                )
                select_X_tooltip = "Number of pixels in the X dimension."
                select_X_container = self._add_tooltip_button_to_container(
                    self._reorganize_dims_X_combobox,
                    select_X_tooltip,
                )
                self._reorganize_dims_separate_channels_checkbox = (
                    create_widget(
                        widget_type="CheckBox",
                        options={"value": False},
                        label="Separate channels",
                    )
                )
                separate_channels_checkbox_tooltip = "Channels will be split so that each one appear in a different channel."

                separate_channels_container = (
                    self._add_tooltip_button_to_container(
                        self._reorganize_dims_separate_channels_checkbox,
                        separate_channels_checkbox_tooltip,
                    )
                )

                separate_channels_container.margins = (0,)*4

                self._reorganize_dims_keep_original_image_checkbox = (
                    create_widget(
                        widget_type="CheckBox",
                        options={"value": False},
                        label="Keep original image",
                    )
                )
                keep_original_image_checkbox_tooltip = (
                    "If unclicked, the original image will be deleted.\n"
                    "This can be useful to prevent Napari dimension sliders\n"
                    "from becoming confusing."
                )

                # self._array_layer_combo.changed.connect(self._update_layer_combos)

                self._array_layer_combo.native.currentIndexChanged.connect(
                    self._update_array_reorganization_comboboxes
                )
                self._array_layer_combo.native.currentTextChanged.connect(
                    self._update_array_reorganization_comboboxes
                )

                keep_original_image_container = (
                    self._add_tooltip_button_to_container(
                        self._reorganize_dims_keep_original_image_checkbox,
                        keep_original_image_checkbox_tooltip,
                    )
                )

                keep_original_image_container.margins = (0,)*4

                T_container = Container(
                    widgets=[
                        select_nb_timepoints_container,
                        EmptyWidget(),
                    ],
                    layout="horizontal",
                    labels=False,
                )

                T_container.margins = (0,)*4

                CZ_container = Container(
                    widgets=[
                        select_nb_channels_container,
                        select_depth_container,
                    ],
                    layout="horizontal",
                    labels=False,
                )

                CZ_container.margins = (0,)*4

                YX_container = Container(
                    widgets=[
                        select_Y_container,
                        select_X_container,
                    ],
                    layout="horizontal",
                    labels=False,
                )

                YX_container.margins = (0,)*4

                print(T_container.margins)
                print('---')
                # print(T_container.native.margins)

                self._organize_array_dimensions = Container(
                    widgets=[
                        refresh_dims_container,
                        # TCZ_container,
                        # select_nb_timepoints_container,
                        T_container,
                        # select_nb_channels_container,
                        # select_depth_container,
                        CZ_container,
                        # select_Y_container,
                        # select_X_container,
                        YX_container,
                        separate_channels_container,
                        keep_original_image_container,
                    ],
                    labels=False,
                )

            # Spectral filtering
            if True:
                self._spectral_filtering_container = Container(
                    widgets=[
                        EmptyWidget(),
                        Label(value="Not implemented yet."),
                        Label(value="Under construction."),
                        EmptyWidget(),
                    ],
                    labels=False,
                )

            # Computing mask
            if True:
                self._compute_mask_method_combo = create_widget(
                    label="Method",
                    options={
                        "choices": ["otsu", "snp otsu"],
                        "value": "snp otsu",
                    },
                )

                compute_mask_method_tooltip = (
                    "otsu: thresholding with Otsu's method on blurred image.\n"
                    "snp otsu: more robust but slower version of thresholding with Otsu's method."
                )
                compute_mask_method_container = (
                    self._add_tooltip_button_to_container(
                        self._compute_mask_method_combo,
                        compute_mask_method_tooltip,
                    )
                )

                self._compute_mask_sigma_blur_slider = create_widget(
                    widget_type="IntSlider",
                    label="Sigma blur",
                    options={"min": 1, "max": 10, "value": 3},
                )

                compute_mask_sigma_blur_tooltip = (
                    "Sigma of the Gaussian blur applied to the image before thresholding\n"
                    "A good default is ~ object radius/3."
                )

                compute_mask_sigma_blur_container = (
                    self._add_tooltip_button_to_container(
                        self._compute_mask_sigma_blur_slider,
                        compute_mask_sigma_blur_tooltip,
                    )
                )

                compute_mask_sigma_blur_container.margins = (0,)*4

                self._compute_mask_threshold_factor_slider = create_widget(
                    widget_type="FloatSlider",
                    label="Threshold factor",
                    options={"min": 0.5, "max": 1.5, "value": 1},
                )
                compute_mask_threshold_factor_tooltip = (
                    "Multiplicative factor applied to the threshold computed by the chosen method\n"
                    "If the mask is too inclusive (fewer pixels should be True) set factor < 1\n"
                    "If the mask is too excusive (fewer pixels should be False) (set factor > 1)."
                )

                compute_mask_threshold_factor_container = (
                    self._add_tooltip_button_to_container(
                        self._compute_mask_threshold_factor_slider,
                        compute_mask_threshold_factor_tooltip,
                    )
                )

                compute_mask_threshold_factor_container.margins = (0,)*4

                # self._convex_hull_checkbox = create_widget(
                #     widget_type="CheckBox",
                #     label="Compute convex hull",
                #     options={"value": False},
                # )

                # convex_hull_checkbox_tooltip = (
                #     "Returns the convex hull of the mask. Really slow."
                # )
                # convex_hull_container = self._add_tooltip_button_to_container(
                #     self._convex_hull_checkbox, convex_hull_checkbox_tooltip
                # )

                self._compute_mask_post_processing_combo = create_widget(
                    label="Post-processing",
                    options={
                        "choices": ["none", "fill_holes", "convex_hull"],
                        "value": "fill_holes",
                    },
                )

                compute_mask_post_processing_tooltip = (
                    "Post-processing applied to the mask after thresholding.\n"
                    "fill_holes: fills holes in the mask.\n"
                    "convex_hull: returns the convex hull of the mask."
                )

                compute_mask_post_processing_container = (
                    self._add_tooltip_button_to_container(
                        self._compute_mask_post_processing_combo,
                        compute_mask_post_processing_tooltip,
                    )
                )

                self._compute_mask_keep_largest_cc_checkbox = create_widget(
                    widget_type="CheckBox",
                    label="Keep largest connected component",
                    options={"value": True},
                )

                keep_largest_cc_tooltip = "If checked, only the largest connected component of the mask will be kept."

                keep_largest_cc_container = (
                    self._add_tooltip_button_to_container(
                        self._compute_mask_keep_largest_cc_checkbox,
                        keep_largest_cc_tooltip,
                    )
                )

                keep_largest_cc_container.margins = (0,)*4

                self._registered_image_checkbox = create_widget(
                    widget_type="CheckBox",
                    label="Registered image",
                    options={"value": False},
                )
                registered_image_tooltip = (
                    "If checked, the image is assumed to have large areas of 0s outside of the tapenade.\n"
                    "These values will be masked"
                )

                registered_image_container = (
                    self._add_tooltip_button_to_container(
                        self._registered_image_checkbox,
                        registered_image_tooltip,
                    )
                )

                registered_image_container.margins = (0,)*4

                self._compute_mask_container = Container(
                    widgets=[
                        compute_mask_method_container,
                        compute_mask_sigma_blur_container,
                        compute_mask_threshold_factor_container,
                        compute_mask_post_processing_container,
                        keep_largest_cc_container,
                        registered_image_container,
                    ],
                    labels=False,
                )

            # Image contrast enhancement
            if True:
                self._local_global_enhancement_checkbox = create_widget(
                    widget_type="CheckBox",
                    label="Perform global contrast enhancement",
                    options={"value": False},
                )

                self._local_global_enhancement_checkbox.clicked.connect(
                    self._update_local_global_enhancement
                )

                local_global_enhancement_tooltip = (
                    "If checked, the contrast enhancementll be performed globally on the whole image.\n"
                    "If unchecked, the contrast enhancementll be performed locally in boxes of length (2*Box size)+1."
                )

                local_global_enhancement_container = (
                    self._add_tooltip_button_to_container(
                        self._local_global_enhancement_checkbox,
                        local_global_enhancement_tooltip,
                    )
                )

                self._local_norm_box_size_slider = create_widget(
                    widget_type="IntSlider",
                    label="Box size",
                    options={"min": 3, "max": 25, "value": 10},
                )
                local_norm_box_size_tooltip = (
                    "Size of the box used for the image contrast enhancement\n"
                    "A good default is ~ 3/2 * object radius."
                )

                self._local_norm_box_size_container = (
                    self._add_tooltip_button_to_container(
                        self._local_norm_box_size_slider,
                        local_norm_box_size_tooltip,
                    )
                )

                self._enhancement_percentiles_slider = create_widget(
                    widget_type="FloatRangeSlider",
                    label="Percentiles",
                    options={"min": 0, "max": 100, "value": [1, 99]},
                )
                enhancement_percentiles_tooltip = (
                    "Percentiles used for the image contrast enhancement."
                )

                enhancement_percentiles_container = (
                    self._add_tooltip_button_to_container(
                        self._enhancement_percentiles_slider,
                        enhancement_percentiles_tooltip,
                    )
                )

                self._contrast_enhancement_container = Container(
                    widgets=[
                        local_global_enhancement_container,
                        self._local_norm_box_size_container,
                        enhancement_percentiles_container,
                    ],
                    labels=False,
                )

            # Intensity normalization
            if True:
                self._int_norm_sigma_slider = create_widget(
                    widget_type="IntSlider",
                    label="Sigma\n(0=automatic)",
                    options={"min": 0, "max": 30, "value": 20},
                )

                int_norm_sigma_tooltip = (
                    "Sigma for the multiscale gaussian smoothing used to normalize the reference signal.\n"
                    "If 0, the sigma is automatically computed."
                )

                int_norm_sigma_container = (
                    self._add_tooltip_button_to_container(
                        self._int_norm_sigma_slider, int_norm_sigma_tooltip
                    )
                )

                self._int_norm_wavelength_combo = create_widget(
                    label="Image wavelength",
                    options={
                        "choices": [
                            "405 nm (default)",
                            "488 nm",
                            "555 nm",
                            "647 nm",
                        ],
                        "value": "405 nm (default)",
                    },
                )

                int_norm_wavelength_tooltip = (
                    "Wavelength of the image. Used to adjust the intensities in the reference layer."
                )

                int_norm_wavelength_container = (
                    self._add_tooltip_button_to_container(
                        self._int_norm_wavelength_combo, int_norm_wavelength_tooltip
                    )
                )

                self._int_norm_width_slider = create_widget(
                    widget_type="IntSlider",
                    label="Width of ref plane",
                    options={"min": 1, "max": 5, "value": 3},
                )
                int_norm_width_tooltip = (
                    "Width of the reference plane used to compute normalization values.\n"
                    "You usually don't need to change this."
                )

                int_norm_width_container = (
                    self._add_tooltip_button_to_container(
                        self._int_norm_width_slider, int_norm_width_tooltip
                    )
                )

                self._int_norm_container = Container(
                    widgets=[
                        int_norm_sigma_container,
                        int_norm_wavelength_container,
                        int_norm_width_container,
                    ],
                    labels=False,
                )

            # Segment with StarDist
            if True:
                self._segment_stardist_model_path = create_widget(
                    widget_type="FileEdit",
                    options={"mode": "d"},
                    label="Model path",
                )

                self._segment_stardist_model_path.native.children()[
                    1
                ].setPlaceholderText("Path to pretrained model folder")

                self._segment_stardist_default_thresholds_checkbox = (
                    create_widget(
                        widget_type="CheckBox",
                        label="Use default thresholds",
                        options={"value": True},
                    )
                )

                default_thresholds_tooltip = (
                    "If checked, the probability threshold and NMS threshold\n"
                    "will be set to the optimized values from the pretrained model."
                )

                default_thresholds_container = (
                    self._add_tooltip_button_to_container(
                        self._segment_stardist_default_thresholds_checkbox,
                        default_thresholds_tooltip,
                    )
                )

                self._segment_stardist_prob_threshold_slider = create_widget(
                    widget_type="FloatSlider",
                    label="Prob threshold",
                    options={"min": 0, "max": 1, "value": 0.5},
                )

                prob_threshold_tooltip = (
                    "Threshold above which a pixel from the probability map is\n"
                    "considered as being a center candidate.\n"
                    "Lower values will result in more objects."
                )

                self._prob_threshold_container = (
                    self._add_tooltip_button_to_container(
                        self._segment_stardist_prob_threshold_slider,
                        prob_threshold_tooltip,
                    )
                )
                self._prob_threshold_container.enabled = False

                self._segment_stardist_nms_threshold_slider = create_widget(
                    widget_type="FloatSlider",
                    label="NMS threshold",
                    options={"min": 0, "max": 1, "value": 0.4},
                )

                nms_threshold_tooltip = (
                    "IoU threshold for non-maximum suppression.\n"
                    "Higher values will discard redundant candidates better\n"
                    "but could also remove valid objects that are very close to each other."
                )

                self._nms_threshold_container = (
                    self._add_tooltip_button_to_container(
                        self._segment_stardist_nms_threshold_slider,
                        nms_threshold_tooltip,
                    )
                )
                self._nms_threshold_container.enabled = False

                self._segment_stardist_default_thresholds_checkbox.changed.connect(
                    self._update_segment_stardist_thresholds
                )

                self._segment_stardist_container = Container(
                    widgets=[
                        self._segment_stardist_model_path,
                        default_thresholds_container,
                        self._prob_threshold_container,
                        self._nms_threshold_container,
                    ],
                    labels=False,
                )

            # Aligning major axis
            if True:
                self._align_major_axis_interp_order_combo = create_widget(
                    label="Interp order",
                    options={
                        "choices": ["Nearest", "Linear", "Cubic"],
                        "value": "Linear",
                    },
                )

                self._align_major_axis_interp_order_combo.bind(
                    self._bind_combo_interpolation_order
                )

                align_major_axis_order_tooltip = "Interpolation order.\n0: Nearest, 1: Linear, 3: Cubic\nBigger means slower"

                align_major_axis_order_container = (
                    self._add_tooltip_button_to_container(
                        self._align_major_axis_interp_order_combo,
                        align_major_axis_order_tooltip,
                    )
                )

                self._align_major_axis_rotation_plane_combo = create_widget(
                    label="Rotation plane",
                    options={"choices": ["XY", "XZ", "YZ"], "value": "XY"},
                )

                self._align_major_axis_rotation_plane_combo.changed.connect(
                    self._update_target_axis_choices
                )

                align_major_axis_rotation_plane_tooltip = (
                    "2D plane in which the major axis of the mask will be computed,\n"
                    "and the rotation will be applied."
                )

                align_major_axis_rotation_plane_container = (
                    self._add_tooltip_button_to_container(
                        self._align_major_axis_rotation_plane_combo,
                        align_major_axis_rotation_plane_tooltip,
                    )
                )

                self._align_major_axis_target_axis_combo = create_widget(
                    label="Target axis",
                    options={"choices": ["Y", "X"]},
                )

                align_major_axis_target_axis_tooltip = (
                    "Axis to align the major axis of the mask with."
                )

                align_major_axis_target_axis_container = (
                    self._add_tooltip_button_to_container(
                        self._align_major_axis_target_axis_combo,
                        align_major_axis_target_axis_tooltip,
                    )
                )

                self._align_major_axis_container = Container(
                    widgets=[
                        align_major_axis_order_container,
                        align_major_axis_rotation_plane_container,
                        align_major_axis_target_axis_container,
                    ],
                    labels=False,
                )

            # Removing labels outside of mask
            if True:
                self._remove_labels_outside_of_mask_container = Container(
                    widgets=[
                        EmptyWidget(),
                    ],
                )

            # Cropping array using mask
            if True:
                self._crop_array_using_mask_margin_checkbox = create_widget(
                    widget_type="CheckBox",
                    options={"value": False},
                    label="Add 1 pixel margin",
                )

                self._crop_array_using_mask_container = Container(
                    widgets=[
                        self._crop_array_using_mask_margin_checkbox,
                        EmptyWidget(),
                    ],
                )

            # Masked gaussian smoothing
            if True:
                self._masked_smoothing_sigma_slider = create_widget(
                    widget_type="FloatSlider",
                    label="Sigma",
                    options={"min": 1, "max": 50, "value": 1},
                )

                masked_smoothing_sigma_tooltip = (
                    "Standard deviation of the Gaussian kernel used for smoothing.\n"
                    "Defines the spatial scale of the result."
                )

                masked_smoothing_sigma_container = (
                    self._add_tooltip_button_to_container(
                        self._masked_smoothing_sigma_slider,
                        masked_smoothing_sigma_tooltip,
                    )
                )

                self._masked_gaussian_smoothing_container = Container(
                    widgets=[
                        Label(
                            value="Currently, the function is only implemented"
                        ),
                        Label(value="for dense smoothing of images."),
                        EmptyWidget(),
                        masked_smoothing_sigma_container,
                    ],
                    labels=False,
                )

            self._func_name_to_func = {
                "reorganize_array_dimensions": reorganize_array_dimensions,
                "change_array_pixelsize": change_array_pixelsize,
                "compute_mask": compute_mask,
                "global_contrast_enhancement": global_contrast_enhancement,
                "local_contrast_enhancement": local_contrast_enhancement,
                "align_array_major_axis": align_array_major_axis,
                "remove_labels_outside_of_mask": remove_labels_outside_of_mask,
                "crop_array_using_mask": crop_array_using_mask,
                "normalize_intensity": normalize_intensity,
                "segment_stardist": segment_stardist,
                "masked_gaussian_smoothing": masked_gaussian_smoothing,
            }

            self._funcs_combobox_text_to_containers = OrderedDict(
                [
                    (
                        "Re-organize array dimensions",
                        self._organize_array_dimensions,
                    ),
                    ("Change layer voxelsize", self._rescale_container),
                    ("Spectral filtering", self._spectral_filtering_container),
                    ("Compute mask from image", self._compute_mask_container),
                    (
                        "Image contrast enhancement",
                        self._contrast_enhancement_container,
                    ),
                    ("Intensity normalization", self._int_norm_container),
                    (
                        "Align layer from mask major axis",
                        self._align_major_axis_container,
                    ),
                    (
                        "Segment with StarDist",
                        self._segment_stardist_container,
                    ),
                    (
                        "Remove labels outside of mask",
                        self._remove_labels_outside_of_mask_container,
                    ),
                    (
                        "Crop layers using mask",
                        self._crop_array_using_mask_container,
                    ),
                    (
                        "Masked gaussian smoothing",
                        self._masked_gaussian_smoothing_container,
                    ),
                ]
            )

            self._funcs_combobox_text_to_func = OrderedDict(
                [
                    (
                        "Re-organize array dimensions",
                        self._run_organize_array_dimensions,
                    ),
                    ("Change layer voxelsize", self._run_rescale),
                    ("Spectral filtering", None),
                    ("Compute mask from image", self._run_compute_mask),
                    (
                        "Image contrast enhancement",
                        self._run_contrast_enhancement,
                    ),
                    ("Intensity normalization", self._run_normalize_intensity),
                    (
                        "Align layer from mask major axis",
                        self._run_align_major_axis,
                    ),
                    (
                        "Segment with StarDist",
                        self._run_segment_stardist,
                    ),
                    (
                        "Remove labels outside of mask",
                        self._run_remove_labels_outside_of_mask,
                    ),
                    (
                        "Crop layers using mask",
                        self._run_crop_array_using_mask,
                    ),
                    (
                        "Masked gaussian smoothing",
                        # None
                        self._run_masked_gaussian_smoothing,
                    ),
                ]
            )

            self._funcs_combobox_text_to_visible_layers = {
                "Re-organize array dimensions": ["array"],
                "Change layer voxelsize": ["array"],
                "Spectral filtering": [],
                "Compute mask from image": ["image"],
                "Image contrast enhancement": ["image", "mask"],
                "Intensity normalization": [
                    "image",
                    "ref_image",
                    "mask",
                    "labels",
                ],
                "Segment with StarDist": ["image"],
                "Align layer from mask major axis": [
                    "array",
                    "mask",
                ],
                "Remove labels outside of mask": ["mask", "labels"],
                "Crop layers using mask": ["array", "mask"],
                "Masked gaussian smoothing": [
                    "image",
                    "mask",
                    "mask_for_volume",
                ],
            }

            self._adjective_dict = {
                "reorganize_array_dimensions": "reorganized",
                "change_array_pixelsize": "rescaled",
                "compute_mask": "mask",
                "global_contrast_enhancement": "enhanced",
                "local_contrast_enhancement": "enhanced_locally",
                "align_array_major_axis": "aligned",
                "remove_labels_outside_of_mask": "curated",
                "crop_array_using_mask": "cropped",
                "normalize_intensity": "normalized",
                "masked_gaussian_smoothing": "smoothed",
                "spectral_filtering": "filtered",
                "segment_stardist": "segmented",
            }

        self._run_button = create_widget(
            widget_type="PushButton", label="Run function"
        )

        self._run_button.clicked.connect(self._run_current_function)

        self._main_combobox = QComboBox()
        self._main_combobox._explicitly_hidden = False
        self._main_combobox.native = self._main_combobox
        self._main_combobox.name = ""

        main_stack = QStackedWidget()
        main_stack.native = main_stack
        main_stack.name = ""

        for name, w in self._funcs_combobox_text_to_containers.items():
            # manage layout stretch and add to main combobox
            if hasattr(w, "native"):
                w.native.layout().addStretch()
                main_stack.addWidget(w.native)
            else:
                w.layout().addStretch()
                main_stack.addWidget(w)

            self._main_combobox.addItem(name)

        self._main_combobox.currentIndexChanged.connect(
            main_stack.setCurrentIndex
        )
        self._main_combobox.currentIndexChanged.connect(
            self._disable_irrelevant_layers
        )

        main_control = Container(
            widgets=[
                self._main_combobox,
                main_stack,
            ],
            labels=False,
        )

        update_layers_combos_button = create_widget(
            widget_type="PushButton", label="Refresh"
        )

        update_layers_combos_button.clicked.connect(self._update_layer_combos)
        viewer.layers.events.changed.connect(self._update_layer_combos)
        viewer.layers.events.reordered.connect(self._update_layer_combos)
        viewer.layers.events.moved.connect(self._update_layer_combos)
        viewer.layers.events.removed.connect(self._update_layer_combos)
        viewer.layers.events.removing.connect(self._update_layer_combos)
        viewer.layers.events.inserted.connect(self._update_layer_combos)
        viewer.layers.events.inserting.connect(self._update_layer_combos)

        label_and_update_container = Container(
            widgets=[
                Label(value="<u>Layers to process:</u>"),
                EmptyWidget(),
                update_layers_combos_button,
            ],
            layout="horizontal",
            labels=False,
        )

        self._add_tooltip_button_to_container(
            label_and_update_container,
            "Click refresh if a layer does not appear in the list or has a wrong name.",
        )

        self._n_jobs_slider = create_widget(
            widget_type="IntSlider",
            options={"min": 1, "max": os.cpu_count(), "value": 1},
        )

        self._n_jobs_container = Container(
            widgets=[
                Label(value="Num. parallel jobs"),
                self._n_jobs_slider,
            ],
            labels=False,
            layout="horizontal",
        )

        n_jobs_tooltip = (
            "Number of parallel jobs to run.\n"
            "Increasing this number will speed up the computation,\n"
            "but can dramatically increase the amount of memory used.\n"
            'When running functions in the "Functions" tab, parallel computation\n'
            "is triggered if the input arrays are detected as being temporal.\n"
            'When running a macro in the "Macro recording" tab, each frame is\n'
            "processed in parallel."
        )

        self._add_tooltip_button_to_container(
            self._n_jobs_container, n_jobs_tooltip
        )

        self._function_tab_container = Container(
            widgets=[
                # self._n_jobs_container,
                label_and_update_container,
                layer_combos_container,
                # update_layers_combos_button,
                Label(value="<u>Processing functions:</u>"),
                main_control,
                self._run_button,
            ],
            labels=False,
        )

        ### Recording of parameters
        self._record_parameters_path = create_widget(
            widget_type="FileEdit",
            options={"mode": "d"},
            label="Macro save path",
        )
        self._record_parameters_path.native.children()[1].setPlaceholderText(
            "Path to save the macro"
        )

        self._record_parameters_button = create_widget(
            widget_type="PushButton", label="Start recording macro"
        )

        self._run_macro_parameters_path = create_widget(
            widget_type="FileEdit",
            options={"mode": "r", "filter": "*.json"},
            label="Macro",
        )
        self._run_macro_parameters_path.native.children()[
            1
        ].setPlaceholderText("Path to the macro json")

        self._run_macro_save_path = create_widget(
            widget_type="FileEdit", options={"mode": "d"}, label="Save"
        )
        self._run_macro_save_path.native.children()[1].setPlaceholderText(
            "Path to save the results"
        )

        self._macro_widgets = {}

        self._run_macro_save_all_checkbox = create_widget(
            widget_type="CheckBox",
            label="Save all intermediate results",
            options={"value": False},
        )
        self._run_macro_save_all_checkbox.native.setEnabled(False)

        self._run_macro_compress_checkbox = create_widget(
            widget_type="CheckBox",
            label="Compress when saving",
            options={"value": False},
        )

        self._run_macro_button = create_widget(
            widget_type="PushButton", label="Run macro"
        )
        self._run_macro_button.clicked.connect(self._run_macro)

        self._test_button = create_widget(
            widget_type="PushButton", label="Display macro graph"
        )

        self._macro_tab_container = Container(
            widgets=[
                Label(value="<u>Recording macro</u>"),
                Label(value="Path to save the macro json file:"),
                self._record_parameters_path,
                self._record_parameters_button,
                EmptyWidget(),
                Label(value="<u>Running macro</u>"),
                Label(value="Paths to macro parameters:"),
                self._run_macro_parameters_path,
            ],
            layout="vertical",
            labels=False,
        )

        self._run_macro_parameters_path.changed.connect(
            self._update_macro_widgets
        )

        self._recorder = MacroRecorder()
        self._is_recording_parameters = False

        self._record_parameters_button.clicked.connect(
            self._manage_macro_widget
        )

        self._processing_graph = None
        ###

        logo_path = str(Path(__file__).parent / "logo" / "tapenade3.png")

        # label = create_widget(
        #     widget_type="Label", label=f'<img src="{logo_path}"></img>'
        # )
        pixmap = QPixmap(logo_path)
        pixmap = pixmap.scaled(80, 60, transformMode=Qt.SmoothTransformation)
        label = QLabel()
        label.setPixmap(pixmap)

        label._explicitly_hidden = False
        label.native = label
        label.name = ""

        link_website = "morphotiss.org/"
        link_DOI = "https://doi.org/10.1101/2024.08.13.607832"

        texts_container = Container(
            widgets=[
                Label(
                    value="<small>This plugin is part of TAPENADE.<br>"
                    f"Using it in your research ?<br>"
                    f'Please <a href="{link_DOI}" style="color:gray;">cite us</a>.'
                    # f'</small><br><br><tt><a href="https://www.{link_website}" style="color:gray;">{link_website}</a></tt>'
                ),
            ],
            layout="vertical",
            labels=False,
        )

        self._header_container = Container(
            widgets=[label, texts_container],
            layout="horizontal",
            labels=False,
        )

        tabs = QTabWidget()
        # tabs = MultiLineTabWidget()

        ### Advanced parameters
        self._overwrite_checkbox = create_widget(
            widget_type="CheckBox",
            label="New layers overwrite previous ones",
            options={"value": False},
        )

        self._overwrite_tooltip = (
            "If checked, a newly computed layer will overwrite the one\n"
            "of the same type that was used as input.\n"
            "This can be useful to save memory."
        )

        overwrite_container = self._add_tooltip_button_to_container(
            self._overwrite_checkbox, self._overwrite_tooltip
        )

        # self._systematic_crop_checkbox = create_widget(
        #     widget_type="CheckBox",
        #     label="Results are cropped using mask",
        #     options={"value": False},
        # )
        # self._systematic_crop_checkbox.native.setEnabled(False)

        # self._systematic_crop_tooltip = (
        #     "If checked, the results of the functions will be systematically cropped using the mask.\n"
        #     "This can be useful to save memory.\n"
        #     "If not, the results will have the same shape as the input layers."
        # )

        self._general_parameters_tab_container = Container(
            widgets=[
                overwrite_container,
                # self._systematic_crop_checkbox,
            ],
            layout="vertical",
            labels=False,
        )
        ###

        self._general_parameters_tab_container.native.layout().addStretch(1)
        self._macro_tab_container.native.layout().addStretch(1)
        self._function_tab_container.native.layout().addStretch(1)

        tabs.addTab(self._function_tab_container.native, "Functions")
        tabs.addTab(self._macro_tab_container.native, "Macro recording")
        tabs.addTab(
            self._general_parameters_tab_container.native, "General params"
        )

        self.setLayout(QVBoxLayout())

        self.layout().addWidget(self._header_container.native)
        self.layout().addWidget(self._n_jobs_container.native)
        self.layout().addWidget(tabs)
        self.layout().addStretch(1)

        self._disable_irrelevant_layers(0)
        self._update_layer_combos()

    def _update_local_global_enhancement(self, event):
        self._local_norm_box_size_container.enabled = not event

    def _update_segment_stardist_thresholds(self, event):
        self._prob_threshold_container.enabled = not event
        self._nms_threshold_container.enabled = not event

    def _add_tooltip_button_to_container(self, container, tooltip_text):
        button = HoverTooltipButton(tooltip_text)
        button.native = button
        button._explicitly_hidden = False
        button.name = ""

        if isinstance(container, Container):
            container.append(button)
        else:
            if isinstance(container, CheckBox | PushButton):
                container = Container(
                    widgets=[container, button],
                    labels=False,
                    layout="horizontal",
                )
            else:
                container_label = container.label
                container.label = ""
                container = Container(
                    widgets=[Label(value=container_label), container, button],
                    labels=False,
                    layout="horizontal",
                )

            container.margins = (8,)*4
            return container
        return None

    def _bind_combo_interpolation_order(self, obj):
        if obj.native.currentText() == "Nearest":
            return 0
        elif obj.native.currentText() == "Linear":
            return 1
        elif obj.native.currentText() == "Cubic":
            return 3

    def _bind_layer_combo(self, obj):
        """
        This used so that when calling layer_combo.value, we get the layer object,
        not the name of the layer
        """
        name = obj.native.currentText()
        if name not in ("", "-----"):
            return self._viewer.layers[name]
        else:
            return None

    def _update_layer_combos(self):

        # self._array_layer_combo.native.blockSignals(True)

        ### 1. Clear all combos but keep the previous choice if possible
        previous_texts = []

        # clear all combos and add None
        for c in (
            self._array_layer_combo,
            self._image_layer_combo,
            self._ref_image_layer_combo,
            self._mask_layer_combo,
            self._mask_for_volume_layer_combo,
            self._labels_layer_combo,
            # self._tracks_layer_combo
        ):
            previous_texts.append(c.native.currentText())
            c.native.clear()
            # check if combo is nullable
            if c._nullable:
                c.native.addItem(None)

        ### 2. Add layers to combos
        # add layers to compatible combos
        for layer in self._viewer.layers:
            if (
                isinstance(layer, napari.layers.Image | napari.layers.Labels)
                and self._array_layer_combo.enabled
            ):
                self._array_layer_combo.native.addItem(layer.name)
            if isinstance(layer, napari.layers.Image):
                if layer.data.dtype == bool:
                    if self._mask_layer_combo.enabled:
                        self._mask_layer_combo.native.addItem(layer.name)
                    if self._mask_for_volume_layer_combo.enabled:
                        self._mask_for_volume_layer_combo.native.addItem(
                            layer.name
                        )
                else:
                    if self._image_layer_combo.enabled:
                        self._image_layer_combo.native.addItem(layer.name)
                    if self._ref_image_layer_combo.enabled:
                        self._ref_image_layer_combo.native.addItem(layer.name)
            elif (
                isinstance(layer, napari.layers.Labels)
                and self._labels_layer_combo.enabled
            ):
                self._labels_layer_combo.native.addItem(layer.name)
            # elif isinstance(layer, Tracks):
            #     self._tracks_layer_combo.addItem(layer.name)

        ### 3. Reset combo current choice to previous text if possible
        for index_c, c in enumerate(
            [
                self._array_layer_combo,
                self._image_layer_combo,
                self._ref_image_layer_combo,
                self._mask_layer_combo,
                self._mask_for_volume_layer_combo,
                self._labels_layer_combo,
                # self._tracks_layer_combo
            ]
        ):
            all_choices = [
                c.native.itemText(i) for i in range(c.native.count())
            ]
            if previous_texts[index_c] in all_choices:

                # if the previous layer is None, set it to the newest layer
                if previous_texts[index_c] == c.native.itemText(0):
                    c.native.setCurrentIndex(c.native.count() - 1)
                else:
                    c.native.setCurrentText(previous_texts[index_c])
            else:
                c.native.setCurrentIndex(0)

        # self._array_layer_combo.native.blockSignals(False)

    def _manage_macro_widget(self):
        path = str(self._record_parameters_path.value)

        if path == "." or not os.path.exists(path):
            napari.utils.notifications.show_warning(
                "Please enter a path to record the macro"
            )
        else:
            if not self._is_recording_parameters:
                self._is_recording_parameters = True

                self._record_parameters_button.native.setText(
                    "Stop recording and save macro"
                )
                self._record_parameters_path.enabled = False
                self._test_button.enabled = True
            else:  # if was recording
                self._is_recording_parameters = False

                self._recorder.dump_recorded_parameters(path)

                self._record_parameters_button.native.setText(
                    "Start tecording macro"
                )
                self._record_parameters_path.enabled = True
                self._test_button.enabled = False

    def _bool_layers_filter(self, wdg: ComboBox):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, Image) and layer.data.dtype == bool
        ]

    def _not_bool_layers_filter(self, wdg: ComboBox):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, Image) and layer.data.dtype != bool
        ]

    def _run_current_function(self):
        function_text = self._main_combobox.currentText()
        function = self._funcs_combobox_text_to_func[function_text]
        # run function
        function()

    def _disable_irrelevant_layers(self, event):

        name, _ = list(self._funcs_combobox_text_to_containers.items())[event]

        list_layers_enabled = self._funcs_combobox_text_to_visible_layers[name]

        for layer_type in [
            "array",
            "image",
            "ref_image",
            "mask",
            "mask_for_volume",
            "labels",
        ]:
            combo = getattr(self, f"_{layer_type}_layer_combo")
            # combo.enabled = layer_type in list_layers_enabled
            combo.visible = layer_type in list_layers_enabled

    def _identify_layer_type(self, layer: "napari.layers.Layer"):
        layer_type = layer.__class__.__name__
        if layer_type in ("Image", "Labels"):
            return layer_type
        else:
            return "Other"

    def _assert_basic_layer_properties(
        self, layer: "napari.layers.Layer", allowed_types: list
    ):

        if layer is None:
            msg = "Please select a layer"
            napari.utils.notifications.show_warning(msg)
            # raise ValueError(msg)
            return None, None

        if layer.data.ndim not in (3, 4, 5):
            msg = "The layer must be 3D (ZYX) or 3D+time (TZYX) or 3D+channels+time (CTZYX)"
            napari.utils.notifications.show_warning(msg)
            # raise ValueError(msg)
            return None, None

        layer_type = self._identify_layer_type(layer)
        if layer_type not in allowed_types:
            msg = f"The layer must be part of {allowed_types}"
            napari.utils.notifications.show_warning(msg)
            # raise ValueError(msg)
            return None, None

        return layer, layer_type

    def _transmissible_image_layer_properties(
        self, layer: "napari.layers.Image", exclude: list = []
    ):
        properties = {
            "contrast_limits": layer.contrast_limits,
            "gamma": layer.gamma,
            "colormap": layer.colormap,
            "blending": layer.blending,
            "opacity": layer.opacity,
        }

        return {k: v for k, v in properties.items() if k not in exclude}

    def _transmissible_labels_layer_properties(
        self, layer: "napari.layers.Labels", exclude: list = []
    ):
        return {
            "colormap": layer.colormap,
            "blending": layer.blending,
            "opacity": layer.opacity,
        }

    def _run_rescale(self):

        layer, layer_type = self._assert_basic_layer_properties(
            self._array_layer_combo.value, ["Image", "Labels"]
        )
        if layer is None:
            return

        if layer.data.dtype == bool:
            layer_type = "Mask"

        input_pixelsize = self._rescale_input_pixelsize.value
        output_pixelsize = self._rescale_output_pixelsize.value

        assert not (
            any(factor <= 0 for factor in input_pixelsize)
        ), "Input voxel size must have non-zero elements"
        assert not (
            any(factor <= 0 for factor in output_pixelsize)
        ), "Output voxel size must have non-zero elements"

        func_params = {
            "order": self._rescale_interp_order_combo.value,
            "input_pixelsize": input_pixelsize,
            "output_pixelsize": output_pixelsize,
            "n_jobs": self._n_jobs_slider.value,
        }

        if layer_type in ("Labels", "Mask"):
            func_params["order"] = 0

        start_time = time.time()
        result_array = change_array_pixelsize(layer.data, **func_params)
        napari.utils.notifications.show_info(
            f"Array rescaling took {time.time() - start_time} seconds"
        )

        old_name = layer.name
        name = f"{old_name}_{self._adjective_dict['change_array_pixelsize']}"

        if self._overwrite_checkbox.value:
            layer.data = result_array
            layer.name = name
        else:
            if layer_type in ("Image", "Mask"):
                self._viewer.add_image(
                    result_array,
                    name=name,
                    **self._transmissible_image_layer_properties(layer),
                )
            else:
                self._viewer.add_labels(
                    result_array,
                    name=name,
                    **self._transmissible_labels_layer_properties(layer),
                )
            self._array_layer_combo.native.setCurrentIndex(
                self._array_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:

            input_params_to_layer_names_and_types_dict = {
                "array": (old_name, layer_type),
            }
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("rescaled_array", (name, layer_type))]
            )
            self._recorder.record(
                function_name="change_array_pixelsize",
                func_params=func_params,
                main_input_param_name="array",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _dim_from_duplicate_suffix(self, name):
        return int(name.split("dim ")[-1].split(")")[0])

    def _remove_duplicate_suffix(self, name):
        return name.split(" (")[0]

    def _dims_str_from_shape(
        self, array_shape, nb_channels, nb_timepoints, nb_depth, nb_Y, nb_X
    ):
        dimensions_as_string = "CTZYX"
        ordered_shape = [nb_channels, nb_timepoints, nb_depth, nb_Y, nb_X]
        is_duplicate_list = [" " in elem for elem in ordered_shape]

        if nb_channels == "None":
            dimensions_as_string = dimensions_as_string.replace("C", "")
        if nb_timepoints == "None":
            dimensions_as_string = dimensions_as_string.replace("T", "")
        if nb_depth == "None":
            dimensions_as_string = dimensions_as_string.replace("Z", "")

        inds_transpose = []

        for s, is_duplicate in zip(ordered_shape, is_duplicate_list):
            if s == "None":
                pass
            elif is_duplicate:
                ind_dim = self._dim_from_duplicate_suffix(s)
                inds_transpose.append(ind_dim)
            else:
                inds_transpose.append(array_shape.index(int(s)))

        inds_transpose = np.argsort(inds_transpose)
        dimensions_as_string = "".join(
            [dimensions_as_string[ind] for ind in inds_transpose]
        )

        return dimensions_as_string

    def _run_organize_array_dimensions(self):

        layer, layer_type = self._assert_basic_layer_properties(
            self._array_layer_combo.value, ["Image", "Labels"]
        )
        if layer is None:
            return

        nb_channels = self._reorganize_dims_nb_channels_combobox.value
        nb_timepoints = self._reorganize_dims_nb_timepoints_combobox.value
        nb_depth = self._reorganize_dims_depth_combobox.value
        nb_Y = self._reorganize_dims_Y_combobox.value
        nb_X = self._reorganize_dims_X_combobox.value

        dimensions_as_string = self._dims_str_from_shape(
            layer.data.shape, nb_channels, nb_timepoints, nb_depth, nb_Y, nb_X
        )

        func_params = {
            "bool_seperate_channels": self._reorganize_dims_separate_channels_checkbox.value,
            "dimensions_as_string": dimensions_as_string,
        }

        shape = [str(i) for i in layer.data.shape] + ["None"] * (
            5 - len(layer.data.shape)
        )
        selected_dims = [
            self._remove_duplicate_suffix(i)
            for i in [
                nb_channels,
                nb_timepoints,
                nb_depth,
                nb_Y,
                nb_X,
            ]
        ]
        if sorted(selected_dims) != sorted(
            shape
        ):  # if dimensions do not match, e.g if the same dim is selected 2 times
            msg = (
                "Dimensions selected do not match the shape of the image"
                f"{np.sort(selected_dims)} != {np.sort(shape)}\n"
            )
            napari.utils.notifications.show_warning(msg)
            return

        start_time = time.time()
        reorganized_array = reorganize_array_dimensions(
            layer.data, **func_params
        )
        napari.utils.notifications.show_info(
            f"Reorganization of array dimensions took {time.time() - start_time} seconds"
        )

        old_name = layer.name
        name = (
            f"{old_name}_{self._adjective_dict['reorganize_array_dimensions']}"
        )

        if func_params["bool_seperate_channels"]:
            channel_names = [
                f"{name}_ch{index}" for index in range(len(reorganized_array))
            ]
            for channel_name, channel_array in zip(channel_names, reorganized_array):

                if layer_type == "Image":
                    self._viewer.add_image(
                        channel_array,
                        name=channel_name,
                        **self._transmissible_image_layer_properties(
                            layer, exclude=["contrast_limits"]
                        ),
                    )
                elif layer_type == "Labels":
                    self._viewer.add_labels(
                        channel_array,
                        name=channel_name,
                        **self._transmissible_labels_layer_properties(layer),
                    )
                else:
                    raise ValueError("Layer type not recognized")
        elif self._overwrite_checkbox.value:
            layer.data = reorganized_array
            layer.name = name
        else:
            if layer_type == "Image":
                self._viewer.add_image(
                    reorganized_array,
                    name=name,
                    **self._transmissible_image_layer_properties(layer),
                )
            elif layer_type == "Labels":
                self._viewer.add_labels(
                    reorganized_array,
                    name=name,
                    **self._transmissible_labels_layer_properties(layer),
                )
            else:
                raise ValueError("Layer type not recognized")
        if (
            not self._reorganize_dims_keep_original_image_checkbox.value
            and not self._overwrite_checkbox.value
        ):
            print("removing original image")
            self._viewer.layers.remove(layer)

        if self._is_recording_parameters:

            input_params_to_layer_names_and_types_dict = {
                "array": (old_name, layer_type),
            }
            if func_params["bool_seperate_channels"]:
                output_params_to_layer_names_and_types_dict = OrderedDict(
                    [
                        (
                            f"reorganized_array_ch{index}",
                            (channel_name, layer_type),
                        )
                        for index, channel_name in enumerate(channel_names)
                    ]
                )
            else:
                output_params_to_layer_names_and_types_dict = OrderedDict(
                    [("reorganized_array", (name, layer_type))]
                )
            self._recorder.record(
                function_name="reorganize_array_dimensions",
                func_params=func_params,
                main_input_param_name="array",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _run_compute_mask(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ["Image"]
        )
        if layer is None:
            return

        func_params = {
            "method": self._compute_mask_method_combo.value,
            "sigma_blur": self._compute_mask_sigma_blur_slider.value,
            "threshold_factor": self._compute_mask_threshold_factor_slider.value,
            "post_processing_method": self._compute_mask_post_processing_combo.value,
            "keep_largest_cc": self._compute_mask_keep_largest_cc_checkbox.value,
            "registered_image": self._registered_image_checkbox.value,
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        mask = compute_mask(layer.data, **func_params)
        napari.utils.notifications.show_info(
            f"Mask computation took {time.time() - start_time} seconds"
        )

        name = f"{layer.name}_{self._adjective_dict['compute_mask']}"

        self._viewer.add_image(
            mask,
            name=name,
            blending="additive",
            opacity=0.7,
        )

        self._mask_layer_combo.native.setCurrentIndex(
            self._mask_layer_combo.native.count() - 1
        )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "image": (layer.name, "Image"),
            }
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("mask", (name, "Image"))]
            )

            self._recorder.record(
                function_name="compute_mask",
                func_params=func_params,
                main_input_param_name="image",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _run_contrast_enhancement(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ["Image"]
        )
        if layer is None:
            return

        mask_available = self._mask_layer_combo.value is not None

        if mask_available:
            mask_layer, _ = self._assert_basic_layer_properties(
                self._mask_layer_combo.value, ["Image"]
            )
            if mask_layer is None:
                return
            mask_layer_data = mask_layer.data
            assert (
                mask_layer_data.shape == layer.data.shape
            ), "Mask and data must have the same shape"
        else:
            mask_layer_data = None

        perc_low, perc_high = self._enhancement_percentiles_slider.value

        if self._local_global_enhancement_checkbox.value:

            func_name = "global_contrast_enhancement"

            func_params = {
                "perc_low": perc_low,
                "perc_high": perc_high,
                "n_jobs": self._n_jobs_slider.value,
            }

            start_time = time.time()
            enhanced_array = global_contrast_enhancement(
                layer.data, mask=mask_layer_data, **func_params
            )
            napari.utils.notifications.show_info(
                f"Global contrast enhancement took {time.time() - start_time} seconds"
            )

        else:

            func_name = "local_contrast_enhancement"

            func_params = {
                "perc_low": perc_low,
                "perc_high": perc_high,
                "box_size": self._local_norm_box_size_slider.value,
                "n_jobs": self._n_jobs_slider.value,
            }

            start_time = time.time()
            enhanced_array = local_contrast_enhancement(
                layer.data, mask=mask_layer_data, **func_params
            )
            napari.utils.notifications.show_info(
                f"Local contrast enhancement took {time.time() - start_time} seconds"
            )

        name = f"{layer.name}_{self._adjective_dict[func_name]}"

        if self._overwrite_checkbox.value:
            layer.data = enhanced_array
            layer.contrast_limits = (0, 1)
            layer.name = name
        else:
            image_properties = self._transmissible_image_layer_properties(
                layer
            )
            image_properties["contrast_limits"] = (0, 1)
            self._viewer.add_image(
                enhanced_array,
                name=name,
                **image_properties,
            )

            self._image_layer_combo.native.setCurrentIndex(
                self._image_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "image": (layer.name, "Image"),
            }
            if mask_available:
                input_params_to_layer_names_and_types_dict["mask"] = (
                    mask_layer.name,
                    "Image",
                )
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("enhanced_image", (name, "Image"))]
            )
            self._recorder.record(
                function_name=func_name,
                func_params=func_params,
                main_input_param_name="image",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _run_normalize_intensity(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ["Image"]
        )
        if layer is None:
            return

        ref_layer, _ = self._assert_basic_layer_properties(
            self._ref_image_layer_combo.value, ["Image"]
        )
        if ref_layer is None:
            return

        mask_available = self._mask_layer_combo.value is not None

        if mask_available:
            mask_layer, _ = self._assert_basic_layer_properties(
                self._mask_layer_combo.value, ["Image"]
            )
            if mask_layer is None:
                return
            mask_layer_data = mask_layer.data
            assert (
                mask_layer_data.shape == layer.data.shape
            ), "Mask and data must have the same shape"
        else:
            mask_layer_data = None

        labels_available = self._labels_layer_combo.value is not None

        if labels_available:
            labels_layer, _ = self._assert_basic_layer_properties(
                self._labels_layer_combo.value, ["Labels"]
            )
            if labels_layer is None:
                return
            labels_layer_data = labels_layer.data
            assert (
                labels_layer_data.shape == layer.data.shape
            ), "Labels and data must have the same shape"
        else:
            labels_layer_data = None

        sigma = self._int_norm_sigma_slider.value
        if sigma == 0:
            sigma = None

        image_wavelength = int(self._int_norm_image_wavelength_slider.value.split(" ")[0])

        width = self._int_norm_width_slider.value

        func_params = {
            "sigma": sigma,
            "width": width,
            "image_wavelength": image_wavelength,
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        normalized_image = normalize_intensity(
            image=layer.data,
            ref_image=ref_layer.data,
            mask=mask_layer_data,
            labels=labels_layer_data,
            **func_params,
        )
        napari.utils.notifications.show_info(
            f"intensity normalization took {time.time() - start_time} seconds"
        )

        if mask_layer_data is not None:
            normalized_image = np.where(mask_layer_data, normalized_image, 0.0)

        old_name = layer.name
        name = f"{old_name}_{self._adjective_dict['normalize_intensity']}"

        if self._overwrite_checkbox.value:
            layer.data = normalized_image
            layer.name = name
        else:
            self._viewer.add_image(
                normalized_image,
                name=name,
                **self._transmissible_image_layer_properties(layer),
            )

            self._image_layer_combo.native.setCurrentIndex(
                self._image_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "image": (old_name, "Image"),
                "ref_image": (ref_layer.name, "Image"),
            }
            if mask_available:
                input_params_to_layer_names_and_types_dict["mask"] = (
                    mask_layer.name,
                    "Image",
                )
            if labels_available:
                input_params_to_layer_names_and_types_dict["labels"] = (
                    labels_layer.name,
                    "Labels",
                )
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("normalized_image", (name, "Image"))]
            )
            self._recorder.record(
                function_name="normalize_intensity",
                func_params=func_params,
                main_input_param_name="image",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _run_segment_stardist(self):

        image_layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ["Image"]
        )
        if image_layer is None:
            return

        model_path = str(self._segment_stardist_model_path.value)

        if model_path == "." or not os.path.exists(model_path):
            warnings.warn("Please enter a path to the StarDist model")
            return

        if self._segment_stardist_default_thresholds_checkbox.value:
            thresholds_dict = None
        else:
            thresholds_dict = {
                "prob": self._segment_stardist_prob_threshold_slider.value,
                "nms": self._segment_stardist_nms_threshold_slider.value,
            }

        func_params = {
            "model_path": model_path,
            "thresholds_dict": thresholds_dict,
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        labels = segment_stardist(image_layer.data, **func_params)
        napari.utils.notifications.show_info(
            f"StarDist segmentation took {time.time() - start_time} seconds"
        )

        old_name = image_layer.name
        name = f"{old_name}_{self._adjective_dict['segment_stardist']}"
        self._viewer.add_labels(labels, name=name)

        self._labels_layer_combo.native.setCurrentIndex(
            self._labels_layer_combo.native.count() - 1
        )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "image": (old_name, "Image"),
            }
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("segmented_labels", (name, "Labels"))]
            )
            self._recorder.record(
                function_name="segment_stardist",
                func_params=func_params,
                main_input_param_name="image",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _run_align_major_axis(self):

        mask_layer, _ = self._assert_basic_layer_properties(
            self._mask_layer_combo.value, ["Image"]
        )
        if mask_layer is None:
            return

        array_layer, layer_type = self._assert_basic_layer_properties(
            self._array_layer_combo.value, ["Image", "Labels"]
        )
        if array_layer is None:
            return

        if array_layer.data.dtype == bool:
            layer_type = "Mask"

        func_params = {
            "target_axis": self._align_major_axis_target_axis_combo.value,
            "rotation_plane": self._align_major_axis_rotation_plane_combo.value,
            "order": self._align_major_axis_interp_order_combo.value,
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        array = align_array_major_axis(
            mask=mask_layer.data, array=array_layer.data, **func_params
        )
        if layer_type == "Mask":
            array = array.astype(bool)
        napari.utils.notifications.show_info(
            f"Alignment took {time.time() - start_time} seconds"
        )

        old_name = array_layer.name
        name = f"{old_name}_{self._adjective_dict['align_array_major_axis']}"

        if self._overwrite_checkbox.value:
            array_layer.data = array
            array_layer.name = name
        else:
            if layer_type in ("Image", "Mask"):
                self._viewer.add_image(
                    array,
                    name=name,
                    **self._transmissible_image_layer_properties(array_layer),
                )
            else:
                self._viewer.add_labels(
                    array,
                    name=name,
                    **self._transmissible_labels_layer_properties(array_layer),
                )
            self._array_layer_combo.native.setCurrentIndex(
                self._array_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "mask": (mask_layer.name, "Image"),
                "array": (old_name, layer_type),
            }
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("aligned_array", (name, layer_type))]
            )

            self._recorder.record(
                function_name="align_array_major_axis",
                func_params=func_params,
                main_input_param_name="array",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _add_suffix_to_duplicates(self, dimensions_str: list):
        """
        Add a suffix to dimensions that are the same
        """
        dimensions_str = [
            (
                f"{dim} (dim {i})"
                if (dim != "None" and dimensions_str.count(dim) > 1)
                else dim
            )
            for i, dim in enumerate(dimensions_str)
        ]
        return dimensions_str

    def _update_array_reorganization_comboboxes(self, event):

        if self._array_layer_combo.value is not None:
            layer, _ = self._assert_basic_layer_properties(
                self._array_layer_combo.value, ["Image", "Labels"]
            )
            if layer is None:
                return

            dimensions_str = [str(i) for i in layer.data.shape] + [
                "None"
            ]  # choices in the comboboxes
            dimensions_str = self._add_suffix_to_duplicates(dimensions_str)
            if layer.data.ndim == 2:  # YX
                default_dimensions = [
                    "None",
                    "None",
                    "None",
                    dimensions_str[0],
                    dimensions_str[1],
                ]
            elif layer.data.ndim == 3:  # ZYX
                default_dimensions = [
                    "None",
                    "None",
                    dimensions_str[0],
                    dimensions_str[1],
                    dimensions_str[2],
                ]
            elif layer.data.ndim == 4:  # CZYX #
                default_dimensions = [
                    "None",
                    dimensions_str[1],
                    dimensions_str[0],
                    dimensions_str[2],
                    dimensions_str[3],
                ]
                # default_dimensions = [dimensions_str[0], 'None', dimensions_str[1], dimensions_str[2], dimensions_str[3]] # if you rather have time and no channel, uncomment here (TZYX)
            elif layer.data.ndim == 5:  # CTZYX
                default_dimensions = [
                    dimensions_str[0],
                    dimensions_str[2],
                    dimensions_str[1],
                    dimensions_str[3],
                    dimensions_str[4],
                ]
            else:
                default_dimensions = ["None", "None", "None", "None", "None"]

            comboboxes = [
                self._reorganize_dims_nb_timepoints_combobox,
                self._reorganize_dims_nb_channels_combobox,
                self._reorganize_dims_depth_combobox,
                self._reorganize_dims_Y_combobox,
                self._reorganize_dims_X_combobox,
            ]

            for combobox, value in zip(comboboxes, default_dimensions):
                combobox.choices = dimensions_str
                combobox.value = value

    def _update_target_axis_choices(self, event):

        if event == "XY":
            self._align_major_axis_target_axis_combo.choices = ["Y", "X"]
            self._align_major_axis_target_axis_combo.value = "Y"
        elif event == "XZ":
            self._align_major_axis_target_axis_combo.choices = ["Z", "X"]
            self._align_major_axis_target_axis_combo.value = "Z"
        else:
            self._align_major_axis_target_axis_combo.choices = ["Y", "Z"]
            self._align_major_axis_target_axis_combo.value = "Y"

    def _run_remove_labels_outside_of_mask(self):

        mask_layer, _ = self._assert_basic_layer_properties(
            self._mask_layer_combo.value, ["Image"]
        )
        if mask_layer is None:
            return

        labels_layer, _ = self._assert_basic_layer_properties(
            self._labels_layer_combo.value, ["Labels"]
        )
        if labels_layer is None:
            return

        assert (
            mask_layer is not None and labels_layer is not None
        ), "Please select both mask and labels layers"
        assert (
            mask_layer.data.shape == labels_layer.data.shape
        ), "Mask and labels must have the same shape"

        func_params = {
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        labels_cropped = remove_labels_outside_of_mask(
            labels=labels_layer.data, mask=mask_layer.data, **func_params
        )
        napari.utils.notifications.show_info(
            f"Removing labels took {time.time() - start_time} seconds"
        )

        old_name = labels_layer.name
        name = f"{old_name}_curated"

        if self._overwrite_checkbox.value:
            labels_layer.data = labels_cropped
            labels_layer.name = name
        else:
            self._viewer.add_labels(
                labels_cropped,
                name=name,
                **self._transmissible_labels_layer_properties(labels_layer),
            )
            self._labels_layer_combo.native.setCurrentIndex(
                self._labels_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "mask": (mask_layer.name, "Image"),
                "labels": (old_name, "Labels"),
            }
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("curated_labels", (name, "Labels"))]
            )
            self._recorder.record(
                function_name="remove_labels_outside_of_mask",
                func_params=func_params,
                main_input_param_name="labels",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _run_crop_array_using_mask(self):

        mask_layer, _ = self._assert_basic_layer_properties(
            self._mask_layer_combo.value, ["Image"]
        )
        if mask_layer is None:
            return

        array_layer, layer_type = self._assert_basic_layer_properties(
            self._array_layer_combo.value, ["Image", "Labels"]
        )
        if array_layer is None:
            return

        if array_layer.data.dtype == bool:
            layer_type = "Mask"

        func_params = {
            "margin": int(self._crop_array_using_mask_margin_checkbox.value),
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        array = crop_array_using_mask(
            mask=mask_layer.data, array=array_layer.data, **func_params
        )
        napari.utils.notifications.show_info(
            f"Cropping took {time.time() - start_time} seconds"
        )

        old_name = array_layer.name
        name = f"{old_name}_{self._adjective_dict['crop_array_using_mask']}"

        if self._overwrite_checkbox.value:
            array_layer.data = array
            array_layer.name = name
        else:
            if layer_type in ("Image", "Mask"):
                self._viewer.add_image(
                    array,
                    name=name,
                    **self._transmissible_image_layer_properties(array_layer),
                )
            else:
                self._viewer.add_labels(
                    array,
                    name=name,
                    **self._transmissible_labels_layer_properties(array_layer),
                )
            self._array_layer_combo.native.setCurrentIndex(
                self._array_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "mask": (mask_layer.name, "Image"),
                "array": (old_name, layer_type),
            }

            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("cropped_array", (name, layer_type))]
            )

            self._recorder.record(
                function_name="crop_array_using_mask",
                func_params=func_params,
                main_input_param_name="array",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _run_masked_gaussian_smoothing(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ["Image"]
        )
        if layer is None:
            return

        mask_available = self._mask_layer_combo.value is not None

        if mask_available:
            mask_layer, _ = self._assert_basic_layer_properties(
                self._mask_layer_combo.value, ["Image"]
            )
            if mask_layer is None:
                return
            mask_layer_data = mask_layer.data
            assert (
                mask_layer_data.shape == layer.data.shape
            ), "Mask and data must have the same shape"
        else:
            mask_layer_data = None

        mask_for_volume_available = (
            self._mask_for_volume_layer_combo.value is not None
        )

        if mask_for_volume_available:
            mask_for_volume_layer, _ = self._assert_basic_layer_properties(
                self._mask_for_volume_layer_combo.value, ["Image"]
            )
            if mask_for_volume_layer is None:
                return
            mask_for_volume_layer_data = mask_for_volume_layer.data
            assert (
                mask_for_volume_layer_data.shape == layer.data.shape
            ), "Mask (volume) and data must have the same shape"
        else:
            mask_for_volume_layer_data = None

        sigma = self._masked_smoothing_sigma_slider.value

        func_params = {
            "sigmas": sigma,
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        smoothed_array = masked_gaussian_smoothing(
            layer.data,
            mask=mask_layer_data,
            mask_for_volume=mask_for_volume_layer_data,
            **func_params,
        )
        napari.utils.notifications.show_info(
            f"Smoothing took {time.time() - start_time} seconds"
        )

        name = (
            f"{layer.name}_{self._adjective_dict['masked_gaussian_smoothing']}"
        )

        if self._overwrite_checkbox.value:
            layer.data = smoothed_array
            layer.name = name

        else:
            self._viewer.add_image(
                smoothed_array,
                name=name,
                **self._transmissible_image_layer_properties(layer),
            )
            self._image_layer_combo.native.setCurrentIndex(
                self._image_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "image": (layer.name, "Image")
            }
            if mask_available:
                input_params_to_layer_names_and_types_dict["mask"] = (
                    mask_layer.name,
                    "Image",
                )
            if mask_for_volume_available:
                input_params_to_layer_names_and_types_dict[
                    "mask_for_volume"
                ] = (
                    mask_for_volume_layer.name,
                    "Image",
                )
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("smoothed_image", (name, "Image"))]
            )
            self._recorder.record(
                function_name="masked_gaussian_smoothing",
                func_params=func_params,
                main_input_param_name="image",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

    def _reset_macro_widgets(self):
        self._macro_widgets = {}
        self._macro_tab_container.clear()

        self._macro_tab_container.extend(
            [
                Label(value="<u>Recording macro</u>"),
                Label(value="Path to save the macro json file:"),
                self._record_parameters_path,
                self._record_parameters_button,
                EmptyWidget(),
                Label(value="<u>Running macro</u>"),
                Label(value="Paths to macro parameters:"),
                self._run_macro_parameters_path,
            ]
        )

    def _update_macro_widgets(self):
        parameters_path = self._run_macro_parameters_path.value

        self._reset_macro_widgets()
        if parameters_path != "." and os.path.isfile(parameters_path):
            with open(parameters_path) as file:
                recorded_functions_calls_list = json.loads(file.read())

            self._processing_graph = ProcessingGraph(
                recorded_functions_calls_list=recorded_functions_calls_list
            )

            root_layer_ids_to_types_dict = (
                self._processing_graph.roots_layers_ids
            )

            for input_id, (layer_id, input_types) in enumerate(
                root_layer_ids_to_types_dict.items(), start=1
            ):

                input_types = [
                    elem for elem in input_types if elem != "undefined"
                ]

                widget = create_widget(
                    widget_type="FileEdit", options={"mode": "d"}
                )
                widget.native.children()[1].setPlaceholderText(
                    f"Path to folder ({input_types} {input_id})"
                )
                self._macro_tab_container.append(
                    Label(value=f"Path to folder ({input_types} {input_id}):")
                )
                self._macro_tab_container.append(widget)
                self._macro_widgets[layer_id] = widget

            self._macro_tab_container.extend(
                [
                    EmptyWidget(),
                    Label(value="Path to save outputs folders of tifs:"),
                    self._run_macro_save_path,
                    EmptyWidget(),
                    self._run_macro_compress_checkbox,
                    # self._run_macro_save_all_checkbox,
                    EmptyWidget(),
                    Label(value="Don't forget to update the jobs slider!"),
                    self._run_macro_button,
                ]
            )

    def _path_to_files(self, path_to_folder, file_type="tif"):
        return natsorted(glob.glob(f"{path_to_folder}/*.{file_type}"))

    def _run_macro(self):
        parameters_path = self._run_macro_parameters_path.value
        save_path = self._run_macro_save_path.value

        if parameters_path == "." or not os.path.exists(parameters_path):
            warnings.warn("Please enter a path to the macro parameters")
            return
        if save_path == "." or not os.path.isdir(save_path):
            warnings.warn("Please enter a path to save the outputs")
            return

        compress_params = {}
        if self._run_macro_compress_checkbox.value:
            compress_params.update({"compression": ("zlib", 1)})

        layer_id_to_folder_path_dict = {
            layer_id: widget.value
            for layer_id, widget in self._macro_widgets.items()
        }

        for layer_id, path in layer_id_to_folder_path_dict.items():
            if path == "." or not os.path.exists(path):
                warnings.warn(f"Please enter a path for layer {layer_id}")
                return
            files = self._path_to_files(path)
            if not files:
                warnings.warn(f"No tif files found in folder {path}")
                return

        for node_function in self._processing_graph.nodes_functions.values():
            function_name = node_function.function_name
            func_params = node_function.func_params
            func_params["n_jobs"] = self._n_jobs_slider.value
            function = self._func_name_to_func[function_name]

            input_params_to_layer_ids_dict = (
                node_function.input_params_to_layer_ids_dict
            )
            # this should be an OrderedDict
            output_params_to_layer_ids_dict = (
                node_function.output_params_to_layer_ids_dict
            )

            main_param = node_function.main_input_param_name
            main_layer_folder_name = os.path.basename(
                layer_id_to_folder_path_dict[
                    input_params_to_layer_ids_dict[main_param]
                ]
            )
            # create a folder for the outputs of this function
            folder_name = f"{main_layer_folder_name}_{self._adjective_dict[function_name]}"
            if len(output_params_to_layer_ids_dict) == 1:
                folder_path = f"{save_path}/{folder_name}"
                self._create_folder_if_needed(folder_path)
                layer_id = next(iter(output_params_to_layer_ids_dict.values()))
                layer_id_to_folder_path_dict.update({layer_id: folder_path})
            else:
                # for i in range(1, len(output_params_to_layer_ids_dict) + 1):
                for i, layer_id in enumerate(
                    output_params_to_layer_ids_dict.values(), start=1
                ):
                    subfolder_name = f"{folder_name}_{i}"
                    folder_path = f"{save_path}/{subfolder_name}"
                    self._create_folder_if_needed(folder_path)
                    layer_id_to_folder_path_dict.update(
                        {layer_id: folder_path}
                    )

            input_params_to_list_of_tifpaths_dict = {
                param: self._path_to_files(
                    layer_id_to_folder_path_dict[layer_id]
                )
                for param, layer_id in input_params_to_layer_ids_dict.items()
            }

            n_tifs = len(input_params_to_list_of_tifpaths_dict[main_param])

            parallel_function = build_parallel_function(
                function,
                input_params_to_list_of_tifpaths_dict,
                output_params_to_layer_ids_dict,
                layer_id_to_folder_path_dict,
                func_params,
                compress_params,
            )

            # * functions that need to be ran sequentially
            if function_name in []:
                for i in tqdm(
                    range(n_tifs), desc=f"Processing {function_name}"
                ):
                    parallel_function(i)
            # * functions that need special wrapping (e.g cropping)
            elif function_name == "reorganize_array_dimensions":
                if func_params["bool_seperate_channels"]:
                    output_id = next(
                        iter(output_params_to_layer_ids_dict.values())
                    )
                    output_folders = layer_id_to_folder_path_dict[output_id]
                else:
                    output_ids = list(output_params_to_layer_ids_dict.values())
                    output_folders = [
                        layer_id_to_folder_path_dict[output_id]
                        for output_id in output_ids
                    ]
                output_folder = layer_id_to_folder_path_dict[output_id]
                reorganize_array_dimensions_from_files(
                    input_params_to_list_of_tifpaths_dict["array"],
                    output_folders,
                    compress_params,
                    func_params,
                )

            elif function_name == "crop_array_using_mask":
                output_id = next(
                    iter(output_params_to_layer_ids_dict.values())
                )
                output_folder = layer_id_to_folder_path_dict[output_id]
                crop_array_using_mask_from_files(
                    input_params_to_list_of_tifpaths_dict["mask"],
                    input_params_to_list_of_tifpaths_dict["array"],
                    output_folder,
                    compress_params,
                    func_params,
                )
            elif function_name == "align_array_major_axis":
                output_id = next(
                    iter(output_params_to_layer_ids_dict.values())
                )
                output_folder = layer_id_to_folder_path_dict[output_id]
                align_array_major_axis_from_files(
                    input_params_to_list_of_tifpaths_dict["mask"],
                    input_params_to_list_of_tifpaths_dict["array"],
                    output_folder,
                    compress_params,
                    func_params,
                )
            elif function_name == "segment_stardist":
                # stardist is ran sequentially but needs special
                # wrapping for gpu memory management
                output_id = next(
                    iter(output_params_to_layer_ids_dict.values())
                )
                output_folder = layer_id_to_folder_path_dict[output_id]
                segment_stardist_from_files(
                    input_params_to_list_of_tifpaths_dict["image"],
                    output_folder,
                    compress_params,
                    func_params,
                )
            # * functions that can be ran in parallel
            else:
                process_map(
                    parallel_function,
                    range(n_tifs),
                    chunksize=1,
                    max_workers=func_params["n_jobs"],
                    desc=f"Processing {function_name}",
                )

        napari.utils.notifications.show_info("Macro processing finished!")

    def _create_folder_if_needed(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def _save_array_to_tif_folder(self, folder_path, array, compress_params):

        num_frames = array.shape[0]
        num_zeros = int(np.ceil(np.log10(num_frames)))

        for i, array_slice in enumerate(array):
            tifffile.imwrite(
                f"{folder_path}/frame_{i:0{num_zeros}d}.tif",
                array_slice,
                **compress_params,
            )

    def _read_folder_of_tifs(self, folder_path):
        tif_files = glob.glob(f"{folder_path}/*.tif")
        tif_files.sort()
        sample_dtype = tifffile.imread(tif_files[0]).dtype
        return np.array(
            [tifffile.imread(tif_file) for tif_file in tif_files],
            dtype=sample_dtype,
        )


def parallel_function(
    i,
    function,
    input_params_to_list_of_tifpaths_dict,
    output_params_to_layer_ids_dict,
    layer_id_to_folder_path_dict,
    func_params,
    compress_params,
):
    input_param_to_tif_dict = {
        param: tifffile.imread(tif_paths[i])
        for param, tif_paths in input_params_to_list_of_tifpaths_dict.items()
    }

    function_result = function(**input_param_to_tif_dict, **func_params)

    if len(output_params_to_layer_ids_dict) == 1:
        output_param, output_layer_id = next(
            iter(output_params_to_layer_ids_dict.items())
        )
        output_folder_path = layer_id_to_folder_path_dict[output_layer_id]

        tifffile.imwrite(
            f"{output_folder_path}/{output_param}_{i:>04}.tif",
            function_result,
            **compress_params,
        )
    else:
        for (output_param, output_layer_id), result in zip(
            output_params_to_layer_ids_dict.items(), function_result
        ):
            output_folder_path = layer_id_to_folder_path_dict[output_layer_id]

            tifffile.imwrite(
                f"{output_folder_path}/{output_param}_{i:>04}.tif",
                result,
                **compress_params,
            )


def build_parallel_function(
    function,
    input_params_to_list_of_tifpaths_dict,
    output_params_to_layer_ids_dict,
    layer_id_to_folder_path_dict,
    func_params,
    compress_params,
):

    parallel_func = partial(
        parallel_function,
        function=function,
        input_params_to_list_of_tifpaths_dict=input_params_to_list_of_tifpaths_dict,
        output_params_to_layer_ids_dict=output_params_to_layer_ids_dict,
        layer_id_to_folder_path_dict=layer_id_to_folder_path_dict,
        func_params=func_params,
        compress_params=compress_params,
    )

    return parallel_func


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    widget = TapenadeProcessingWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
