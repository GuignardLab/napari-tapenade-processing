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
    local_image_equalization,
    normalize_intensity,
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
            label="Array",
            annotation="napari.layers.Layer",
            options={"nullable": False},
        )

        self._image_layer_combo = create_widget(
            label="Image",
            annotation="napari.layers.Image",
            options={"nullable": True},
        )

        self._ref_image_layer_combo = create_widget(
            label="Image (ref)",
            annotation="napari.layers.Image",
            options={"nullable": True},
        )

        self._mask_layer_combo = create_widget(
            label="Mask",
            annotation="napari.layers.Image",
            options={"nullable": True},
        )

        self._labels_layer_combo = create_widget(
            label="Labels",
            annotation="napari.layers.Labels",
            options={"nullable": True},
        )

        layer_combos_container = Container(
            widgets=[
                self._array_layer_combo,
                self._image_layer_combo,
                self._ref_image_layer_combo,
                self._mask_layer_combo,
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
        self._labels_layer_combo.bind(self._bind_layer_combo)

        if True:
            # Making array isotropic
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

            # Spectral filtering
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
            self._compute_mask_method_combo = create_widget(
                label="Method",
                options={"choices": ["otsu", "snp otsu"], "value": "snp otsu"},
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

            self._compute_mask_threshold_factor_slider = create_widget(
                widget_type="FloatSlider",
                label="Threshold factor",
                options={"min": 0.5, "max": 1.5, "value": 1},
            )
            compute_mask_threshold_factor_tooltip = (
                "Multiplicative factor applied to the threshold computed by the chosen method\n"
                "Usually only if the mask is too inclusive (put factor > 1) or exclusive (put factor < 1)."
            )

            compute_mask_threshold_factor_container = (
                self._add_tooltip_button_to_container(
                    self._compute_mask_threshold_factor_slider,
                    compute_mask_threshold_factor_tooltip,
                )
            )

            self._convex_hull_checkbox = create_widget(
                widget_type="CheckBox",
                label="Compute convex hull",
                options={"value": False},
            )

            convex_hull_checkbox_tooltip = (
                "Returns the convex hull of the mask. Really slow."
            )
            convex_hull_container = self._add_tooltip_button_to_container(
                self._convex_hull_checkbox, convex_hull_checkbox_tooltip
            )

            self._registered_image_checkbox = create_widget(
                widget_type="CheckBox",
                label="Registered image",
                options={"value": False},
            )
            registered_image_tooltip = (
                "If checked, the image is assumed to have large areas of 0s outside of the tapenade.\n"
                "These values will be masked"
            )

            registered_image_container = self._add_tooltip_button_to_container(
                self._registered_image_checkbox, registered_image_tooltip
            )

            self._compute_mask_container = Container(
                widgets=[
                    compute_mask_method_container,
                    compute_mask_sigma_blur_container,
                    compute_mask_threshold_factor_container,
                    convex_hull_container,
                    registered_image_container,
                ],
                labels=False,
            )

            # Local equalization
            self._local_norm_box_size_slider = create_widget(
                widget_type="IntSlider",
                label="Box size",
                options={"min": 3, "max": 25, "value": 10},
            )
            local_norm_box_size_tooltip = (
                "Size of the box used for the local equalization\n"
                "A good default is ~ 3/2 * object radius."
            )

            local_norm_box_size_container = (
                self._add_tooltip_button_to_container(
                    self._local_norm_box_size_slider,
                    local_norm_box_size_tooltip,
                )
            )

            self._local_norm_percentiles_slider = create_widget(
                widget_type="FloatRangeSlider",
                label="Percentiles",
                options={"min": 0, "max": 100, "value": [1, 99]},
            )
            local_norm_percentiles_tooltip = (
                "Percentiles used for the local equalization."
            )

            local_norm_percentiles_container = (
                self._add_tooltip_button_to_container(
                    self._local_norm_percentiles_slider,
                    local_norm_percentiles_tooltip,
                )
            )

            self._local_equalization_container = Container(
                widgets=[
                    local_norm_box_size_container,
                    local_norm_percentiles_container,
                ],
                labels=False,
            )

            # Intensity normalization
            self._int_norm_sigma_slider = create_widget(
                widget_type="IntSlider",
                label="Sigma\n(0=automatic)",
                options={"min": 0, "max": 30, "value": 20},
            )

            int_norm_sigma_tooltip = (
                "Sigma for the multiscale gaussian smoothing used to normalize the reference signal.\n"
                "If 0, the sigma is automatically computed."
            )

            int_norm_sigma_container = self._add_tooltip_button_to_container(
                self._int_norm_sigma_slider, int_norm_sigma_tooltip
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

            int_norm_width_container = self._add_tooltip_button_to_container(
                self._int_norm_width_slider, int_norm_width_tooltip
            )

            self._int_norm_container = Container(
                widgets=[
                    int_norm_sigma_container,
                    int_norm_width_container,
                ],
                labels=False,
            )

            # Aligning major axis
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
            self._remove_labels_outside_of_mask_container = Container(
                widgets=[
                    EmptyWidget(),
                ],
            )

            # Cropping array using mask
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
            self._masked_gaussian_smoothing_container = Container(
                widgets=[
                    EmptyWidget(),
                    Label(value="Not implemented yet."),
                    Label(value="Under construction."),
                    EmptyWidget(),
                ],
                labels=False,
            )

            self._func_name_to_func = {
                "change_array_pixelsize": change_array_pixelsize,
                "compute_mask": compute_mask,
                "local_image_equalization": local_image_equalization,
                "align_array_major_axis": align_array_major_axis,
                "remove_labels_outside_of_mask": remove_labels_outside_of_mask,
                "crop_array_using_mask": crop_array_using_mask,
                "normalize_intensity": normalize_intensity,
            }

            self._funcs_combobox_text_to_containers = OrderedDict(
                [
                    ("Change layer voxelsize", self._rescale_container),
                    ("Spectral filtering", self._spectral_filtering_container),
                    ("Compute mask from image", self._compute_mask_container),
                    (
                        "Local image equalization",
                        self._local_equalization_container,
                    ),
                    ("Intensity normalization", self._int_norm_container),
                    (
                        "Align layer from mask major axis",
                        self._align_major_axis_container,
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
                    ("Change layer voxelsize", self._run_rescale),
                    ("Spectral filtering", None),
                    ("Compute mask from image", self._run_compute_mask),
                    (
                        "Local image equalization",
                        self._run_local_equalization,
                    ),
                    ("Intensity normalization", self._run_normalize_intensity),
                    (
                        "Align layer from mask major axis",
                        self._run_align_major_axis,
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
                        None,
                    ),
                ]
            )

            self._funcs_combobox_text_to_visible_layers = {
                "Change layer voxelsize": ["array"],
                "Spectral filtering": [],
                "Compute mask from image": ["image"],
                "Local image equalization": ["image", "mask"],
                "Intensity normalization": [
                    "image",
                    "ref_image",
                    "mask",
                    "labels",
                ],
                "Align layer from mask major axis": [
                    "array",
                    "mask",
                ],
                "Remove labels outside of mask": ["mask", "labels"],
                "Crop layers using mask": ["array", "mask"],
                "Masked gaussian smoothing": [],
            }

            self._adjective_dict = {
                "change_array_pixelsize": "rescaled",
                "compute_mask": "mask",
                "local_image_equalization": "equalized",
                "align_array_major_axis": "aligned",
                "remove_labels_outside_of_mask": "curated",
                "crop_array_using_mask": "cropped",
                "normalize_intensity": "normalized",
                "masked_gaussian_smoothing": "smoothed",
                "spectral_filtering": "filtered",
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
            "Increasing this number will speed up the computation,"
            "but can dramatically increase the amount of memory used.\n"
            'When running functions in the "Functions" tab, parallel computation'
            "is triggered if the input arrays are detected as being temporal.\n"
            'When running a macro in the "Macro recording" tab, each frame is'
            "processed in parallel."
        )

        self._add_tooltip_button_to_container(
            self._n_jobs_container, n_jobs_tooltip
        )

        self._function_tab_container = Container(
            widgets=[
                self._n_jobs_container,
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
        pixmap = pixmap.scaled(150, 112, transformMode=Qt.SmoothTransformation)
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
                    f'</small><br><br><tt><a href="https://www.{link_website}" style="color:gray;">{link_website}</a></tt>'
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
            "If checked, the new layers will overwrite the previous ones.\n"
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
        self.layout().addWidget(tabs)
        self.layout().addStretch(1)

        self._disable_irrelevant_layers(0)
        self._update_layer_combos()

    def _add_tooltip_button_to_container(self, container, tooltip_text):
        button = HoverTooltipButton(tooltip_text)
        button.native = button

        if isinstance(container, Container):
            container.append(button)
        else:
            if isinstance(container, CheckBox):
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

        previous_texts = []

        # clear all combos and add None
        for c in (
            self._array_layer_combo,
            self._image_layer_combo,
            self._ref_image_layer_combo,
            self._mask_layer_combo,
            self._labels_layer_combo,
            # self._tracks_layer_combo
        ):
            previous_texts.append(c.native.currentText())
            c.native.clear()
            # check if combo is nullable
            if c.native.property("nullable"):
                c.native.addItem(None)

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

        # reset combo current choice to previous text if possible
        for index_c, c in enumerate(
            [
                self._array_layer_combo,
                self._image_layer_combo,
                self._ref_image_layer_combo,
                self._mask_layer_combo,
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

        for layer_type in ["array", "image", "ref_image", "mask", "labels"]:
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
            raise ValueError(msg)

        if layer.data.ndim not in (3, 4):
            msg = "The layer must be 3D (ZYX) or 3D+time (TZYX)"
            napari.utils.notifications.show_warning(msg)
            raise ValueError(msg)

        layer_type = self._identify_layer_type(layer)
        if layer_type not in allowed_types:
            msg = f"The layer must be part of {allowed_types}"
            napari.utils.notifications.show_warning(msg)
            raise ValueError(msg)

        return layer, layer_type

    def _transmissive_image_layer_properties(
        self, layer: "napari.layers.Image"
    ):
        return {
            "contrast_limits": layer.contrast_limits,
            "gamma": layer.gamma,
            "colormap": layer.colormap,
            "blending": layer.blending,
            "opacity": layer.opacity,
        }

    def _transmissive_labels_layer_properties(
        self, layer: "napari.layers.Labels"
    ):
        return {
            "color": layer.color,
            "blending": layer.blending,
            "opacity": layer.opacity,
        }

    def _run_rescale(self):

        layer, layer_type = self._assert_basic_layer_properties(
            self._array_layer_combo.value, ["Image", "Labels"]
        )
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
        print(f"Array rescaling took {time.time() - start_time} seconds")

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
                    **self._transmissive_image_layer_properties(layer),
                )
            else:
                self._viewer.add_labels(
                    result_array,
                    name=name,
                    **self._transmissive_labels_layer_properties(layer),
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

            if self._macro_graph is not None:
                self._update_graph_widget()

    def _run_compute_mask(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ["Image"]
        )

        func_params = {
            "method": self._compute_mask_method_combo.value,
            "sigma_blur": self._compute_mask_sigma_blur_slider.value,
            "threshold_factor": self._compute_mask_threshold_factor_slider.value,
            "compute_convex_hull": self._convex_hull_checkbox.value,
            "registered_image": self._registered_image_checkbox.value,
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        mask = compute_mask(layer.data, **func_params)
        print(f"Mask computation took {time.time() - start_time} seconds")

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

            if self._macro_graph is not None:
                self._update_graph_widget()

    def _run_local_equalization(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ["Image"]
        )

        mask_available = self._mask_layer_combo.value is not None

        if mask_available:
            mask_layer, _ = self._assert_basic_layer_properties(
                self._mask_layer_combo.value, ["Image"]
            )
            mask_layer_data = mask_layer.data
            assert (
                mask_layer_data.shape == layer.data.shape
            ), "Mask and data must have the same shape"
        else:
            mask_layer_data = None

        perc_low, perc_high = self._local_norm_percentiles_slider.value

        func_params = {
            "perc_low": perc_low,
            "perc_high": perc_high,
            "box_size": self._local_norm_box_size_slider.value,
            "n_jobs": self._n_jobs_slider.value,
        }

        start_time = time.time()
        equalized_array = local_image_equalization(
            layer.data, mask=mask_layer_data, **func_params
        )
        print(f"Local equalization took {time.time() - start_time} seconds")

        if mask_layer_data is not None:
            equalized_array = np.where(mask_layer_data, equalized_array, 0.0)

        name = (
            f"{layer.name}_{self._adjective_dict['local_image_equalization']}"
        )

        if self._overwrite_checkbox.value:
            layer.data = equalized_array
            layer.contrast_limits = (0, 1)
            layer.name = name
        else:
            image_properties = self._transmissive_image_layer_properties(layer)
            image_properties["contrast_limits"] = (0, 1)
            self._viewer.add_image(
                equalized_array,
                name=name,
                **image_properties,
            )

            self._image_layer_combo.native.setCurrentIndex(
                self._image_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "image": (layer.name, "Image"),
                "mask": (
                    mask_layer.name if mask_layer is not None else None,
                    "Mask",
                ),
            }
            output_params_to_layer_names_and_types_dict = OrderedDict(
                [("equalized_image", (name, "Image"))]
            )
            self._recorder.record(
                function_name="local_image_equalization",
                func_params=func_params,
                main_input_param_name="image",
                input_params_to_layer_names_and_types_dict=input_params_to_layer_names_and_types_dict,
                output_params_to_layer_names_and_types_dict=output_params_to_layer_names_and_types_dict,
            )

            if self._macro_graph is not None:
                self._update_graph_widget()

    def _run_normalize_intensity(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ["Image"]
        )

        ref_layer, _ = self._assert_basic_layer_properties(
            self._ref_image_layer_combo.value, ["Image"]
        )

        mask_available = self._mask_layer_combo.value is not None

        if mask_available:
            mask_layer, _ = self._assert_basic_layer_properties(
                self._mask_layer_combo.value, ["Image"]
            )
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
            labels_layer_data = labels_layer.data
            assert (
                labels_layer_data.shape == layer.data.shape
            ), "Labels and data must have the same shape"
        else:
            labels_layer_data = None

        sigma = self._int_norm_sigma_slider.value
        if sigma == 0:
            sigma = None
        width = self._int_norm_width_slider.value

        func_params = {
            "sigma": sigma,
            "width": width,
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
        print(
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
                **self._transmissive_image_layer_properties(layer),
            )

            self._image_layer_combo.native.setCurrentIndex(
                self._image_layer_combo.native.count() - 1
            )

        if self._is_recording_parameters:
            input_params_to_layer_names_and_types_dict = {
                "image": (old_name, "Image"),
                "ref_image": (ref_layer.name, "Image"),
                "mask": (
                    mask_layer.name if mask_available else None,
                    "Mask",
                ),
                "labels": (
                    labels_layer.name if labels_available else None,
                    "Labels",
                ),
            }
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

            if self._macro_graph is not None:
                self._update_graph_widget()

    def _run_align_major_axis(self):

        mask_layer, _ = self._assert_basic_layer_properties(
            self._mask_layer_combo.value, ["Image"]
        )

        array_layer, layer_type = self._assert_basic_layer_properties(
            self._array_layer_combo.value, ["Image", "Labels"]
        )

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
        print(f"Alignment took {time.time() - start_time} seconds")

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
                    **self._transmissive_image_layer_properties(array_layer),
                )
            else:
                self._viewer.add_labels(
                    array,
                    name=name,
                    **self._transmissive_labels_layer_properties(array_layer),
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

            if self._macro_graph is not None:
                self._update_graph_widget()

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

        labels_layer, _ = self._assert_basic_layer_properties(
            self._labels_layer_combo.value, ["Labels"]
        )

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
        print(f"Removing labels took {time.time() - start_time} seconds")

        old_name = labels_layer.name
        name = f"{old_name}_curated"

        if self._overwrite_checkbox.value:
            labels_layer.data = labels_cropped
            labels_layer.name = name
        else:
            self._viewer.add_labels(
                labels_cropped,
                name=name,
                **self._transmissive_labels_layer_properties(labels_layer),
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

            if self._macro_graph is not None:
                self._update_graph_widget()

    def _run_crop_array_using_mask(self):

        mask_layer, _ = self._assert_basic_layer_properties(
            self._mask_layer_combo.value, ["Image"]
        )

        array_layer, layer_type = self._assert_basic_layer_properties(
            self._array_layer_combo.value, ["Image", "Labels"]
        )

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
        print(f"Cropping took {time.time() - start_time} seconds")

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
                    **self._transmissive_image_layer_properties(array_layer),
                )
            else:
                self._viewer.add_labels(
                    array,
                    name=name,
                    **self._transmissive_labels_layer_properties(array_layer),
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

            if self._macro_graph is not None:
                self._update_graph_widget()

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
                    Label(value="Path to save outputs folders of tifs:"),
                    self._run_macro_save_path,
                    self._run_macro_compress_checkbox,
                    self._run_macro_save_all_checkbox,
                    self._n_jobs_container,
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
            # * functions that can be ran in parallel
            else:
                process_map(
                    parallel_function,
                    range(n_tifs),
                    chunksize=1,
                    max_workers=func_params["n_jobs"],
                    desc=f"Processing {function_name}",
                )

        napari.utils.notifications.show_info("Macro processing finished")

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
