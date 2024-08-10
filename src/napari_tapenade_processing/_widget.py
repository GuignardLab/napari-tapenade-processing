import glob
import json
import os
import time # type: ignore
import warnings
from collections import OrderedDict
from datetime import datetime
from os import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING

import napari.utils
import numpy as np
import tifffile
from magicgui.widgets import (ComboBox, Container, EmptyWidget, Label,
                              create_widget)
from napari.layers import Image
from tapenade.preprocessing import (align_array_major_axis,
                                                  compute_mask,
                                                  crop_array_using_mask,
                                                  local_image_equalization,
                                                  change_arrays_pixelsize,
                                                  normalize_intensity)
from tapenade.preprocessing.segmentation_postprocessing import \
    remove_labels_outside_of_mask
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                            QLabel, QPushButton, QScrollArea, QSizePolicy,
                            QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from napari_tapenade_processing._macro_recorder import MacroRecorder

if TYPE_CHECKING:
    import napari

import napari

"""
! TODO:
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

        self._funcs_dict = {
            'change_arrays_pixelsize': change_arrays_pixelsize,
            'compute_mask': compute_mask,
            'local_image_equalization': local_image_equalization,
            'align_array_major_axis': align_array_major_axis,
            'remove_labels_outside_of_mask': remove_labels_outside_of_mask,
            'crop_array_using_mask': crop_array_using_mask,
            'normalize_intensity': normalize_intensity,
        }

        
        self._image_layer_combo = create_widget(
            label='Image',
            annotation="napari.layers.Image",
            options={'nullable': True}
        )

        self._ref_image_layer_combo = create_widget(
            label='Image (ref)',
            annotation="napari.layers.Image",
            options={'nullable': True}
        )

        self._mask_layer_combo = create_widget(
            label='Mask',
            annotation="napari.layers.Image",
            options={'nullable': True}
        )

        self._labels_layer_combo = create_widget(
            label='Labels',
            annotation="napari.layers.Labels",
            options={'nullable': True}
        )

        layer_combos_container = Container(
            widgets=[
                self._image_layer_combo,
                self._ref_image_layer_combo,
                self._mask_layer_combo,
                self._labels_layer_combo,
            ],
            layout='vertical',
            labels=True,
        )

 

        self._update_layer_combos()

        # self._image_layer_combo.native.currentIndexChanged.connect(self._populate_layer_combos_values)
        self._image_layer_combo.bind(self._bind_layer_combo)
        self._ref_image_layer_combo.bind(self._bind_layer_combo)
        self._mask_layer_combo.bind(self._bind_layer_combo)
        self._labels_layer_combo.bind(self._bind_layer_combo)




        if True: 
            # Making array isotropic
            self._rescale_interp_order_combo = create_widget(
                label='Interp order', 
                options={'choices':[0, 1, 3], 'value':1},
            )
            tooltip_rescale = 'Interpolation order.\n0: Nearest, 1: Linear, 3: Cubic\nBigger means slower'
            self._rescale_interp_order_combo.native.setToolTip(tooltip_rescale)

            rescale_interp_order_label = Label(value='Images interp order')
            rescale_interp_order_label.native.setToolTip(tooltip_rescale)


            rescale_interp_order_container = Container(
                widgets=[
                    rescale_interp_order_label,
                    self._rescale_interp_order_combo,
                ],
                layout='horizontal',
                labels=False,
            )

            self._rescale_input_pixelsize = create_widget(
                widget_type="TupleEdit", label='In',
                options={'value':(1.,1.,1.), 'layout':'vertical',
                'options':{'min':0}},
            )

            self._rescale_output_pixelsize = create_widget(
                widget_type="TupleEdit", label='Out',
                options={'value':(1.,1.,1.), 'layout':'vertical',
                'options':{'min':0}},
            )

            # pixel_size = self._viewer.layers[0].scale

            pixelsizes_container = Container(
                widgets=[
                    self._rescale_input_pixelsize,
                    self._rescale_output_pixelsize,
                ],
                layout='horizontal',
                labels=True,
            )

            self._rescale_container = Container(
                widgets=[
                    rescale_interp_order_container,
                    # self._rescale_interp_order_combo,
                    Label(value='Voxelsizes (ZYX):'),
                    # self._rescale_reshape_factors
                    # self._rescale_input_pixelsize,
                    # EmptyWidget(),
                    # self._rescale_output_pixelsize,
                    pixelsizes_container,
                ],
                labels=False,
            )

            # Spectral filtering
            self._spectral_filtering_container = Container(
                widgets=[
                    EmptyWidget(),
                    Label(value='Not implemented yet.'),
                    Label(value='Under construction.'),
                    EmptyWidget(),
                ],
                labels=False,
            )

            # Computing mask
            self._compute_mask_method_combo = create_widget(
                label='Method', 
                options={'choices':['otsu', 'snp otsu'], 'value':'snp otsu'},
            )
            self._compute_mask_method_combo.native.setToolTip(
                'otsu: thresholding with Otsu\'s method on blurred image.\n' \
                'snp otsu: more robust version of thresholding with Otsu\'s method. Also slower.'
            )

            self._compute_mask_sigma_blur_slider = create_widget(
                widget_type="IntSlider", label='Sigma blur',
                options={'min':1, 'max':10, 'value':3},
            )
            self._compute_mask_sigma_blur_slider.native.setToolTip(
                'Sigma of the Gaussian blur applied to the image before thresholding\n' \
                'A good default is ~ object radius/3.'
            )

            self._compute_mask_threshold_factor_slider = create_widget(
                widget_type="FloatSlider", label='Threshold factor',
                options={'min':0.5, 'max':1.5, 'value':1},
            )
            self._compute_mask_threshold_factor_slider.native.setToolTip(
                'Multiplicative factor applied to the threshold computed by the chosen method\n' \
                'Usually only if the mask is too inclusive (put factor > 1) or exclusive (put factor < 1).'
            )

            self._convex_hull_checkbox = create_widget(
                widget_type="CheckBox", label='Compute convex hull',
                options={'value': False}
            )
            self._convex_hull_checkbox.native.setToolTip(
                'Returns the convex hull of the mask. Really slow.'
            )

            self._registered_image_checkbox = create_widget(
                widget_type="CheckBox", label='Registered image',
                options={'value': False}
            )
            self._registered_image_checkbox.native.setToolTip(
                'If checked, the image is assumed to have large areas of 0s outside of the tapenade.\n' \
                'These values will be masked'
            )

            self._compute_mask_container = Container(
                widgets=[
                    self._compute_mask_method_combo,
                    self._compute_mask_sigma_blur_slider,
                    self._compute_mask_threshold_factor_slider,
                    self._convex_hull_checkbox,
                    self._registered_image_checkbox,
                ],
            )

            # Local equalization
            self._local_norm_box_size_slider = create_widget(
                widget_type="IntSlider", label='Box size',
                options={'min':3, 'max':25, 'value':10},
            )
            self._local_norm_box_size_slider.native.setToolTip(
                'Size of the box used for the local equalization\n' \
                'A good default is ~ 3/2 * object radius.'
            )

            self._local_norm_percentiles_slider = create_widget(
                widget_type="FloatRangeSlider", label='Percentiles',
                options={'min':0, 'max':100, 'value':[1, 99]},
            )
            self._local_norm_percentiles_slider.native.setToolTip(
                'Percentiles used for the local equalization.'
            )

            self._local_equalization_container = Container(
                widgets=[
                    self._local_norm_box_size_slider,
                    self._local_norm_percentiles_slider,
                ],
            )

            # Intensity normalization
            self._int_norm_sigma_slider = create_widget(
                widget_type="IntSlider", 
                label='Sigma\n(0=automatic)',
                options={'min':0, 'max':30, 'value':20},
            )

            self._int_norm_sigma_slider.native.setToolTip(
                'Sigma for the multiscale gaussian smoothing used to normalize the reference signal.\n' \
                'If 0, the sigma is automatically computed.'
            )

            self._int_norm_width_slider = create_widget(
                widget_type="IntSlider", label='Width of ref plane',
                options={'min':1, 'max':5, 'value':3},
            )
            self._int_norm_width_slider.native.setToolTip(
                'Width of the reference plane used to compute normalization values.\n' \
                'You usually don\'t need to change this.'
            )

            self._int_norm_container = Container(
                widgets=[
                    self._int_norm_sigma_slider,
                    self._int_norm_width_slider,
                ],
            )

            # Aligning major axis
            self._align_major_axis_rotation_plane_combo = create_widget(
                label='Rotation plane', 
                options={'choices':['XY', 'XZ', 'YZ'], 'value':'XY'},
            )

            self._align_major_axis_rotation_plane_combo.changed.connect(self._update_target_axis_choices)

            self._align_major_axis_target_axis_combo = create_widget(
                label='Target axis', 
                options={'choices':['Y', 'X']},
            )

            self._align_major_axis_target_axis_combo.native.setToolTip(
                'Axis to align the major axis of the mask with.'
            )

            self._align_major_axis_container = Container(
                widgets=[
                    self._align_major_axis_rotation_plane_combo,
                    self._align_major_axis_target_axis_combo,
                ],
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
                options={'value': False},
                label='Add 1 pixel margin',
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
                    Label(value='Not implemented yet.'),
                    Label(value='Under construction.'),
                    EmptyWidget(),
                ],
                labels=False,
            )

            

            self._dict_widgets = OrderedDict([
                ('Rescale layers', self._rescale_container),
                ('Spectral filtering', self._spectral_filtering_container),
                ('Compute mask from image', self._compute_mask_container),
                ('Local image equalization', self._local_equalization_container),
                ('Intensity normalization', self._int_norm_container),
                ('Align layers from mask major axis', self._align_major_axis_container),
                ('Remove labels outside of mask', self._remove_labels_outside_of_mask_container),
                ('Crop layers using mask', self._crop_array_using_mask_container),
                ('Masked gaussian smoothing', self._masked_gaussian_smoothing_container),
            ])

            self._dict_widgets_layers_visibilities = {
                'Rescale layers': ['image', 'mask', 'labels'],
                'Spectral filtering': [],
                'Compute mask from image': ['image'],
                'Local image equalization': ['image', 'mask'],
                'Intensity normalization': ['image', 'ref_image', 'mask', 'labels'],
                'Align layers from mask major axis': ['image', 'mask', 'labels'],
                'Remove labels outside of mask': ['mask', 'labels'],
                'Crop layers using mask': ['image', 'mask', 'labels'],
                'Masked gaussian smoothing': [],
            }



        self._run_button = create_widget(
            widget_type="PushButton", label='Run function'
        )

        self._run_button.clicked.connect(self._run_current_function)

        self._main_combobox = QComboBox()
        self._main_combobox._explicitly_hidden = False
        self._main_combobox.native = self._main_combobox
        self._main_combobox.name = ""

        main_stack = QStackedWidget()
        main_stack.native = main_stack
        main_stack.name = ""

        for i, (name,w) in enumerate(self._dict_widgets.items()):
            # manage layout stretch and add to main combobox
            if hasattr(w, 'native'):
                w.native.layout().addStretch()
                main_stack.addWidget(w.native)
            else:
                w.layout().addStretch()
                main_stack.addWidget(w)

            self._main_combobox.addItem(name)
            

        self._main_combobox.currentIndexChanged.connect(main_stack.setCurrentIndex)
        self._main_combobox.currentIndexChanged.connect(self._disable_irrelevant_layers)

        main_control = Container(
            widgets=[
                self._main_combobox,
                main_stack,
            ],
            labels=False,
        )

        update_layers_combos_button = create_widget(
            widget_type="PushButton", label='Refresh'
        )
        update_layers_combos_button.native.setToolTip(
            'Click refresh if a layer does not appear in the list or has a wrong name.'
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
                Label(value='<u>Layers to process:</u>'),
                EmptyWidget(),
                update_layers_combos_button,
            ],
            layout='horizontal',
            labels=False,
        )

        self._function_tab_container = Container(
            widgets=[
                label_and_update_container,
                # Label(value='<u>Layers to process:</u>'),
                layer_combos_container,
                # update_layers_combos_button,
                Label(value='<u>Processing functions:</u>'),
                main_control,
                self._run_button,
            ],
            labels=False
        )

        ### Recording of parameters
        self._record_parameters_path = create_widget(
            widget_type="FileEdit", 
            options={'mode':'d'},
            label='Macro save path'
        )
        self._record_parameters_path.native.children()[1].setPlaceholderText('Path to save the macro')

        self._record_parameters_button = create_widget(
            widget_type="PushButton", label='Record macro'
        )

        self._run_macro_parameters_path =  create_widget(
            widget_type="FileEdit", 
            options={'mode':'r', 'filter':'*.json'},
            label='Macro'
        )
        self._run_macro_parameters_path.native.children()[1].setPlaceholderText('Path to the macro json')

        self._run_macro_mask_path = create_widget(
            widget_type="FileEdit",
            options={'mode':'d'},
            label='Mask'
        )
        self._run_macro_mask_path.native.children()[1].setPlaceholderText('Path to mask folder')
            
        self._run_macro_image_path = create_widget(
            widget_type="FileEdit",
            options={'mode':'d'},
            label='Image'
        )
        self._run_macro_image_path.native.children()[1].setPlaceholderText('Path to image folder')

        self._run_macro_labels_path = create_widget(
            widget_type="FileEdit",
            options={'mode':'d'},
            label='Labels'
        )
        self._run_macro_labels_path.native.children()[1].setPlaceholderText('Path to labels folder')

        self._run_macro_tracks_path = create_widget(
            widget_type="FileEdit",
            options={'mode':'d'},
            label='Tracks'
        )
        self._run_macro_tracks_path.native.children()[1].setPlaceholderText('Path to tracks folder')

        self._run_macro_save_path = create_widget(
            widget_type="FileEdit",
            options={'mode':'d'},
            label='Save'
        )
        self._run_macro_save_path.native.children()[1].setPlaceholderText('Path to save the results')

        self._run_macro_save_all_checkbox = create_widget(
            widget_type="CheckBox",
            label='Save all intermediate results',
            options={'value': False}
        )

        self._run_macro_compress_checkbox = create_widget(
            widget_type="CheckBox",
            label='Compress when saving',
            options={'value': False}
        )

        self._run_macro_button = create_widget(
            widget_type="PushButton", label='Run macro'
        )

        self._run_macro_button.clicked.connect(self._run_macro)

        self._macro_tab_container = Container(
            widgets=[
                self._record_parameters_path,
                self._record_parameters_button,
                EmptyWidget(),
                Label(value='<u>Paths to macro parameters:</u>'),
                self._run_macro_parameters_path,
                self._run_macro_mask_path,
                self._run_macro_image_path,
                self._run_macro_labels_path,
                # self._run_macro_tracks_path,
                self._run_macro_save_path,
                self._run_macro_save_all_checkbox,
                self._run_macro_compress_checkbox,
                self._run_macro_button
            ],
            layout='vertical',
            labels=False,
        )

        adjective_dict = {
            'change_arrays_pixelsize':          'rescaled',
            'compute_mask':                  'mask',
            'local_image_equalization':     'equalized',
            'align_array_major_axis':        'aligned',
            'remove_labels_outside_of_mask': 'labels_cleared',
            'crop_array_using_mask':         'cropped',
            'normalize_intensity':           'normalized',
            'masked_gaussian_smoothing':     'smoothed',
            'spectral_filtering':            'filtered'
        }
        
        self._recorder = MacroRecorder(adjective_dict)
        self._is_recording_parameters = False

        self._record_parameters_button.clicked.connect(self._manage_recording)
        ###

        # options_text = EmptyWidget(label='<u>Options:</u>')
        # self._record_parameters_text = EmptyWidget(label='<u>Macro recording settings:</u>')
        # self._choose_layers_text = EmptyWidget(label='<u>Choose layers to process:</u>')
        # choose_function_text = EmptyWidget(label='<u>Choose processing function:</u>')
        
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

        link_website = 'morphotiss.org/'
        link_DOI = 'XXX'

        texts_container = Container(
            widgets=[
                Label(value=
                      '<small>This plugin is part of TAPENADE.<br>'\
                      f'Using it in your research ?<br>'\
                      f'Please <a href="{link_DOI}" style="color:gray;">cite us</a>.'\
                      f'</small><br><br><tt><a href="https://www.{link_website}" style="color:gray;">{link_website}</a></tt>'),
            ],
            layout='vertical',
            labels=False,
        )


        self._header_container = Container(
            widgets=[
                label,
                texts_container
            ],
            layout='horizontal',
            labels=False,
        )

        tabs = QTabWidget()
        # tabs = MultiLineTabWidget()

        ### Advanced parameters
        self._overwrite_checkbox = create_widget(
            widget_type="CheckBox", label='New layers overwrite previous ones', 
            options={'value': False}
        )

        self._overwrite_checkbox.native.setToolTip(
            'If checked, the new layers will overwrite the previous ones with the same name.\n' \
            'This can be useful to save memory.\n' \
            'If not, a suffix will be added to the new layers names.'
        )


        self._systematic_crop_checkbox = create_widget(
            widget_type="CheckBox", label='Results are cropped using mask', 
            options={'value': False}
        )

        self._systematic_crop_checkbox.native.setToolTip(
            'If checked, the results of the functions will be systematically cropped using the mask.\n' \
            'This can be useful to save memory.\n' \
            'If not, the results will have the same shape as the input layers.'
        )

        self._n_jobs_slider = create_widget(
            widget_type="IntSlider", label='# parallel jobs', 
            options={'min':1, 'max':cpu_count(), 'value':cpu_count()}
        )

        self._advanced_parameters_tab_container = Container(
            widgets=[
                self._overwrite_checkbox,
                self._systematic_crop_checkbox,
                self._n_jobs_slider,
            ],
            layout='vertical',
            labels=True,
        )
        ###


        self._advanced_parameters_tab_container.native.layout().addStretch(1)
        self._macro_tab_container.native.layout().addStretch(1)
        self._function_tab_container.native.layout().addStretch(1)

        tabs.addTab(self._function_tab_container.native, 'Functions')
        tabs.addTab(self._macro_tab_container.native, 'Macro recording')
        tabs.addTab(self._advanced_parameters_tab_container.native, 'Advanced params')
        # tabs.addTab(self._advanced_parameters_tab_container.native, 'Advanced params')
        # tabs.addTab(self._advanced_parameters_tab_container.native, 'Advanced params')
        # tabs.addTab(self._advanced_parameters_tab_container.native, 'Advanced params')

        self.setLayout(QVBoxLayout())


        self.layout().addWidget(self._header_container.native)
        self.layout().addWidget(tabs)
        self.layout().addStretch(1)

        self._disable_irrelevant_layers(0)
        self._update_layer_combos()

    def _bind_layer_combo(self, obj):
        """
        This used so that when calling layer_combo.value, we get the layer object,
        not the name of the layer
        """
        name = obj.native.currentText()
        if not(name in ('', '-----')):
            return self._viewer.layers[name]
        else:
            return None
        
        

    def _update_layer_combos(self):

        previous_texts = []

        # clear all combos and add None
        for c in (
            self._image_layer_combo,
            self._ref_image_layer_combo,
            self._mask_layer_combo,
            self._labels_layer_combo,
            #self._tracks_layer_combo
        ):
            previous_texts.append(c.native.currentText())
            c.native.clear()
            c.native.addItem(None)

        # add layers to compatible combos
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Image):
                if layer.data.dtype == bool:
                    if self._mask_layer_combo.enabled:
                        self._mask_layer_combo.native.addItem(layer.name)
                else:
                    if self._image_layer_combo.enabled:
                        self._image_layer_combo.native.addItem(layer.name)
                    if self._ref_image_layer_combo.enabled:
                        self._ref_image_layer_combo.native.addItem(layer.name)
            elif isinstance(layer, napari.layers.Labels):
                if self._labels_layer_combo.enabled:
                    self._labels_layer_combo.native.addItem(layer.name)
            # elif isinstance(layer, Tracks):
            #     self._tracks_layer_combo.addItem(layer.name)

        # reset combo to previous text if possible
        for index_c, c in enumerate([
            self._image_layer_combo,
            self._ref_image_layer_combo,
            self._mask_layer_combo,
            self._labels_layer_combo,
            #self._tracks_layer_combo
        ]):
            all_choices = [c.native.itemText(i) for i in range(c.native.count())]
            if previous_texts[index_c] in all_choices:

                # if the previous layer is None, set it to the newest layer
                if previous_texts[index_c] == c.native.itemText(0):
                    c.native.setCurrentIndex(c.native.count()-1)
                else:
                    c.native.setCurrentText(previous_texts[index_c])
            else:
                c.native.setCurrentIndex(0)

        
    def _manage_recording(self):
        path = str(self._record_parameters_path.value)

        if path == '.' or not os.path.exists(path):
            napari.utils.notifications.show_warning('Please enter a path to record the macro')
        else:
            if not self._is_recording_parameters:
                self._is_recording_parameters = True

                self._record_parameters_button.native.setText('Save macro')
                self._record_parameters_path.enabled = False
            else: # if was recording
                self._is_recording_parameters = False

                self._recorder.dump_recorded_parameters(path)

                self._record_parameters_button.native.setText('Record macro')
                self._record_parameters_path.enabled = True


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

        params = None

        function_index = self._main_combobox.currentIndex()
        if function_index == 0:
            params = self._run_rescale()
        elif function_index == 1:
            params = self._run_compute_mask()
        elif function_index == 2:
            params = self._run_local_equalization()
        elif function_index == 3:
            params = self._run_normalize_intensity()
        elif function_index == 4:
            params = self._run_align_major_axis()
        elif function_index == 5:
            params = self._run_remove_labels_outside_of_mask()
        elif function_index == 6:
            params = self._run_crop_array_using_mask()

        if self._is_recording_parameters and params is not None:
            self._record_parameters_list.append(params)
    
    
    def _disable_irrelevant_layers(self, event):

        name, _ = list(self._dict_widgets.items())[event]

        list_layers_enabled = self._dict_widgets_layers_visibilities[name]

        for layer_type in ['image', 'ref_image', 'mask', 'labels']:
            combo = getattr(self, f'_{layer_type}_layer_combo')
            combo.enabled = layer_type in list_layers_enabled



    def _identify_layer_type(self, layer: "napari.layers.Layer"):
        layer_type = layer.__class__.__name__
        if layer_type in ('Image', 'Labels'):
            return layer_type
        else:
            return 'Other'
    
    def _assert_basic_layer_properties(self, layer: "napari.layers.Layer", allowed_types: list):

        if layer is None:
            msg = 'Please select a layer'
            napari.utils.notifications.show_warning(msg)
            raise ValueError(msg)
        
        if not(layer.data.ndim in (3, 4)):
            msg = 'The layer must be 3D (ZYX) or 3D+time (TZYX)'
            napari.utils.notifications.show_warning(msg)
            raise ValueError(msg)
        
        layer_type = self._identify_layer_type(layer)
        if layer_type not in allowed_types:
            msg = f'The layer must be part of {allowed_types}'
            napari.utils.notifications.show_warning(msg)
            raise ValueError(msg)

        # assert layer is not None, 'Please select a layer'
        # assert layer.data.ndim in (3, 4), 'The layer must be 3D (ZYX) or 3D+time (TZYX)'
        # layer_type = self._identify_layer_type(layer)
        # assert layer_type in allowed_types, f'The layer must be part of {allowed_types}'

        return layer, layer_type
    
    def _transmissive_image_layer_properties(self, layer: "napari.layers.Image"):
        return {
            'contrast_limits': layer.contrast_limits, 'gamma': layer.gamma,
            'colormap': layer.colormap, 'blending': layer.blending,
            'opacity': layer.opacity,
        }
    
    def _transmissive_labels_layer_properties(self, layer: "napari.layers.Labels"):
        return {
            'color': layer.color, 'blending': layer.blending,
            'opacity': layer.opacity,
        }




    def _run_rescale(self):

        layers_properties = {
            'mask': (),
            'image': (),
            'labels': (),
        }

        for layer_type in layers_properties.keys():
            layer = getattr(self, f'_{layer_type}_layer_combo').value
            if layer is not None:
                layer, napari_type = self._assert_basic_layer_properties(
                    layer, ['Image', 'Labels']
                )
                layers_properties[layer_type] = (layer, napari_type)
            else:
                layers_properties[layer_type] = (None, None)


        if all(layer is None for layer, _ in layers_properties.values()):
            warnings.warn('Please select at least one layer')
            return

        # reshape_factors = self._rescale_reshape_factors.value
        input_pixelsize = self._rescale_input_pixelsize.value
        output_pixelsize = self._rescale_output_pixelsize.value
        
        assert not(any(factor <= 0 for factor in input_pixelsize)), 'Input voxel size must have non-zero elements'
        assert not(any(factor <= 0 for factor in output_pixelsize)), 'Output voxel size must have non-zero elements'

        func_params = {
            'order': self._rescale_interp_order_combo.value,
            'input_pixelsize': input_pixelsize,
            'output_pixelsize': output_pixelsize,
            'n_jobs': self._n_jobs_slider.value,
        }

        arrays = {
            layer_type: (value[0].data if value[0] is not None else None) \
                for layer_type, value in layers_properties.items()
        }

        start_time = time.time()
        result_arrays = change_arrays_pixelsize(
            **arrays, **func_params
        )
        print(f'Isotropization took {time.time() - start_time} seconds')

        if isinstance(result_arrays, tuple):
            multiple_results = True
            result_counter = 0
        else:
            multiple_results = False

        layers_names_out = OrderedDict()

        for layer_type, (layer, napari_type) in layers_properties.items():
            if layer is not None:

                if multiple_results:
                    array = result_arrays[result_counter]
                    result_counter += 1
                else:
                    array = result_arrays

                if self._overwrite_checkbox.value:
                    layer.data = array
                    layers_names_out[layer_type] = layer.name
                else:

                    name = f'{layer.name} rescaled'
                    layers_names_out[layer_type] = name

                    if napari_type == 'Image':
                        self._viewer.add_image(
                            array,
                            name=name,
                            **self._transmissive_image_layer_properties(layer)
                        )
                        if array.dtype == bool:
                            self._mask_layer_combo.native.setCurrentIndex(self._mask_layer_combo.native.count()-1)
                        else:
                            self._image_layer_combo.native.setCurrentIndex(self._image_layer_combo.native.count()-1)
                    else:
                        self._viewer.add_labels(
                            array,
                            name=name,
                            **self._transmissive_labels_layer_properties(layer)
                        )
                        self._labels_layer_combo.native.setCurrentIndex(self._labels_layer_combo.native.count()-1)

            else:
                layers_names_out[layer_type] = None
        
        if self._is_recording_parameters:
            layers_names_in = {
                layer_type: (v[0].name if v[0] is not None else None) \
                    for layer_type, v in layers_properties.items()
            }

            self._recorder.record(
                function_name='change_arrays_pixelsize',
                layers_names_in=layers_names_in,
                layers_names_out=layers_names_out,
                func_params=func_params,
                overwrite=self._overwrite_checkbox.value
            )

        if self._systematic_crop_checkbox.value and layers_properties['mask'][0] is not None:
            old_overwrite = self._overwrite_checkbox.value
            self._overwrite_checkbox.value = True
            self._run_crop_array_using_mask()
            self._overwrite_checkbox.value = old_overwrite

    def _run_compute_mask(self):
        
        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ['Image']
        )

        func_params = {
            'method': self._compute_mask_method_combo.value,
            'sigma_blur': self._compute_mask_sigma_blur_slider.value,
            'threshold_factor': self._compute_mask_threshold_factor_slider.value,
            'compute_convex_hull': self._convex_hull_checkbox.value,
            'registered_image': self._registered_image_checkbox.value,
            'n_jobs': self._n_jobs_slider.value,
        }

        start_time = time.time()
        mask = compute_mask(
            layer.data, **func_params
        )
        print(f'Mask computation took {time.time() - start_time} seconds')

        self._viewer.add_image(
            mask,
            name=f'{layer.name} mask',
            blending='additive',
            opacity=0.7,

        )

        self._mask_layer_combo.native.setCurrentIndex(self._mask_layer_combo.native.count()-1)
            
        if self._is_recording_parameters:
            layers_names_in = {
                'image': layer.name
            }
            layers_names_out = {
                'mask': f'{layer.name} mask'
            }

            self._recorder.record(
                function_name='compute_mask',
                layers_names_in=layers_names_in,
                layers_names_out=layers_names_out,
                func_params=func_params,
                overwrite=self._overwrite_checkbox.value
            )
        
        if self._systematic_crop_checkbox.value:
            old_overwrite = self._overwrite_checkbox.value
            self._overwrite_checkbox.value = True
            self._run_crop_array_using_mask()
            self._overwrite_checkbox.value = old_overwrite

    def _run_local_equalization(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ['Image']
        )

        layers_names_in = {
            'image': layer.name
        }

        mask_available = self._mask_layer_combo.value is not None

        if mask_available:
            mask_layer, _ = self._assert_basic_layer_properties(
                self._mask_layer_combo.value, ['Image']
            )
            mask_layer_data = mask_layer.data
            assert mask_layer_data.shape == layer.data.shape, 'Mask and data must have the same shape'

            layers_names_in['mask'] = mask_layer.name
        else:
            mask_layer_data = None
            layers_names_in['mask'] = None

        perc_low, perc_high = self._local_norm_percentiles_slider.value

        func_params = {
            'perc_low': perc_low,
            'perc_high': perc_high,
            'box_size': self._local_norm_box_size_slider.value,
            'n_jobs': self._n_jobs_slider.value,
        }

        start_time = time.time()
        equalized_array = local_image_equalization(
            layer.data,
            mask=mask_layer_data,
            **func_params
        )
        print(f'Local equalization took {time.time() - start_time} seconds')

        if mask_layer_data is not None:
            equalized_array = np.where(mask_layer_data, equalized_array, 0.0)

        if self._overwrite_checkbox.value:
            layer.data = equalized_array
            layer.contrast_limits = (0, 1)

            layers_names_out = {
                'image': layer.name
            }
        else:
            name = f'{layer.name} equalized'

            self._viewer.add_image(
                equalized_array,
                name=name,
                colormap=layer.colormap, blending=layer.blending,
                opacity=layer.opacity,
            )

            self._image_layer_combo.native.setCurrentIndex(self._image_layer_combo.native.count()-1)

            layers_names_out = {
                'image': name
            }
        
        if self._is_recording_parameters:
            self._recorder.record(
                function_name='local_image_equalization',
                layers_names_in=layers_names_in,
                layers_names_out=layers_names_out,
                func_params=func_params,
                overwrite=self._overwrite_checkbox.value
            )

        if mask_available and self._systematic_crop_checkbox.value:
            old_overwrite = self._overwrite_checkbox.value
            self._overwrite_checkbox.value = True
            self._run_crop_array_using_mask()
            self._overwrite_checkbox.value = old_overwrite

    def _run_normalize_intensity(self):

        layer, _ = self._assert_basic_layer_properties(
            self._image_layer_combo.value, ['Image']
        )

        ref_layer, _ = self._assert_basic_layer_properties(
            self._ref_image_layer_combo.value, ['Image']
        )

        layers_names_in = {
            'image': layer.name,
            'ref_image': ref_layer.name,
        }

        mask_available = self._mask_layer_combo.value is not None

        if mask_available:
            mask_layer, _ = self._assert_basic_layer_properties(
                self._mask_layer_combo.value, ['Image']
            )
            mask_layer_data = mask_layer.data
            assert mask_layer_data.shape == layer.data.shape, 'Mask and data must have the same shape'

            layers_names_in['mask'] = mask_layer.name
        else:
            mask_layer_data = None
            layers_names_in['mask'] = None

        labels_available = self._labels_layer_combo.value is not None

        if labels_available:
            labels_layer, _ = self._assert_basic_layer_properties(
                self._labels_layer_combo.value, ['Labels']
            )
            labels_layer_data = labels_layer.data
            assert labels_layer_data.shape == layer.data.shape, 'Labels and data must have the same shape'

            layers_names_in['labels'] = labels_layer.name
        else:
            labels_layer_data = None
            layers_names_in['labels'] = None

        sigma = self._int_norm_sigma_slider.value
        if sigma == 0:
            sigma = None
        width = self._int_norm_width_slider.value

        func_params = {
            'sigma': sigma,
            'width': width,
            'n_jobs': self._n_jobs_slider.value,
        }

        start_time = time.time()
        normalized_array, normalized_ref_array = normalize_intensity(
            image=layer.data,
            ref_image=ref_layer.data,
            mask=mask_layer_data,
            labels=labels_layer_data,
            **func_params
        )
        print(f'intensity normalization took {time.time() - start_time} seconds')

        if mask_layer_data is not None:
            normalized_array = np.where(mask_layer_data, normalized_array, 0.0)
            normalized_ref_array = np.where(mask_layer_data, normalized_ref_array, 0.0)
        
        if self._overwrite_checkbox.value:
            layer.data = normalized_array
            ref_layer.data = normalized_ref_array

            layers_names_out = {
                'image': layer.name,
                'ref_image': ref_layer.name,
            }
        else:
            name = f'{layer.name} normalized'
            ref_name = f'{ref_layer.name} normalized'

            self._viewer.add_image(
                normalized_array,
                name=name,
                colormap=layer.colormap, blending=layer.blending,
                opacity=layer.opacity,
            )

            self._viewer.add_image(
                normalized_ref_array,
                name=ref_name,
                colormap=ref_layer.colormap, blending=ref_layer.blending,
                opacity=ref_layer.opacity,
            )

            self._image_layer_combo.native.setCurrentIndex(self._image_layer_combo.native.count()-1)

            layers_names_out = {
                'image': name,
                'ref_image': ref_name,
            }
        
        if self._is_recording_parameters:
            self._recorder.record(
                function_name='normalize_intensity',
                layers_names_in=layers_names_in,
                layers_names_out=layers_names_out,
                func_params=func_params,
                overwrite=self._overwrite_checkbox.value
            )

        if mask_available and self._systematic_crop_checkbox.value:
            old_overwrite = self._overwrite_checkbox.value
            self._overwrite_checkbox.value = True
            self._run_crop_array_using_mask()
            self._overwrite_checkbox.value = old_overwrite


    def _run_align_major_axis(self):

        mask_layer, _ = self._assert_basic_layer_properties(
            self._mask_layer_combo.value, ['Image']
        )

        layers_properties = {
            'mask': (mask_layer, 'Image'),
        }

        for layer_type in ['image', 'labels']:
            layer = getattr(self, f'_{layer_type}_layer_combo').value
            if layer is not None:
                layer, napari_type = self._assert_basic_layer_properties(
                    layer, ['Image', 'Labels']
                )
                layers_properties[layer_type] = (layer, napari_type)

                assert layer.data.shape == mask_layer.data.shape, f'{layer_type} and mask must have the same shape'
            else:
                layers_properties[layer_type] = (None, None)

        arrays = {
            layer_type: (value[0].data if value[0] is not None else None) \
                for layer_type, value in layers_properties.items()
        }

        func_params = {
            'target_axis': self._align_major_axis_target_axis_combo.value,
            'rotation_plane': self._align_major_axis_rotation_plane_combo.value,
            'n_jobs': self._n_jobs_slider.value,
        }

        start_time = time.time()
        result_arrays = align_array_major_axis(
            **arrays, **func_params
        )
        print(f'Alignment took {time.time() - start_time} seconds')

        if isinstance(result_arrays, tuple):
            multiple_results = True
            result_counter = 0
        else:
            multiple_results = False

        layers_names_out = OrderedDict()

        for layer_type, (layer, napari_type) in layers_properties.items():
            if layer is not None:

                if multiple_results:
                    array = result_arrays[result_counter]
                    result_counter += 1
                else:
                    array = result_arrays

                if self._overwrite_checkbox.value:
                    layer.data = array
                    layers_names_out[layer_type] = layer.name
                else:

                    name = f'{layer.name} aligned'
                    layers_names_out[layer_type] = name

                    if napari_type == 'Image':
                        self._viewer.add_image(
                            array,
                            name=name,
                            **self._transmissive_image_layer_properties(layer)
                        )
                        if array.dtype == bool:
                            self._mask_layer_combo.native.setCurrentIndex(self._mask_layer_combo.native.count()-1)
                        else:
                            self._image_layer_combo.native.setCurrentIndex(self._image_layer_combo.native.count()-1)
                    else:
                        self._viewer.add_labels(
                            array,
                            name=name,
                            **self._transmissive_labels_layer_properties(layer)
                        )
                        self._labels_layer_combo.native.setCurrentIndex(self._labels_layer_combo.native.count()-1)

            else:
                layers_names_out[layer_type] = None

        if self._is_recording_parameters:
            layers_names_in = {
                layer_type: (v[0].name if v[0] is not None else None) \
                    for layer_type, v in layers_properties.items()
            }

            self._recorder.record(
                function_name='align_array_major_axis',
                layers_names_in=layers_names_in,
                layers_names_out=layers_names_out,
                func_params=func_params,
                overwrite=self._overwrite_checkbox.value
            )
    
        if self._systematic_crop_checkbox.value:
            old_overwrite = self._overwrite_checkbox.value
            self._overwrite_checkbox.value = True
            self._run_crop_array_using_mask()
            self._overwrite_checkbox.value = old_overwrite
 
    def _update_target_axis_choices(self, event):

        if event == 'XY':
            self._align_major_axis_target_axis_combo.choices = ['Y', 'X']
            self._align_major_axis_target_axis_combo.value = 'Y'
        elif event == 'XZ':
            self._align_major_axis_target_axis_combo.choices = ['Z', 'X']
            self._align_major_axis_target_axis_combo.value = 'Z'
        else:
            self._align_major_axis_target_axis_combo.choices = ['Y', 'Z']
            self._align_major_axis_target_axis_combo.value = 'Y'

    def _run_remove_labels_outside_of_mask(self):
            
        mask_layer, _ = self._assert_basic_layer_properties(
            self._mask_layer_combo.value, ['Image']
        )

        labels_layer, _ = self._assert_basic_layer_properties(
            self._labels_layer_combo.value, ['Labels']
        )

        assert mask_layer is not None and labels_layer is not None, 'Please select both mask and labels layers'
        assert mask_layer.data.shape == labels_layer.data.shape, 'Mask and labels must have the same shape'

        layers_names_in = {
            'mask': mask_layer.name,
            'labels': labels_layer.name,
        }

        func_params = {
            'n_jobs': self._n_jobs_slider.value,
        }

        start_time = time.time()
        labels_cropped = remove_labels_outside_of_mask(
            labels_layer.data,
            mask_layer.data,
            **func_params
        )
        print(f'Removing labels took {time.time() - start_time} seconds')

        layers_names_out = OrderedDict()

        if self._overwrite_checkbox.value:
            labels_layer.data = labels_cropped
            layers_names_out['labels'] = labels_layer.name
        else:
            name = f'{labels_layer.name} cropped'

            self._viewer.add_labels(
                labels_cropped,
                name=name,
                **self._transmissive_labels_layer_properties(labels_layer)
            )
            self._labels_layer_combo.native.setCurrentIndex(self._labels_layer_combo.native.count()-1)

            layers_names_out['labels'] = name

        
        if self._is_recording_parameters:
            self._recorder.record(
                function_name='remove_labels_outside_of_mask',
                layers_names_in=layers_names_in,
                layers_names_out=layers_names_out,
                func_params=func_params,
                overwrite=self._overwrite_checkbox.value
            )
    
        if self._systematic_crop_checkbox.value:
            old_overwrite = self._overwrite_checkbox.value
            self._overwrite_checkbox.value = True
            self._run_crop_array_using_mask()
            self._overwrite_checkbox.value = old_overwrite

    def _run_crop_array_using_mask(self):
            
        mask_layer, _ = self._assert_basic_layer_properties(
            self._mask_layer_combo.value, ['Image']
        )

        layers_properties = {
            'mask': (mask_layer, 'Image'),
        }

        for layer_type in ['image', 'labels']:
            layer = getattr(self, f'_{layer_type}_layer_combo').value
            if layer is not None:
                layer, napari_type = self._assert_basic_layer_properties(
                    layer, ['Image', 'Labels']
                )
                layers_properties[layer_type] = (layer, napari_type)

                assert layer.data.shape == mask_layer.data.shape, f'{layer_type} and mask must have the same shape'
            else:
                layers_properties[layer_type] = (None, None)

        arrays = {
            layer_type: (value[0].data if value[0] is not None else None) \
                for layer_type, value in layers_properties.items()
        }

        func_params = {
            'margin': int(self._crop_array_using_mask_margin_checkbox.value),
            'n_jobs': self._n_jobs_slider.value,
        }


        start_time = time.time()
        result_arrays = crop_array_using_mask(
            **arrays, **func_params
        )
        print(f'Cropping took {time.time() - start_time} seconds')

        if isinstance(result_arrays, tuple):
            multiple_results = True
            result_counter = 0
        else:
            multiple_results = False

        layers_names_out = OrderedDict()

        for layer_type, (layer, napari_type) in layers_properties.items():
            if layer is not None:

                if multiple_results:
                    array = result_arrays[result_counter]
                    result_counter += 1
                else:
                    array = result_arrays

                if self._overwrite_checkbox.value:
                    layer.data = array
                    layers_names_out[layer_type] = layer.name
                else:

                    name = f'{layer.name} cropped'
                    layers_names_out[layer_type] = name

                    if napari_type == 'Image':
                        self._viewer.add_image(
                            array,
                            name=name,
                            **self._transmissive_image_layer_properties(layer)
                        )
                        if array.dtype == bool:
                            self._mask_layer_combo.native.setCurrentIndex(self._mask_layer_combo.native.count()-1)
                        else:
                            self._image_layer_combo.native.setCurrentIndex(self._image_layer_combo.native.count()-1)
                    else:
                        self._viewer.add_labels(
                            array,
                            name=name,
                            **self._transmissive_labels_layer_properties(layer)
                        )
                        self._labels_layer_combo.native.setCurrentIndex(self._labels_layer_combo.native.count()-1)

            else:
                layers_names_out[layer_type] = None

        if self._is_recording_parameters:
            layers_names_in = {
                layer_type: (v[0].name if v[0] is not None else None) \
                    for layer_type, v in layers_properties.items()
            }

            self._recorder.record(
                function_name='crop_array_using_mask',
                layers_names_in=layers_names_in,
                layers_names_out=layers_names_out,
                func_params=func_params,
                overwrite=self._overwrite_checkbox.value
            )




    def _run_macro(self):
        parameters_path = self._run_macro_parameters_path.value
        save_path = self._run_macro_save_path.value

        if parameters_path == '.' or not os.path.exists(parameters_path):
            warnings.warn('Please enter a path to the macro parameters')
            return
        if save_path == '.' or not os.path.isdir(save_path):
            warnings.warn('Please enter a path to save the outputs')
            return
        
        with open(parameters_path, 'r') as file:
            parameters_list = json.loads(file.read())

        mask_path = str(self._run_macro_mask_path.value)
        image_path = str(self._run_macro_image_path.value)
        labels_path = str(self._run_macro_labels_path.value)
        # tracks_path = str(self._run_macro_tracks_path.value)

        if mask_path == '.' or not os.path.exists(mask_path):
            mask_input = None
        else: ###! MODIFY THIS
            mask_input = self._read_folder_of_tifs(mask_path)
        
        if image_path == '.' or not os.path.exists(image_path):
            image_input = None
        else:
            image_input = self._read_folder_of_tifs(image_path)
        
        if labels_path == '.' or not os.path.exists(labels_path):
            labels_input = None
        else:
            labels_input = self._read_folder_of_tifs(labels_path)

        # if tracks_path == '.' or not os.path.exists(tracks_path):
        #     tracks_input = None
        # else:
        #     tracks_input = tifffile.imread(tracks_path)

        data_dict = {
            'mask':OrderedDict({'mask':mask_input}),
            'image':OrderedDict({'image':image_input}),
            'labels':OrderedDict({'labels':labels_input}),
            # 'tracks':OrderedDict({'tracks':tracks_input}),
        }

        for params in parameters_list:
            function = self._funcs_dict[params['function']]
            func_params = params['func_params']

            in_dict = params['in']
            out_dict = OrderedDict(params['out'])

            function_inputs = {
                k:data_dict[k][v] for k, v in in_dict.items() if v is not None
            }

            results = function(**function_inputs, **func_params)

            out_dict = OrderedDict({k:v for k,v in out_dict.items() if v is not None})

            if len(out_dict) == 1:
                results_dict = {next(iter(out_dict)):results}
            else:
                # enumerate returns the keys
                results_dict = {
                    layer_type:results[i] for i, layer_type in enumerate(out_dict)
                }

            for layer_type, name in out_dict.items():
                data_dict[layer_type][name] = results_dict[layer_type]

            if self._run_macro_save_all_checkbox.value:
                for layer_type, data in results_dict.items():
                    if data is not None:
                        if self._run_macro_compress_checkbox.value and data.dtype != bool:
                            compress_params = {'compress': ('zlib', 1)}
                        else:
                            compress_params = {}

                        self._create_folder_if_needed(f'{save_path}/{out_dict[layer_type]}')
                        
                        self._save_array_to_tif_folder(
                            f'{save_path}/{out_dict[layer_type]}', 
                            data, compress_params
                        )

        if not self._run_macro_save_all_checkbox.value:
            for _, data_dict_of_that_type in data_dict.items():
                # this is where the OrderedDict is useful: it allows to get
                # the last element of the dict, which is the last output of the
                # chain of functions
                last_name, last_data = next(reversed(data_dict_of_that_type.items()))
                if last_data is not None:
                    if self._run_macro_compress_checkbox.value and data.dtype != bool:
                        compress_params = {'compress': ('zlib', 1)}
                    else:
                        compress_params = {}
                    tifffile.imwrite(
                        f'{save_path}/{last_name}.tif', last_data,
                        **compress_params
                    )

    def _create_folder_if_needed(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    def _save_array_to_tif_folder(self, folder_path, array, compress_params):

        num_frames = array.shape[0]
        num_zeros = int(np.ceil(np.log10(num_frames)))

        for i, array_slice in enumerate(array):
            tifffile.imwrite(
                f'{folder_path}/frame_{i:0{num_zeros}d}.tif', array_slice,
                **compress_params
            )

    def _read_folder_of_tifs(self, folder_path):
        tif_files = glob.glob(f'{folder_path}/*.tif')
        tif_files.sort()
        sample_dtype = tifffile.imread(tif_files[0]).dtype
        return np.array([tifffile.imread(tif_file) for tif_file in tif_files], dtype=sample_dtype)

# if __name__ == '__main__':
#     # create a viewer and add some data
#     viewer = napari.Viewer()

#     # create the widget
#     widget = TapenadeProcessingWidget(viewer)

#     # add the widget to the viewer
#     viewer.window.add_dock_widget(widget)

#     napari.run()