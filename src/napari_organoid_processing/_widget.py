import tifffile
import numpy as np
from magicgui.widgets import Container, create_widget, EmptyWidget, ComboBox
from qtpy.QtWidgets import QComboBox, QStackedWidget
from collections import OrderedDict
import json
import os
from time import time
from os import cpu_count
from organoid.preprocessing.segmentation_postprocessing import remove_labels_outside_of_mask
from organoid.preprocessing.preprocessing import make_array_isotropic, compute_mask, \
    local_image_normalization, align_array_major_axis, crop_array_using_mask
from napari.layers import Image, Labels
from fractions import Fraction
from datetime import datetime

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari
import napari

"""
! TODO:
- Add manual rotation of principal axis
- Add denoising widget
- Add Napari progress bars https://napari.org/stable/api/napari.utils.progress.html
- Block function calls when a function is already running ? 
- Add tracks processing functions


- Use adjective dict in functions
""" 


class MacroRecorder:
    def __init__(self):
        self._is_recording_parameters = False
        self._record_parameters_list = []

        # key: napari name, value: standardized name
        self._record_data_dict = {
            'mask':  {},
            'image': {},
            'labels':{},
            'tracks':{},
        }

        self._adjective_dict = {
            'make_array_isotropic':          'isotropized',
            'compute_mask':                  'mask',
            'local_image_normalization':     'normalized',
            'align_array_major_axis':        'aligned',
            'remove_labels_outside_of_mask': 'labels_cleared',
            'crop_array_using_mask':         'cropped',
        }

    def _reset_recording(self):
        for k in self._record_data_dict.keys():
            self._record_data_dict[k] = {}
        self._record_parameters_list = []

    def dump_recorded_parameters(self, path: str):
        date = str(datetime.now()).split('.')[:-1][0].replace(' ','_').replace(':', '-')
        filename = f'recorded_parameters_{date}.json'

        with open(os.path.join(path, filename), 'w') as f:
            json.dump(self._record_parameters_list, f)
        
        self._reset_recording()

    def record(self, function_name: str, 
               layers_names_in: dict, layers_names_out: dict, 
               func_params: dict, overwrite: bool):
        """
        layer_names are dicts that map layer_type (e.g 'mask', 'image'...) to 
        the names of the layers in the viewer
        """

        dict_in = dict()
        dict_out = OrderedDict()

        for layer_type in ['mask', 'image', 'labels', 'tracks']:
            # building dict_in
            layer_in = None
            if layer_type in layers_names_in:
                layer_name_in = layers_names_in[layer_type]

                if layer_name_in is not None:
                    layer_in = self._record_data_dict[layer_type].get(layer_name_in, layer_type)

                dict_in[layer_type] = layer_in
            # building dict_out
            if layer_type in layers_names_out:
                layer_name_out = layers_names_out[layer_type]

                if layer_name_out is not None:
                    if overwrite and layer_in is not None:
                        layer_out = layer_in
                    elif layer_in is not None:
                        adjective = self._adjective_dict[function_name]
                        layer_out = f'{layer_in}_{adjective}'
                        self._record_data_dict[layer_type][layer_name_out] = layer_out
                    else: 
                        layer_out = layer_type
                        self._record_data_dict[layer_type][layer_name_out] = layer_out
                else:
                    layer_out = None

                dict_out[layer_type] = layer_out

        params = {
            'function': function_name,
            'in': dict_in,
            'out': dict_out,
            'func_params': func_params,
        }

        self._record_parameters_list.append(params)
        


    
class OrganoidProcessing(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(scrollable=True)

        self._viewer = viewer

        self._funcs_dict = {
            'make_array_isotropic': make_array_isotropic,
            'compute_mask': compute_mask,
            'local_image_normalization': local_image_normalization,
            'align_array_major_axis': align_array_major_axis,
            'remove_labels_outside_of_mask': remove_labels_outside_of_mask,
            'crop_array_using_mask': crop_array_using_mask,
        }

        ### Recording of parameters
        self._record_parameters_path = create_widget(
            widget_type="FileEdit", 
            options={'mode':'d'}
        )

        self._record_parameters_button = create_widget(
            widget_type="PushButton", label='Record macro'
        )

        # self._record_parameters_img = create_widget(
        #     widget_type="Image"
        # )
        # img = np.zeros((20,20,3), dtype='int')
        # img[:,:,1] = 255
        # self._record_parameters_img.set_data(img)
        # print(self._record_parameters_img.image_rgba)

        self._record_parameters_container = Container(
            widgets=[
                self._record_parameters_path,
                self._record_parameters_button,
                # self._record_parameters_img,
            ],
            layout='horizontal',
            label='   Macro save directory',
        )
        
        self._recorder = MacroRecorder()
        self._is_recording_parameters = False

        self._record_parameters_button.clicked.connect(self._manage_recording)
        ###

        self._overwrite_checkbox = create_widget(
            widget_type="CheckBox", label='Newly computed layers overwrite previous ones', 
            options={'value': False}
        )

        self._systematic_crop_checkbox = create_widget(
            widget_type="CheckBox", label='Results are systematically cropped using mask', 
            options={'value': False}
        )

        self._n_jobs_slider = create_widget(
            widget_type="IntSlider", label='# parallel jobs', 
            options={'min':1, 'max':cpu_count(), 'value':cpu_count()}
        )


        self._image_layer_combo = create_widget(
            label='   Image layer',
            annotation="napari.layers.Image",
            options={'nullable': True, 'choices': self._not_bool_layers_filter}
        )

        self._mask_layer_combo = create_widget(
            label='   Mask layer',
            annotation="napari.layers.Image",
            options={'nullable': True, 'choices': self._bool_layers_filter}
        )


        self._labels_layer_combo = create_widget(
            label='   Labels layer',
            annotation="napari.layers.Labels",
            options={'nullable': True}
        )

        self._tracks_layer_combo = create_widget(
            label='   Tracks layer',
            annotation="napari.layers.Tracks",
            options={'nullable': True},
        )


        if True: 
            # Making array isotropic
            self._isotropize_interp_order_combo = create_widget(
                label='Interpolation order\n(for images)', 
                options={'choices':[0, 1, 3], 'value':1},
            )

            self._isotropize_reshape_factors = create_widget(
                widget_type="TupleEdit", label='Reshape factors (ZYX)',
                options={'value':(1.,1.,1.), 'layout':'vertical'},
            )

            self._isotropize_container = Container(
                widgets=[
                    self._isotropize_interp_order_combo,
                    self._isotropize_reshape_factors
                ],
            )

            # Computing mask
            self._compute_mask_method_combo = create_widget(
                label='Method', 
                options={'choices':['otsu', 'histogram min', 'snp otsu'], 'value':'snp otsu'},
            )

            self._compute_mask_sigma_blur_slider = create_widget(
                widget_type="IntSlider", label='Sigma blur (~ object size/3)',
                options={'min':1, 'max':15, 'value':10},
            )

            self._compute_mask_threshold_factor_slider = create_widget(
                widget_type="FloatSlider", label='Threshold mult. factor',
                options={'min':0.5, 'max':1.5, 'value':1},
            )

            self._convex_hull_checkbox = create_widget(
                widget_type="CheckBox", label='Compute convex hull\n(slower)',
                options={'value': False}
            )

            self._compute_mask_container = Container(
                widgets=[
                    self._compute_mask_method_combo,
                    self._compute_mask_sigma_blur_slider,
                    self._compute_mask_threshold_factor_slider,
                    self._convex_hull_checkbox,
                ],
            )

            # Local normalization
            self._local_norm_box_size_slider = create_widget(
                widget_type="IntSlider", label='Box size (~ object size)',
                options={'min':3, 'max':25, 'value':10},
            )

            self._local_norm_percentiles_slider = create_widget(
                widget_type="FloatRangeSlider", label='Percentiles',
                options={'min':0, 'max':100, 'value':[1, 99]},
            )

            self._local_norm_container = Container(
                widgets=[
                    self._local_norm_box_size_slider,
                    self._local_norm_percentiles_slider,
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

            self._run_macro_parameters_path =  create_widget(
                widget_type="FileEdit", 
                options={'mode':'r', 'filter':'*.json'},
                label='Path to macro'
            )

            self._run_macro_mask_path = create_widget(
                widget_type="FileEdit",
                options={'mode':'r', 'filter':'*.tif'},
                label='Path to mask'
            )
                
            self._run_macro_image_path = create_widget(
                widget_type="FileEdit",
                options={'mode':'r', 'filter':'*.tif'},
                label='Path to image'
            )

            self._run_macro_labels_path = create_widget(
                widget_type="FileEdit",
                options={'mode':'r', 'filter':'*.tif'},
                label='Path to labels'
            )

            self._run_macro_tracks_path = create_widget(
                widget_type="FileEdit",
                options={'mode':'r', 'filter':'*.csv'},
                label='Path to tracks'
            )

            self._run_macro_save_path = create_widget(
                widget_type="FileEdit",
                options={'mode':'d'},
                label='Path to save'
            )

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

            self._run_macro_container = Container(
                widgets=[
                    self._run_macro_parameters_path,
                    self._run_macro_mask_path,
                    self._run_macro_image_path,
                    self._run_macro_labels_path,
                    self._run_macro_tracks_path,
                    self._run_macro_save_path,
                    self._run_macro_save_all_checkbox,
                    self._run_macro_compress_checkbox,
                ]
            )



            list_widgets = [
                ('Isotropize layers', self._isotropize_container),
                ('Compute mask from image', self._compute_mask_container),
                ('Local image normalization', self._local_norm_container),
                ('Align layers from mask major axis', self._align_major_axis_container),
                ('Remove labels outside of mask', self._remove_labels_outside_of_mask_container),
                ('Crop layers using mask', self._crop_array_using_mask_container),
                ('Run macro', self._run_macro_container),
            ]

            for _, w in list_widgets:
                if hasattr(w, 'native'):
                    w.native.layout().addStretch()
                else:
                    w.layout().addStretch()

        self._run_button = create_widget(
            widget_type="PushButton", label='Run function'
        )
        self._run_button.clicked.connect(self._run_current_function)

        self._progress_bar = create_widget(
            widget_type="ProgressBar", label='   Progress',
            options={'min':0, 'max':100, 'value':0}
        )

        self._main_combobox = QComboBox()
        self._main_combobox._explicitly_hidden = False
        self._main_combobox.native = self._main_combobox

        main_stack = QStackedWidget()
        main_stack.native = main_stack

        for i, (name,w) in enumerate(list_widgets):
            self._main_combobox.addItem(name)
            if hasattr(w, 'native'):
                main_stack.addWidget(w.native)
            else:
                main_stack.addWidget(w)

        self._main_combobox.currentIndexChanged.connect(main_stack.setCurrentIndex)
        self._main_combobox.currentIndexChanged.connect(self._reset_progress_bar)
        self._main_combobox.currentIndexChanged.connect(self._disable_irrelevant_layers)
        self._main_combobox.name = ""
        main_stack.name = ""

        main_control = Container(
            widgets=[
                self._main_combobox,
                main_stack,
            ],
            labels=False
        )

        choose_function_text = EmptyWidget(label='Choose function:')
        # font = foo.native.font()
        # font.setPointSize(44)
        # font.setBold(True)
        # foo.native.setFont(font)
        # foo.visible=True
        self._record_parameters_text = EmptyWidget(label='Macro recording settings:')




        # append into/extend the container with your widgets
        self.extend(
            [
                EmptyWidget(),
                self._n_jobs_slider,
                # EmptyWidget(),
                self._overwrite_checkbox,
                self._systematic_crop_checkbox,
                EmptyWidget(),
                self._image_layer_combo,
                self._mask_layer_combo,
                self._labels_layer_combo,
                self._tracks_layer_combo,
                EmptyWidget(),
                choose_function_text,
                # self._main_combobox,
                # main_stack,
                main_control,
                # EmptyWidget(),
                self._run_button,
                self._progress_bar,
                EmptyWidget(),
                self._record_parameters_text,
                self._record_parameters_container,
            ]
        )

        
    def _manage_recording(self):
        path = str(self._record_parameters_path.value)

        if path == '.' or not os.path.exists(path):
            print('Please enter a path to record the macro')
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
    
    def _reset_progress_bar(self, event):
        self._progress_bar.value = 0
    
    def _run_current_function(self):
        # do that while waiting for proper parallel progress bars
        self._progress_bar.value = 1

        params = None

        function_index = self._main_combobox.currentIndex()
        if function_index == 0:
            params = self._run_isotropize()
        elif function_index == 1:
            params = self._run_compute_mask()
        elif function_index == 2:
            params = self._run_local_normalization()
        elif function_index == 3:
            params = self._run_align_major_axis()
        elif function_index == 4:
            params = self._run_remove_labels_outside_of_mask()
        elif function_index == 5:
            params = self._run_crop_array_using_mask()
        elif function_index == 6:
            self._run_macro()

        if self._is_recording_parameters and params is not None:
            self._record_parameters_list.append(params)

        # do that while waiting for proper parallel progress bars
        self._progress_bar.value = 100
    
    
    def _disable_irrelevant_layers(self, event):

        if event == 0: # Isotropize
            self._image_layer_combo.enabled = True
            self._mask_layer_combo.enabled = True
            self._labels_layer_combo.enabled = True
            self._tracks_layer_combo.enabled = False #! TODO: add tracks isotropization
        elif event == 1: # Compute mask
            self._image_layer_combo.enabled = True
            self._mask_layer_combo.enabled = False
            self._labels_layer_combo.enabled = False
            self._tracks_layer_combo.enabled = False
        elif event == 2: # Local normalization
            self._image_layer_combo.enabled = True
            self._mask_layer_combo.enabled = True
            self._labels_layer_combo.enabled = False
            self._tracks_layer_combo.enabled = False
        elif event == 3: # Align major axis
            self._image_layer_combo.enabled = True
            self._mask_layer_combo.enabled = True
            self._labels_layer_combo.enabled = True
            self._tracks_layer_combo.enabled = False #! TODO: add tracks rotation
        elif event == 4: # Remove labels outside of mask
            self._image_layer_combo.enabled = False
            self._mask_layer_combo.enabled = True
            self._labels_layer_combo.enabled = True
            self._tracks_layer_combo.enabled = False
        elif event == 5: # Crop array using mask
            self._image_layer_combo.enabled = True
            self._mask_layer_combo.enabled = True
            self._labels_layer_combo.enabled = True
            self._tracks_layer_combo.enabled = False #! TODO: add tracks cropping
        elif event == 6: # Run macro
            self._image_layer_combo.visible = False
            self._mask_layer_combo.visible = False
            self._labels_layer_combo.visible = False
            self._tracks_layer_combo.visible = False
            self._record_parameters_text.visible = False
            self._record_parameters_container.visible = False

        if event != 6:
            self._image_layer_combo.visible = True
            self._mask_layer_combo.visible = True
            self._labels_layer_combo.visible = True
            self._tracks_layer_combo.visible = True
            self._record_parameters_text.visible = True
            self._record_parameters_container.visible = True



    def _identify_layer_type(self, layer: "napari.layers.Layer"):
        layer_type = layer.__class__.__name__
        if layer_type in ('Image', 'Labels'):
            return layer_type
        else:
            return 'Other'
    
    def _assert_basic_layer_properties(self, layer: "napari.layers.Layer", allowed_types: list):

        assert layer is not None, 'Please select a layer'
        assert layer.data.ndim in (3, 4), 'The layer must be 3D (ZYX) or 3D+time (TZYX)'
        layer_type = self._identify_layer_type(layer)
        assert layer_type in allowed_types, f'The layer must be part of {allowed_types}'

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




    def _run_isotropize(self):

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
            print('Please select at least one layer')
            return

        reshape_factors = self._isotropize_reshape_factors.value
        
        assert not(any(factor == 0 for factor in reshape_factors)), 'Reshape factors must be non-zero'

        func_params = {
            'order': self._isotropize_interp_order_combo.value,
            'reshape_factors': reshape_factors,
            'n_jobs': self._n_jobs_slider.value,
        }

        arrays = {
            layer_type: (value[0].data if value[0] is not None else None) \
                for layer_type, value in layers_properties.items()
        }

        start_time = time()
        result_arrays = make_array_isotropic(
            **arrays, **func_params
        )
        print(f'Isotropization took {time() - start_time} seconds')

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

                    name = f'{layer.name} isotropized'
                    layers_names_out[layer_type] = name

                    if napari_type == 'Image':
                        self._viewer.add_image(
                            array,
                            name=name,
                            **self._transmissive_image_layer_properties(layer)
                        )
                        if array.dtype == bool:
                            self._mask_layer_combo.value = self._viewer.layers[-1]
                        else:
                            self._image_layer_combo.value = self._viewer.layers[-1]
                    else:
                        self._viewer.add_labels(
                            array,
                            name=name,
                            **self._transmissive_labels_layer_properties(layer)
                        )
                        self._labels_layer_combo.value = self._viewer.layers[-1]

            else:
                layers_names_out[layer_type] = None
        
        if self._is_recording_parameters:
            layers_names_in = {
                layer_type: (v[0].name if v[0] is not None else None) \
                    for layer_type, v in layers_properties.items()
            }

            self._recorder.record(
                function_name='make_array_isotropic',
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
            'n_jobs': self._n_jobs_slider.value,
        }

        start_time = time()
        mask = compute_mask(
            layer.data, **func_params
        )
        print(f'Mask computation took {time() - start_time} seconds')

        self._viewer.add_image(
            mask,
            name=f'{layer.name} mask',
            blending='additive',
            opacity=0.7,

        )

        self._mask_layer_combo.value = self._viewer.layers[-1]
            
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

    def _run_local_normalization(self):

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

        start_time = time()
        normalized_array = local_image_normalization(
            layer.data,
            mask=mask_layer_data,
            **func_params
        )
        print(f'Local normalization took {time() - start_time} seconds')

        if mask_layer_data is not None:
            normalized_array = np.where(mask_layer_data, normalized_array, 0.0)

        if self._overwrite_checkbox.value:
            layer.data = normalized_array
            layer.contrast_limits = (0, 1)

            layers_names_out = {
                'image': layer.name
            }
        else:
            name = f'{layer.name} normalized'

            self._viewer.add_image(
                normalized_array,
                name=name,
                colormap=layer.colormap, blending=layer.blending,
                opacity=layer.opacity,
            )

            self._image_layer_combo.value = self._viewer.layers[-1]

            layers_names_out = {
                'image': name
            }
        
        if self._is_recording_parameters:
            self._recorder.record(
                function_name='local_image_normalization',
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

        start_time = time()
        result_arrays = align_array_major_axis(
            **arrays, **func_params
        )
        print(f'Alignment took {time() - start_time} seconds')

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
                            self._mask_layer_combo.value = self._viewer.layers[-1]
                        else:
                            self._image_layer_combo.value = self._viewer.layers[-1]
                    else:
                        self._viewer.add_labels(
                            array,
                            name=name,
                            **self._transmissive_labels_layer_properties(layer)
                        )
                        self._labels_layer_combo.value = self._viewer.layers[-1]

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

        start_time = time()
        labels_cropped = remove_labels_outside_of_mask(
            labels_layer.data,
            mask_layer.data,
            **func_params
        )
        print(f'Removing labels took {time() - start_time} seconds')

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
            self._labels_layer_combo.value = self._viewer.layers[-1]

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


        start_time = time()
        result_arrays = crop_array_using_mask(
            **arrays, **func_params
        )
        print(f'Cropping took {time() - start_time} seconds')

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
                            self._mask_layer_combo.value = self._viewer.layers[-1]
                        else:
                            self._image_layer_combo.value = self._viewer.layers[-1]
                    else:
                        self._viewer.add_labels(
                            array,
                            name=name,
                            **self._transmissive_labels_layer_properties(layer)
                        )
                        self._labels_layer_combo.value = self._viewer.layers[-1]

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
            print('Please enter a path to the macro parameters')
            return
        if save_path == '.' or not os.path.isdir(save_path):
            print('Please enter a path to save the outputs')
            return
        
        with open(parameters_path, 'r') as file:
            parameters_list = json.loads(file.read())

        mask_path = str(self._run_macro_mask_path.value)
        image_path = str(self._run_macro_image_path.value)
        labels_path = str(self._run_macro_labels_path.value)
        tracks_path = str(self._run_macro_tracks_path.value)

        if mask_path == '.' or not os.path.exists(mask_path):
            mask_input = None
        else:
            mask_input = tifffile.imread(mask_path)
        
        if image_path == '.' or not os.path.exists(image_path):
            image_input = None
        else:
            image_input = tifffile.imread(image_path)
        
        if labels_path == '.' or not os.path.exists(labels_path):
            labels_input = None
        else:
            labels_input = tifffile.imread(labels_path)

        if tracks_path == '.' or not os.path.exists(tracks_path):
            tracks_input = None
        else:
            tracks_input = tifffile.imread(tracks_path)

        data_dict = {
            'mask':OrderedDict({'mask':mask_input}),
            'image':OrderedDict({'image':image_input}),
            'labels':OrderedDict({'labels':labels_input}),
            'tracks':OrderedDict({'tracks':tracks_input}),
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
                        tifffile.imwrite(
                            f'{save_path}/{out_dict[layer_type]}.tif', data,
                            **compress_params
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



# path_to_data = '/home/jvanaret/data/project_egg/raw/fusion4'
# data = tifffile.imread(f'{path_to_data}/fusion4_smol.tif')

# viewer = napari.Viewer()
# viewer.add_image(data, name='fusion4')
# widget_op = OrganoidProcessing(viewer)
# viewer.window.add_dock_widget(widget_op)

# napari.run()