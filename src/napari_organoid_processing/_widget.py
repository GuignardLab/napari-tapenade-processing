from itertools import product
import numpy as np
from magicgui.widgets import Container, create_widget, EmptyWidget, ComboBox
from magicgui import magicgui
from time import time
from os import cpu_count
from organoid.preprocessing.segmentation_postprocessing import remove_labels_outside_of_mask
from organoid.preprocessing.preprocessing import make_array_isotropic, compute_mask_with_histomin, \
    compute_mask_with_otsu, local_image_normalization, align_array_major_axis, crop_array_using_mask, \
    compute_mask_with_snp
from napari.layers import Image, Labels
from fractions import Fraction

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari


    
class OrganoidProcessing(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self._viewer = viewer

        self._overwrite_checkbox = create_widget(
            widget_type="CheckBox", label='Newly computed layers overwrite previous ones', 
            options={'value': True}
        )

        self._n_jobs_slider = create_widget(
            widget_type="IntSlider", label='# parallel jobs', 
            options={'min':1, 'max':cpu_count(), 'value':cpu_count()}
        )

        # Making array isotropic
        self._isotropize_layer_combo = create_widget(
            label='Isotropize layer', 
            annotation="napari.layers.Layer",
        )

        self._isotropize_interp_order_combo = create_widget(
            label='Interpolation order', 
            options={'choices':[0, 1, 3], 'value':1},
        )

        self._isotropize_zoom_factors = create_widget(
            widget_type="LineEdit", label='Zoom factors (ZYX)',
            options={'value':'1,1,1'},
        )

        self._isotropize_run_button = create_widget(
            widget_type="PushButton", label='Run isotropize'
        )

        self._isotropize_run_button.clicked.connect(self._run_isotropize)

        # Computing mask
        self._compute_mask_data_layer_combo = create_widget(
            label='Compute mask from', 
            annotation="napari.layers.Image",
            options={'choices': self._not_bool_layers_filter}
        )


        self._compute_mask_method_combo = create_widget(
            label='Method', 
            options={'choices':['otsu', 'histogram min', 'snp otsu'], 'value':'otsu'},
        )

        self._compute_mask_sigma_blur_slider = create_widget(
            widget_type="IntSlider", label='Sigma blur (~ object size/3)',
            options={'min':1, 'max':20, 'value':10},
        )

        self._compute_mask_ostsu_factor_slider = create_widget(
            widget_type="FloatSlider", label='Otsu factor',
            options={'min':0.3, 'max':3, 'value':1},
        )

        self._compute_mask_run_button = create_widget(
            widget_type="PushButton", label='Run compute mask'
        )

        self._compute_mask_run_button.clicked.connect(self._run_compute_mask)

        # Local normalization
        self._local_normalization_data_layer_combo = create_widget(
            label='Local normalization from', 
            annotation="napari.layers.Image",
            options={'choices': self._not_bool_layers_filter}
        )

        self._local_normalization_mask_layer_combo = create_widget(
            label='Remove values outside of mask',
            annotation="napari.layers.Image",
            options={'nullable': True, 'choices': self._bool_layers_filter}
        )

        self._local_normalization_box_size_slider = create_widget(
            widget_type="IntSlider", label='Box size (~ object size)',
            options={'min':5, 'max':30, 'value':10},
        )

        self._local_normalization_percentiles_slider = create_widget(
            widget_type="FloatRangeSlider", label='Percentiles',
            options={'min':0, 'max':100, 'value':[1, 99]},
        )

        self._local_normalization_run_button = create_widget(
            widget_type="PushButton", label='Run local normalization'
        )

        self._local_normalization_run_button.clicked.connect(self._run_local_normalization)

        # Aligning major axis
        self._align_major_axis_mask_layer_combo = create_widget(
            label='Align major axis from mask', 
            annotation="napari.layers.Image",
            options={'choices': self._bool_layers_filter}
        )

        self._align_major_axis_data_layer_combo = create_widget(
            label='Align major axis of image', 
            annotation="napari.layers.Image",
            options={'nullable': True, 'choices': self._not_bool_layers_filter}
        )

        self._align_major_axis_labels_layer_combo = create_widget(
            label='Align major axis of labels', 
            annotation="napari.layers.Labels",
            options={'nullable': True}
        )

        self._align_major_axis_rotation_plane_combo = create_widget(
            label='Rotation plane', 
            options={'choices':['XY', 'XZ', 'YZ'], 'value':'XY'},
        )

        self._align_major_axis_rotation_plane_combo.changed.connect(self._update_target_axis_choices)

        self._align_major_axis_target_axis_combo = create_widget(
            label='Target axis', 
            options={'choices':['Y', 'X']},
        )

        self._align_major_axis_run_button = create_widget(
            widget_type="PushButton", label='Run align major axis'
        )

        self._align_major_axis_run_button.clicked.connect(self._run_align_major_axis)

        # Removing labels outside of mask
        self._remove_labels_outside_of_mask_mask_layer_combo = create_widget(
            label='Remove labels outside of mask',
            annotation="napari.layers.Image",
            options={'choices': self._bool_layers_filter}
        )

        self._remove_labels_outside_of_mask_labels_layer_combo = create_widget(
            label='Remove labels outside of mask',
            annotation="napari.layers.Labels",
        )

        self._remove_labels_outside_of_mask_run_button = create_widget(
            widget_type="PushButton", label='Run remove labels outside of mask'
        )

        self._remove_labels_outside_of_mask_run_button.clicked.connect(self._run_remove_labels_outside_of_mask)

        # Cropping array using mask
        self._crop_array_using_mask_mask_layer_combo = create_widget(
            label='Crop array using mask', 
            annotation="napari.layers.Image",
            options={'choices': self._bool_layers_filter}
        )

        self._crop_array_using_mask_data_layer_combo = create_widget(
            label='Crop image using mask', 
            annotation="napari.layers.Image",
            options={'nullable': True, 'choices': self._not_bool_layers_filter}
        )

        self._crop_array_using_mask_labels_layer_combo = create_widget(
            label='Crop labels using mask', 
            annotation="napari.layers.Labels",
            options={'nullable': True}
        )

        self._crop_array_using_mask_margin_int_slider = create_widget(
            widget_type="IntSlider", label='Margin', 
            options={'min':0, 'max':20, 'value':0},
        )

        self._crop_array_using_mask_run_button = create_widget(
            widget_type="PushButton", label='Run crop array using mask'
        )

        self._crop_array_using_mask_run_button.clicked.connect(self._run_crop_array_using_mask)


        self_update_layer_combos_button = create_widget(
            widget_type="PushButton", label='Update layer combos'
        )

        # append into/extend the container with your widgets
        self.extend(
            [
                self._overwrite_checkbox,
                self._n_jobs_slider,
                self._isotropize_layer_combo,
                self._isotropize_interp_order_combo,
                self._isotropize_zoom_factors,
                self._isotropize_run_button,
                self._compute_mask_data_layer_combo,
                self._compute_mask_method_combo,
                self._compute_mask_sigma_blur_slider,
                self._compute_mask_ostsu_factor_slider,
                self._compute_mask_run_button,
                self._local_normalization_data_layer_combo,
                self._local_normalization_mask_layer_combo,
                self._local_normalization_box_size_slider,
                self._local_normalization_percentiles_slider,
                self._local_normalization_run_button,
                self._align_major_axis_mask_layer_combo,
                self._align_major_axis_data_layer_combo,
                self._align_major_axis_labels_layer_combo,
                self._align_major_axis_rotation_plane_combo,
                self._align_major_axis_target_axis_combo,
                self._align_major_axis_run_button,
                self._remove_labels_outside_of_mask_mask_layer_combo,
                self._remove_labels_outside_of_mask_labels_layer_combo,
                self._remove_labels_outside_of_mask_run_button,
                self._crop_array_using_mask_mask_layer_combo,
                self._crop_array_using_mask_data_layer_combo,
                self._crop_array_using_mask_labels_layer_combo,
                self._crop_array_using_mask_margin_int_slider,
                self._crop_array_using_mask_run_button,
                # self_update_layer_combos_button,
            ]
        )

        
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

    def _identify_layer_type(self, layer: "napari.layers.Layer"):
        layer_type = layer.__class__.__name__
        if layer_type in ('Image', 'Labels'):
            return layer_type
        else:
            return 'Other'
        
    def _manage_comma_seperated_str_values(self, string: str, name: str, 
                                           assert_positive: bool):

        try:
            values = [float(Fraction(s)) for s in string.replace(' ','').split(',')]
        except ValueError:
            print(f'Invalid {name}')
            return
        
        if assert_positive:
            assert all(v > 0 for v in values), f'{name} must be positive'

        assert len(values) == 3, f'{name} must be 3D'

        return values
    
    def _assert_basic_layer_properties(self, layer: "napari.layers.Layer", allowed_types: list):

        assert layer is not None, 'Please select a layer'
        assert layer.data.ndim in (3, 4), 'The layer must be 3D'
        layer_type = self._identify_layer_type(layer)
        assert layer_type in allowed_types, f'The layer must be part of {allowed_types}'

        return layer, layer_type
    
    def _generic_image_layer_properties(self, layer: "napari.layers.Image"):
        return {
            'contrast_limits': layer.contrast_limits, 'gamma': layer.gamma,
            'colormap': layer.colormap, 'blending': layer.blending,
            'opacity': layer.opacity,
        }
    
    def _generic_labels_layer_properties(self, layer: "napari.layers.Labels"):
        return {
            'color': layer.color, 'blending': layer.blending,
            'opacity': layer.opacity,
        }
    
    

    def _run_isotropize(self):

        layer, layer_type = self._assert_basic_layer_properties(
            self._isotropize_layer_combo.value, ['Image', 'Labels']
        )

        zoom_factors = self._isotropize_zoom_factors.value 

        assert zoom_factors is not None, 'Please enter zoom factors'
        
        zoom_factors = self._manage_comma_seperated_str_values(
            zoom_factors, 'zoom factors', assert_positive=True
        )

        if zoom_factors is None:
            return

        start_time = time()
        array_isotropic = make_array_isotropic(
            layer.data, 
            zoom_factors,
            order=self._isotropize_interp_order_combo.value,
            n_jobs=self._n_jobs_slider.value
        )
        print(f'Isotropization took {time() - start_time} seconds')

        if layer.data.dtype == bool:
            array_isotropic = array_isotropic.astype(bool)

        if self._overwrite_checkbox.value:
            layer.data = array_isotropic

        else:
            if layer_type == 'Image':
                self._viewer.add_image(
                    array_isotropic,
                    name=f'{layer.name} isotropic',
                    **self._generic_image_layer_properties(layer)
                )
            else:
                self._viewer.add_labels(
                    array_isotropic,
                    name=f'{layer.name} isotropic',
                    **self._generic_labels_layer_properties(layer)
                )

    def _run_compute_mask(self):
        
        layer, _ = self._assert_basic_layer_properties(
            self._compute_mask_data_layer_combo.value, ['Image']
        )

        start_time = time()
        
        if self._compute_mask_method_combo.value == 'otsu':
            mask = compute_mask_with_otsu(
                layer.data,
                sigma_blur=self._compute_mask_sigma_blur_slider.value,
                threshold_factor=self._compute_mask_ostsu_factor_slider.value,
                n_jobs=self._n_jobs_slider.value
            )
        elif self._compute_mask_method_combo.value == 'histogram min':
            mask = compute_mask_with_histomin(
                layer.data,
                sigma_blur=self._compute_mask_sigma_blur_slider.value,
                threshold_factor=self._compute_mask_ostsu_factor_slider.value,
                n_jobs=self._n_jobs_slider.value
            )
        elif self._compute_mask_method_combo.value == 'snp otsu':
            mask = compute_mask_with_snp(
                layer.data,
                sigma_blur=self._compute_mask_sigma_blur_slider.value,
                threshold_factor=self._compute_mask_ostsu_factor_slider.value,
                n_jobs=self._n_jobs_slider.value
            )

        print(f'Mask computation took {time() - start_time} seconds')

        self._viewer.add_image(
            mask,
            name=f'{layer.name} mask',
            blending='additive',
            opacity=0.5,
        )

    def _run_local_normalization(self):

        layer, _ = self._assert_basic_layer_properties(
            self._local_normalization_data_layer_combo.value, ['Image']
        )

        if self._local_normalization_mask_layer_combo.value is not None:
            mask_layer, _ = self._assert_basic_layer_properties(
                self._local_normalization_mask_layer_combo.value, ['Image']
            )
            assert mask_layer.data.shape == layer.data.shape, 'Mask and data must have the same shape'
        else:
            mask_layer = None

        perc_low, perc_high = self._local_normalization_percentiles_slider.value

        start_time = time()
        normalized_array = local_image_normalization(
            layer.data,
            perc_low=perc_low, perc_high=perc_high,
            box_size=self._local_normalization_box_size_slider.value,
            n_jobs=self._n_jobs_slider.value
        )
        print(f'Local normalization took {time() - start_time} seconds')

        if mask_layer is not None:
            normalized_array = np.where(mask_layer.data, normalized_array, 0.0)

        if self._overwrite_checkbox.value:
            layer.data = normalized_array
            layer.contrast_limits = (0, 1)
        else:
            self._viewer.add_image(
                normalized_array,
                name=f'{layer.name} normalized',
                colormap=layer.colormap, blending=layer.blending,
                opacity=layer.opacity,
            )

    def _run_align_major_axis(self):

        mask_layer, _ = self._assert_basic_layer_properties(
            self._align_major_axis_mask_layer_combo.value, ['Image']
        )

        if self._align_major_axis_data_layer_combo.value is not None:
            image_layer, _ = self._assert_basic_layer_properties(
                self._align_major_axis_data_layer_combo.value, ['Image']
            )
            assert image_layer.data.shape == mask_layer.data.shape, 'Image and mask must have the same shape'
        else:
            image_layer = None

        if self._align_major_axis_labels_layer_combo.value is not None:
            labels_layer, _ = self._assert_basic_layer_properties(
                self._align_major_axis_labels_layer_combo.value, ['Labels']
            )
            assert labels_layer.data.shape == mask_layer.data.shape, 'Labels and mask must have the same shape'
        else:
            labels_layer = None

        start_time = time()
        aligned_arrays = align_array_major_axis(
            target_axis=self._align_major_axis_target_axis_combo.value,
            rotation_plane=self._align_major_axis_rotation_plane_combo.value,
            mask=mask_layer.data,
            image=image_layer.data if image_layer is not None else None,
            labels=labels_layer.data if labels_layer is not None else None,
            n_jobs=self._n_jobs_slider.value
        )
        print(f'Alignment took {time() - start_time} seconds')

        if image_layer is not None and labels_layer is not None:
            mask_rotated, image_rotated, labels_rotated = aligned_arrays
        elif image_layer is not None:
            mask_rotated, image_rotated = aligned_arrays
        elif labels_layer is not None:
            mask_rotated, labels_rotated = aligned_arrays
        else:
            mask_rotated = aligned_arrays


        if self._overwrite_checkbox.value:
            mask_layer.data = mask_rotated
            if image_layer is not None:
                image_layer.data = image_rotated
            if labels_layer is not None:
                labels_layer.data = labels_rotated
        else:
            self._viewer.add_image(
                mask_rotated,
                name=f'{mask_layer.name} aligned',
                blending='additive',
                opacity=0.5,
            )

            if image_layer is not None:
                self._viewer.add_image(
                    image_rotated,
                    name=f'{image_layer.name} aligned',
                    **self._generic_image_layer_properties(image_layer)
                )

            if labels_layer is not None:
                self._viewer.add_labels(
                    labels_rotated,
                    name=f'{labels_layer.name} aligned',
                    **self._generic_labels_layer_properties(labels_layer)
                )

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
            self._remove_labels_outside_of_mask_mask_layer_combo.value, ['Image']
        )

        labels_layer, _ = self._assert_basic_layer_properties(
            self._remove_labels_outside_of_mask_labels_layer_combo.value, ['Labels']
        )

        assert mask_layer is not None and labels_layer is not None, 'Please select both mask and labels layers'
        assert mask_layer.data.shape == labels_layer.data.shape, 'Mask and labels must have the same shape'

        start_time = time()
        labels_cropped = remove_labels_outside_of_mask(
            labels_layer.data,
            mask_layer.data,
            n_jobs=self._n_jobs_slider.value
        )
        print(f'Removing labels took {time() - start_time} seconds')

        if self._overwrite_checkbox.value:
            labels_layer.data = labels_cropped
        else:
            self._viewer.add_labels(
                labels_cropped,
                name=f'{labels_layer.name} cropped',
                **self._generic_labels_layer_properties(labels_layer)
            )

    def _run_crop_array_using_mask(self):
            
        mask_layer, _ = self._assert_basic_layer_properties(
            self._crop_array_using_mask_mask_layer_combo.value, ['Image']
        )

        if self._crop_array_using_mask_data_layer_combo.value is not None:
            image_layer, _ = self._assert_basic_layer_properties(
                self._crop_array_using_mask_data_layer_combo.value, ['Image']
            )
            assert image_layer.data.shape == mask_layer.data.shape, 'Data and mask must have the same shape'
        else:
            image_layer = None

        if self._crop_array_using_mask_labels_layer_combo.value is not None:
            labels_layer, _ = self._assert_basic_layer_properties(
                self._crop_array_using_mask_labels_layer_combo.value, ['Labels']
            )
            assert labels_layer.data.shape == mask_layer.data.shape, 'Labels and mask must have the same shape'
        else:
            labels_layer = None
        

        assert mask_layer is not None and (image_layer is not None or labels_layer is not None), 'Please select mask and data or labels layers'

        start_time = time()
        cropped_arrays = crop_array_using_mask(
            mask_layer.data,
            image_layer.data if image_layer is not None else None,
            labels_layer.data if labels_layer is not None else None,
            margin=self._crop_array_using_mask_margin_int_slider.value,
            n_jobs=self._n_jobs_slider.value
        )
        print(f'Cropping took {time() - start_time} seconds')

        if image_layer is not None and labels_layer is not None:
            mask_cropped, image_cropped, labels_cropped = cropped_arrays
        elif image_layer is not None:
            mask_cropped, image_cropped = cropped_arrays
        elif labels_layer is not None:
            mask_cropped, labels_cropped = cropped_arrays
        else:
            mask_cropped = cropped_arrays     

        
        if self._overwrite_checkbox.value:
            mask_layer.data = mask_cropped
            if image_layer is not None:
                image_layer.data = image_cropped
            if labels_layer is not None:
                labels_layer.data = labels_cropped
        else:
            self._viewer.add_image(
                mask_cropped,
                name=f'{mask_layer.name} cropped',
                blending='additive',
                opacity=0.5,
            )

            if image_layer is not None:
                self._viewer.add_image(
                    image_cropped,
                    name=f'{image_layer.name} cropped',
                    **self._generic_image_layer_properties(image_layer)
                )

            if labels_layer is not None:
                self._viewer.add_labels(
                    labels_cropped,
                    name=f'{labels_layer.name} cropped',
                    **self._generic_labels_layer_properties(labels_layer)
                )