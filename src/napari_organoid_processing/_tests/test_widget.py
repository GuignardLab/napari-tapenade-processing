import napari
import tifffile
from os import cpu_count
from qtpy.QtWidgets import QComboBox, QStackedWidget, QVBoxLayout, QWidget
from napari.layers import Image, Labels
from magicgui.widgets import Container, create_widget, EmptyWidget, ComboBox
import numpy as np
from napari_organoid_processing import OrganoidProcessing

path_to_data = '/home/jvanaret/data/project_egg/raw/fusion4'
data = tifffile.imread(f'{path_to_data}/fusion4.tif')



    
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
        
        # self._isotropize_layer_combo.label_changed.connect(self._compute_mask_data_layer_combo_changed)

        self._isotropize_interp_order_combo = create_widget(
            label='Interpolation order', 
            options={'choices':[0, 1, 3], 'value':1},
        )

        self._isotropize_zoom_factors = create_widget(
            widget_type="LineEdit", label='Zoom factors (ZYX)',
            options={'value':'1,1,1'},
        )

        self._isotropize_run_button = create_widget(
            widget_type="PushButton", label='Run isotropize',
        )

        self._isotropize_run_button.clicked.connect(self._run_isotropize)

        # isotropize_container = Container(
        #     widgets=[
        #         self._isotropize_layer_combo,
        #         self._isotropize_interp_order_combo,
        #         self._isotropize_zoom_factors,
        #         self._isotropize_run_button
        #     ]
        # )
        isotropize_container = QWidget()
        isotropize_layout = QVBoxLayout()
        isotropize_layout.setContentsMargins(0.1, 0.1, 0.1, 0.1)  # Set margins to 0
        isotropize_layout.addWidget(self._isotropize_layer_combo.native)
        isotropize_layout.addWidget(self._isotropize_interp_order_combo.native)
        isotropize_layout.addWidget(self._isotropize_zoom_factors.native)
        isotropize_layout.addWidget(self._isotropize_run_button.native)
        isotropize_layout.setSpacing(0)  # Set spacing to 0
        isotropize_container.setLayout(isotropize_layout)
        

        # Computing mask
        self._compute_mask_data_layer_combo = create_widget(
            label='Compute mask from', 
            annotation="napari.layers.Image",
        )

        # self._compute_mask_data_layer_combo.changed.connect(
        #     self._compute_mask_data_layer_combo_changed
        # )

        self._compute_mask_method_combo = create_widget(
            label='Method', 
            options={'choices':['otsu', 'local median'], 'value':'otsu'},
        )

        self._compute_mask_box_size_slider = create_widget(
            widget_type="IntSlider", label='Box size (~ object size)',
            options={'min':5, 'max':30, 'value':10},
        )

        self._compute_mask_run_button = create_widget(
            widget_type="PushButton", label='Run compute mask'
        )

        self._compute_mask_run_button.clicked.connect(self._run_compute_mask)

        # compute_mask_container = Container(
        #     widgets=[
        #         self._compute_mask_data_layer_combo,
        #         self._compute_mask_method_combo,
        #         self._compute_mask_box_size_slider,
        #         self._compute_mask_run_button
        #     ]
        # )
        compute_mask_container = QWidget()
        compute_mask_layout = QVBoxLayout()
        compute_mask_layout.addWidget(self._compute_mask_data_layer_combo.native)
        compute_mask_layout.addWidget(self._compute_mask_method_combo.native)
        compute_mask_layout.addWidget(self._compute_mask_box_size_slider.native)
        compute_mask_layout.addWidget(self._compute_mask_run_button.native)
        compute_mask_container.setLayout(compute_mask_layout)


        main_combobox = QComboBox()
        main_combobox._explicitly_hidden = False
        main_combobox.native = main_combobox

        main_stack = QStackedWidget()
        main_stack.native = main_stack

        for i,w in enumerate([isotropize_container, compute_mask_container]):

            main_combobox.addItem(f'Option {i}')
            main_stack.addWidget(w)

        main_combobox.currentIndexChanged.connect(main_stack.setCurrentIndex)
        main_combobox.name = "main_combobox"
        main_stack.name = "main_stack"
        main_stack.setStyleSheet("QStackedWidget > QWidget { margin: 0px; padding: 0px; border: none; }")

        main_control = Container(
            widgets=[
                main_combobox,
                main_stack,
            ],
            labels=False,
        )


        # append into/extend the container with your widgets
        self.extend(
            [
                self._overwrite_checkbox,
                self._n_jobs_slider,
                main_control
            ]
        )

    def _run_isotropize(self):
        print('run isotropize')

    def _run_compute_mask(self):
        print('run compute mask')

    def _not_bool_layers_filter(self, foo):
        print('filter')
    

viewer = napari.Viewer()
viewer.add_image(data, name='fusion4')
op = OrganoidProcessing(viewer)
viewer.window.add_dock_widget(op)

napari.run()

        
