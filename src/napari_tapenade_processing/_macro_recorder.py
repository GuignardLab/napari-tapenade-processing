import os
import json
from datetime import datetime
from collections import OrderedDict


class MacroRecorder:
    def __init__(self, adjective_dict):
        self._is_recording_parameters = False
        self._record_parameters_list = []

        # key: napari name, value: standardized name
        self._record_data_dict = {
            'mask':  {},
            'image': {},
            'labels':{},
            'tracks':{},
        }

        self._adjective_dict = adjective_dict

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