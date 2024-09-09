import json
import os
from collections import OrderedDict
from datetime import datetime

"""
### TODO:
    - change the macro recorded params structure to have 
      INPUTS ? (i) variable parameters (i.e params that are supposed to
          change when you run the macro) 
      FUNC PARAMS ? (ii) fixed parameters that remain the same across runs
"""


class MacroRecorder:
    def __init__(self):
        self._reset_recording()

    def _reset_recording(self):
        self._recorded_functions_calls_list = []
        self._layer_unique_id = -1
        self._name_to_layer_unique_id_dict = {}

    def dump_recorded_parameters(self, path: str):
        date = (
            str(datetime.now())
            .split(".")[:-1][0]
            .replace(" ", "_")
            .replace(":", "-")
        )
        filename = f"recorded_parameters_{date}.json"

        with open(os.path.join(path, filename), "w") as f:
            json.dump(self._recorded_functions_calls_list, f, indent=4)

        self._reset_recording()

    def record(
        self,
        function_name: str,
        input_params_to_layer_names_and_types_dict: dict,
        output_params_to_layer_names_and_types_dict: OrderedDict,
        func_params: dict,
        main_input_param_name: str
    ):
        """
        Parameters
        ----------
        function_name : str
            The name of the function that is being recorded
        
        func_params : dict
            A dictionary that maps the name of the parameters of the 
            function to their values

        Creates a dictionary with structure:
        
        """

        input_params_to_layer_ids_and_types_dict = {}

        for input_param, (name, layer_type) in input_params_to_layer_names_and_types_dict.items():

            if name is None:
                layer_unique_id = None
            elif name in self._name_to_layer_unique_id_dict.keys():
                    layer_unique_id = self._name_to_layer_unique_id_dict[name]
            else:
                self._layer_unique_id += 1
                self._name_to_layer_unique_id_dict[name] = self._layer_unique_id

                layer_unique_id = self._layer_unique_id

            input_params_to_layer_ids_and_types_dict[input_param] = (layer_unique_id, layer_type)


        output_params_to_layer_ids_and_types_dict = OrderedDict()

        for output_param, (name, layer_type) in output_params_to_layer_names_and_types_dict.items():

            if name is None:
                layer_unique_id = None
            elif name in self._name_to_layer_unique_id_dict.keys():
                layer_unique_id = self._name_to_layer_unique_id_dict[name]
            else:
                self._layer_unique_id += 1
                self._name_to_layer_unique_id_dict[name] = self._layer_unique_id

                layer_unique_id = self._layer_unique_id
            
            output_params_to_layer_ids_and_types_dict[output_param] = (layer_unique_id, layer_type)

        
        params = {
            "func_name": function_name,
            "func_params": func_params,
            "main_input_param_name": main_input_param_name,
            "input_params_to_layer_ids_and_types_dict": input_params_to_layer_ids_and_types_dict,
            "output_params_to_layer_ids_and_types_dict": output_params_to_layer_ids_and_types_dict,
        }

        self._recorded_functions_calls_list.append(params)