from collections import OrderedDict

class ProcessingGraph:
    def __init__(self, recorded_functions_calls_list):
        self._recorded_functions_calls_list = recorded_functions_calls_list

        self._node_unique_id = -1

        nodes_functions, edges = self._build_graph(recorded_functions_calls_list)
        self._nodes_functions = nodes_functions
        self._edges = edges


    def _build_graph(self, recorded_functions_calls_list):
        # build nodes_functions and edges from recorded_functions_calls_list
        nodes_functions = OrderedDict()
        edges = []

        target_candidates = {}
        source_candidates = {}

        for recorded_function_call in recorded_functions_calls_list:

            self._node_unique_id += 1

            input_params_ids_types = recorded_function_call["input_params_to_layer_ids_and_types_dict"].items()
            output_params_ids_types = recorded_function_call["output_params_to_layer_ids_and_types_dict"].items()

            input_params_to_layer_ids_dict = OrderedDict(
                [(key, value[0]) for key, value in input_params_ids_types]
            )

            output_params_to_layer_ids_dict = OrderedDict(
                [(key, value[0]) for key, value in output_params_ids_types]
            )

            input_params_to_layer_types_dict = OrderedDict(
                [(key, value[1]) for key, value in input_params_ids_types]
            )

            output_params_to_layer_types_dict = OrderedDict(
                [(key, value[1]) for key, value in output_params_ids_types]
            )

            new_node = NodeFunction(
                function_name=recorded_function_call["func_name"],
                func_params=recorded_function_call["func_params"],
                unique_id=self._node_unique_id,
                input_params_to_layer_ids_dict=input_params_to_layer_ids_dict,
                output_params_to_layer_ids_dict=output_params_to_layer_ids_dict,
                input_params_to_layer_types_dict=input_params_to_layer_types_dict,
                output_params_to_layer_types_dict=output_params_to_layer_types_dict,
                main_input_param_name=recorded_function_call["main_input_param_name"]
            )

            nodes_functions[self._node_unique_id] = new_node

            for param, layer_id in input_params_to_layer_ids_dict.items():
                value = (self._node_unique_id, param)
                if layer_id in target_candidates.keys():
                    target_candidates[layer_id].append(value)
                else:
                    target_candidates[layer_id] = [value]

            # for sources, only one layer-id-to-output-port can exist 
            for param, layer_id in output_params_to_layer_ids_dict.items():
                value = (self._node_unique_id, param)
                source_candidates[layer_id] = value
            
        layer_ids_potentially_in_edges = set(target_candidates.keys()).intersection(set(source_candidates.keys()))

        for layer_id in layer_ids_potentially_in_edges:
            source_node_id, source_param = source_candidates[layer_id]

            for target_node_id, target_param in target_candidates[layer_id]:

                new_edge = Edge(
                    layer_id=layer_id,
                    source_node_id_to_param_dict={source_node_id: source_param},
                    target_node_id_to_param_dict={target_node_id: target_param}
                )
                edges.append(new_edge)

        return nodes_functions, edges
            

    @property
    def nodes_functions(self):
        return self._nodes_functions
    
    @property
    def edges(self):
        return self._edges
    
    @property
    def roots_layers_ids(self):
        """
        Roots are extremal target ports.
        """
        source_layer_ids_to_types_dict = {}
        target_layer_ids_to_types_dict = {}

        for node_function in self._nodes_functions.values():

            for param, layer_id in node_function.output_params_to_layer_ids_dict.items():
                layer_type = node_function.output_params_to_layer_types_dict[param]
                if layer_id in source_layer_ids_to_types_dict.keys():
                    source_layer_ids_to_types_dict[layer_id].append(layer_type)
                else:
                    source_layer_ids_to_types_dict[layer_id] = [layer_type]

            for param, layer_id in node_function.input_params_to_layer_ids_dict.items():
                layer_type = node_function.input_params_to_layer_types_dict[param]
                if layer_id in target_layer_ids_to_types_dict.keys():
                    target_layer_ids_to_types_dict[layer_id].append(layer_type)
                else:
                    target_layer_ids_to_types_dict[layer_id] = [layer_type]

        target_layers_ids = list(
            set(target_layer_ids_to_types_dict.keys()).difference(set(source_layer_ids_to_types_dict.keys()))
        )

        target_layer_ids_to_types_dict = {
            layer_id: target_layer_ids_to_types_dict[layer_id] for layer_id in target_layers_ids
        }

        return target_layer_ids_to_types_dict


                                 

    @property
    def leaves_layers_ids(self):
        """
        Leaves are extremal source ports.
        """
        source_layer_ids_to_types_dict = {}
        target_layer_ids_to_types_dict = {}

        for node_function in self._nodes_functions.values():

            for param, layer_id in node_function.output_params_to_layer_ids_dict.items():
                layer_type = node_function.output_params_to_layer_types_dict[param]
                if layer_id in source_layer_ids_to_types_dict.keys():
                    source_layer_ids_to_types_dict[layer_id].append(layer_type)
                else:
                    source_layer_ids_to_types_dict[layer_id] = [layer_type]

            for param, layer_id in node_function.input_params_to_layer_ids_dict.items():
                layer_type = node_function.input_params_to_layer_types_dict[param]
                if layer_id in target_layer_ids_to_types_dict.keys():
                    target_layer_ids_to_types_dict[layer_id].append(layer_type)
                else:
                    target_layer_ids_to_types_dict[layer_id] = [layer_type]

        source_layers_ids = list(
            set(source_layer_ids_to_types_dict.keys()).difference(set(target_layer_ids_to_types_dict.keys()))
        )

        source_layer_ids_to_types_dict = {
            layer_id: source_layer_ids_to_types_dict[layer_id] for layer_id in source_layers_ids
        }

        return source_layer_ids_to_types_dict

class NodeFunction:
    def __init__(self, function_name, 
                 input_params_to_layer_ids_dict: dict,
                 output_params_to_layer_ids_dict: OrderedDict,
                 input_params_to_layer_types_dict: dict,
                 output_params_to_layer_types_dict: OrderedDict,
                 func_params, unique_id, main_input_param_name):
        """
        input/output_params_to_layer_ids_dict maps the name of the input/output
        parameters to the unique id of the layer that  is being used as input/output
        """
        self._function_name = function_name
        self._input_params_to_layer_ids_dict = input_params_to_layer_ids_dict
        self._output_params_to_layer_ids_dict = output_params_to_layer_ids_dict
        self._input_params_to_layer_types_dict = input_params_to_layer_types_dict
        self._output_params_to_layer_types_dict = output_params_to_layer_types_dict
        self._func_params = func_params

        self._unique_id = unique_id
        self._main_input_param_name = main_input_param_name

    @property
    def function_name(self):
        return self._function_name

    @property
    def input_params_to_layer_ids_dict(self):
        return self._input_params_to_layer_ids_dict
    
    @property
    def output_params_to_layer_ids_dict(self):
        return self._output_params_to_layer_ids_dict
    
    @property
    def input_params_to_layer_types_dict(self):
        return self._input_params_to_layer_types_dict
    
    @property
    def output_params_to_layer_types_dict(self):
        return self._output_params_to_layer_types_dict

    @property
    def func_params(self):
        return self._func_params
    
    @property
    def unique_id(self):
        return self._unique_id
    
    @property
    def main_input_param_name(self):
        return self._main_input_param_name
    

class Edge:
    def __init__(self, layer_id, 
                 source_node_id_to_param_dict, target_node_id_to_param_dict):
        """
        source/target_node_id_to_param_dict maps the unique node id of the source/target
        node to the name of the parameter that is being used as input/output
        """
        self._layer_id = layer_id

        self._source_node_id_to_param_dict = source_node_id_to_param_dict
        self._target_node_id_to_param_dict = target_node_id_to_param_dict

    @property
    def layer_id(self):
        return self._layer_id

    @property
    def source_node_id_to_param_dict(self):
        return self._source_node_id_to_param_dict

    @property
    def target_node_id_to_param_dict(self):
        return self._target_node_id_to_param_dict

    
