import itertools

import onnx

from ..quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizedValue,
    QuantizedValueType,
    attribute_to_kwarg,
    quantize_nparray,
)
from .base_operator import QuantOperatorBase
from .direct_q8 import QDQDirect8BitOp


class QPad(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Pad"

        # Only after version 11, it has the optional constant_value
        # If input[0] is not quantized, do not quanitize this node
        if (self.quantizer.opset_version < 11) or (node.input[0] not in self.quantizer.quantized_value_map):
            super().quantize()
            return
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]

        kwargs = {}
        for attribute in node.attribute:
            kv = attribute_to_kwarg(attribute)
            kwargs.update(kv)

        if "mode" not in kwargs or kwargs["mode"] == b"constant":
            if len(node.input) > 2 and node.input[2] != "":  # There is 3rd input 'constant_value'
                zp_tensor = self.quantizer.model.get_initializer(quantized_input_value.zp_name)
                scale_tensor = self.quantizer.model.get_initializer(quantized_input_value.scale_name)
                if zp_tensor is None or scale_tensor is None:
                    super().quantize()
                    return

                padding_constant_initializer = self.quantizer.model.get_initializer(node.input[2])
                if padding_constant_initializer is not None:
                    zp_array = onnx.numpy_helper.to_array(zp_tensor)
                    zp_value = zp_array.item() if zp_array.ndim == 0 else zp_array[0]
                    scale_array = onnx.numpy_helper.to_array(scale_tensor)
                    scale_value = scale_array.item() if scale_array.ndim == 0 else scale_array[0]
                    padding_constant_array = onnx.numpy_helper.to_array(padding_constant_initializer)
                    quantized_padding_constant_array = quantize_nparray(
                        self.quantizer.activation_qType,
                        padding_constant_array,
                        scale_value,
                        zp_value,
                    )
                    quantized_padding_constant_name = node.input[2] + TENSOR_NAME_QUANT_SUFFIX
                    quantized_padding_constant_initializer = onnx.numpy_helper.from_array(
                        quantized_padding_constant_array,
                        quantized_padding_constant_name,
                    )
                    # Suppose this padding constant initializer only used by the node
                    self.quantizer.model.remove_initializer(padding_constant_initializer)
                    self.quantizer.model.add_initializer(quantized_padding_constant_initializer)
                    node.input[2] = quantized_padding_constant_name
                else:
                    # TODO: check quantize_inputs after sub graph is supported
                    pad_value_qnodes = self.quantizer._get_quantize_input_nodes(
                        node,
                        2,
                        self.quantizer.activation_qType,
                        quantized_input_value.scale_name,
                        quantized_input_value.zp_name,
                        initial_type=scale_tensor.data_type,
                    )
                    self.quantizer.new_nodes.extend(pad_value_qnodes)
                    node.input[2] = pad_value_qnodes[0].output[0]
            else:
                # In quantized format, the `zero` before quantization is mapped
                # to quantized_input_value.zp_name. Thus, padding 0 to
                # original tensor should become padding zero point to quantized
                # tensor.
                if len(node.input) == 2:
                    # Feed quantization's zero point to padding node.
                    node.input.append(quantized_input_value.zp_name)
                else:
                    # Assign quantization's zero point to padding node.
                    assert node.input[2] == ""
                    node.input[2] = quantized_input_value.zp_name

        # Create an entry for output quantized value
        quantized_output_value = QuantizedValue(
            node.output[0],
            node.output[0] + TENSOR_NAME_QUANT_SUFFIX,
            quantized_input_value.scale_name,
            quantized_input_value.zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        node.input[0] = quantized_input_value.q_name
        node.output[0] = quantized_output_value.q_name
        self.quantizer.new_nodes += [node]


class QDQPad(QDQDirect8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def _quantize_output_same_as_input(self):
        return super().quantize()

    def _get_pad_const_val(self, node, attrs_dict):
        const_val = None
        if self.quantizer.opset_version <= 2:
            const_val = attrs_dict.get("value", 0)
        elif len(node.input) >= 3 and node.input[2]:
            const_val = self.quantizer.model.get_constant_value(node.input[2])
        else:
            const_val = 0

        return const_val

    def quantize(self):
        node = self.node
        assert node.op_type == "Pad"

        attrs_dict = {}
        for attribute in node.attribute:
            kv = attribute_to_kwarg(attribute)
            attrs_dict.update(kv)

        pad_mode = attrs_dict.get("mode", b"constant")
        if pad_mode in (b"reflect", b"edge", b"wrap"):
            # These modes pad the output with a value that already exists in the input.
            # So, we can quantize the output the same as the input.
            return self._quantize_output_same_as_input()

        # For 'constant' mode, if padding with 0, we can also quantize the output the same as the input
        # because our quantization floating-point range always includes 0.
        if pad_mode == b"constant":
            const_val = self._get_pad_const_val(node, attrs_dict)
            if const_val == 0:
                return self._quantize_output_same_as_input()

        # Otherwise, just quantize the input and output separately.
        if self.disable_qdq_for_node_output:
            tensors_to_quantize = node.input
        else:
            tensors_to_quantize = itertools.chain(node.input, node.output)

        for tensor_name in tensors_to_quantize:
            self.quantizer.quantize_activation_tensor(tensor_name)
