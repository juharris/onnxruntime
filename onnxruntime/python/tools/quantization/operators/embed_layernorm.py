import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto

'''
Quantizes the EmbedLayerNorm fused ONNXRuntime Op.
 
This Quant operator keeps the input and segment IDs at int32 but will quantize all initializer and
weight inputs associated with the node to uint8.
'''
class EmbedLayerNormalizationQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "EmbedLayerNormalization")

        '''
        Pre-quantization EmbedLayerNorm inputs:
        [0] input_ids (int32)
        [1] segment_ids (int32)
        [2] word_embedding (float32)
        [3] position_embedding (float32)
        [4] segment_embedding (float32)
        [5] layer_norm_weight (float32) 
        [6] layer_norm_bias (float32)
        [7] mask (int32) (optional)
        '''
        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [2, 3, 4, 5, 6])

        # TODO(kreeger): Write a unit test here for this one.
        qembed_layer_norm_name = "" if node.name == "" else node.name + "_quant"

        #
        # TODO(kreeger): Check inputs for gamma/beta/mask with len(node.input) > N
        #

        '''
        Quantized Input Tensor List
        [0] input_ids (int32)
        [1] segment_ids (int32)
        [2] word_embedding (uint8)
        [3] position_embedding (uint8)
        [4] segment_embedding (uint8)
        [5] layer_norm_weight (uint8) 
        [6] layer_norm_bias (uint8)
        [7] word_embedding_scale (float)
        [8] position_embedding_scale (float)
        [9] segment_embedding_scale (float)
        [10] layer_norm_weights_scale (float)
        [11] layer_norm_bias_scale (float)
        [12] word_embedding_zero_point (uint8)
        [13] position_embedding_zero_point (uint8)
        [14] segment_embedding_zero_point (uint8)
        [15] layer_norm_weights_zero_point (uint8)
        [16] layer_norm_bias_zero_point (uint8)
        [17] mask (int32) (optional)
        '''
        inputs = []
        # 'input_ids'
        inputs.extend([node.input[0]])
        # 'segment_ids'
        inputs.extend([node.input[1]])
        # 'word_embedding_quant'
        inputs.extend([quantized_input_names[0]])
        # 'position_embedding_quant'
        inputs.extend([quantized_input_names[1]])
        # 'segment_embedding_quant'
        inputs.extend([quantized_input_names[2]])
        # 'layer_norm_weights_quant'
        inputs.extend([quantized_input_names[3]])
        # 'layer_norm_bias_quant'
        inputs.extend([quantized_input_names[4]])

        # Add all scales:
        inputs.extend([scale_names[0]])
        inputs.extend([scale_names[1]])
        inputs.extend([scale_names[2]])
        inputs.extend([scale_names[3]])
        inputs.extend([scale_names[4]])

        # Add all zero points:
        inputs.extend([zero_point_names[0]])
        inputs.extend([zero_point_names[1]])
        inputs.extend([zero_point_names[2]])
        inputs.extend([zero_point_names[3]])
        inputs.extend([zero_point_names[4]])

        # Put the optional mask arg at the end of the inputs
        if len(node.input) > 7:
            inputs.extend([node.input[7]])

        print(inputs)

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qembed_layer_norm_node = onnx.helper.make_node("QEmbedLayerNormalization", inputs, node.output,
                                                       qembed_layer_norm_name, **kwargs)
        nodes.append(qembed_layer_norm_node)

        self.quantizer.new_nodes += nodes
