import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


# Create one input (ValueInfoProto)
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 2, 2, 2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 2, 2, 2])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 2, 2, 2])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    'MatMul', # node name
    ['A', 'B'], # inputs
    ['Y'] # outputs
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [A, B],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')
onnx.save(model_def, 'matmul.onnx')
print('The model is saved!')