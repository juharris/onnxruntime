import io
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.onnx import TrainingMode

from onnxruntime.capi._pybind_state import GradientGraphBuilder

from ...ortmodule._custom_op_symbolic_registry import CustomOpSymbolicRegistry


def export_gradient_graph(
    model: torch.nn.Module,
    loss_fn: Callable[[Any, Any], Any],
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    example_labels: torch.Tensor,
    gradient_graph_path: Union[Path, str],
    opset_version=12,
    dynamic_axes: Optional[Dict[str, Any]] = None,
) -> None:
    r"""
    Build a gradient graph for `model` so that you can output gradients in an inference session when given specific input and corresponding labels.

    Args:
        model (torch.nn.Module): A gradient graph will be built for this model.

        loss_fn (Callable[[Any, Any], Any]): A function to compute the loss given the model's output and the `example_labels`.
        Predefined loss functions such as `torch.nn.CrossEntropyLoss()` will work but you might not be able to load the graph in other environments such as an InferenceSession in ONNX Runtime Web, instead, use a custom Python method.

        example_input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): Example input that you would give your model for inference/prediction.

        example_labels (torch.Tensor): The expected labels for `example_input`.
        This could be the output of your model when given `example_input`,
        but it might be different if your loss function expects labels to be different (e.g. when using cross entropy loss).

        gradient_graph_path (Union[Path, str]): The path to where you would like to save the gradient graph.

        opset_version (int): See `torch.onnx.export`.

        dynamic_axes (Optional[Dict[str, Any]]): See `torch.onnx.export`.
        If not given or `None`, then this will use dynamic axes for "input",  "labels", and "output".
        Set this if your model has inputs or outputs that are not dynamic in size that depends on batches or that depends on something else.
    """

    # Make sure that loss nodes that expect multiple outputs are set up.
    CustomOpSymbolicRegistry.register_all()

    if not isinstance(gradient_graph_path, str):
        gradient_graph_path = str(gradient_graph_path)

    class WrapperModule(torch.nn.Module):
        def forward(self, model_input, expected_labels, *model_params):
            for param, set_param in zip(model.parameters(), model_params):
                param.data = set_param.data

            # Handle when model has multiple parameters.
            # Assume that the model doesn't process a tuple directly, otherwise the caller will have to wrap the tuple in a tuple.
            if isinstance(model_input, tuple):
                output = model(*model_input)
            else:
                output = model(model_input)
            loss = loss_fn(output, expected_labels)
            return output, loss

    wrapped_model = WrapperModule()

    input_names = ["input"]
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {
                0: "batch_size",
            },
            "labels": {
                0: "batch_size",
            },
            "output": {
                0: "batch_size",
            },
        }

        if isinstance(example_input, tuple):
            for i in range(1, len(example_input)):
                name = f"input_{i}"
                dynamic_axes[name] = {0: "batch_size"}
                input_names.append(name)

    args = (example_input, example_labels, *tuple(model.parameters()))
    # TODO Try to avoid create a tuple for `model_param_names`.
    model_param_names = tuple(name for name, _ in model.named_parameters())
    input_names.append("labels")
    input_names.extend(model_param_names)
    nodes_needing_gradients = set(name for name, param in model.named_parameters() if param.requires_grad)

    f = io.BytesIO()
    torch.onnx.export(
        wrapped_model,
        args,
        f,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=False,
        training=TrainingMode.TRAINING,
        input_names=input_names,
        output_names=["output", "loss"],
        dynamic_axes=dynamic_axes,
    )

    exported_model = f.getvalue()
    builder = GradientGraphBuilder(exported_model, {"loss"}, nodes_needing_gradients, "loss")
    builder.build()
    builder.save(gradient_graph_path)
