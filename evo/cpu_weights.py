"""Retain immutable CPU weights while an inference module temporarily uses CUDA."""


class CpuWeights:
    def __init__(self, model):
        self.model = model
        self.tensors = []
        for module in model.modules():
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    if parameter.device.type != "cpu":
                        raise ValueError("Capture CPU weights before moving the model to CUDA.")
                    self.tensors.append((module, name, parameter.detach(), True))
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    if buffer.device.type != "cpu":
                        raise ValueError("Capture CPU buffers before moving the model to CUDA.")
                    self.tensors.append((module, name, buffer.detach(), False))

    def restore(self):
        # Inference never updates weights. Reattach the original storage instead
        # of copying several GB of identical GPU weights back to the CPU.
        # Preserve Parameter objects (and ties); include nonpersistent buffers.
        import torch

        with torch.no_grad():
            for module, name, tensor, parameter in self.tensors:
                if parameter:
                    module._parameters[name].data = tensor
                else:
                    module._buffers[name] = tensor
        return self.model
