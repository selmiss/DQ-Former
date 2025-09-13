import torch

def check_meta_parameters(model):
    """
    打印出模型中仍然在 meta device 上的参数/缓冲区
    """
    meta_params = []
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor) and param.device.type == "meta":
            meta_params.append(("parameter", name, tuple(param.shape)))
    for name, buffer in model.named_buffers():
        if isinstance(buffer, torch.Tensor) and buffer.device.type == "meta":
            meta_params.append(("buffer", name, tuple(buffer.shape)))

    if meta_params:
        print("⚠️ 以下参数/缓冲区仍然在 meta 上：")
        for kind, name, shape in meta_params:
            print(f" - {kind}: {name}, shape={shape}")
    else:
        print("✅ 模型中没有 meta 参数，全部已 materialize。")

    return meta_params
