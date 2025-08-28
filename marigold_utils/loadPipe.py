import os

def loadPipe(type: str = "depth"):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = None #"fp16"
    dtype = torch.float32 #torch.float16
    pipeType = type.lower()

    if pipeType == "depth":
        from marigold import MarigoldDepthPipeline
        checkpoint_path = os.getenv('MARIGOLD_MODELS_PATH') + "/marigold-depth-v1-1/"
        pipe: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(checkpoint_path, variant=variant, torch_dtype=dtype).to(device)
    elif pipeType == "depthcompletion":
        from marigold_utils.marigold_dc import MarigoldDepthCompletionPipeline
        from diffusers import DDIMScheduler
        checkpoint_path = os.getenv('MARIGOLD_MODELS_PATH') + "/marigold-depth-v1-1/"
        pipe: MarigoldDepthCompletionPipeline = MarigoldDepthCompletionPipeline.from_pretrained(checkpoint_path, prediction_type="depth").to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif pipeType == "normals":
        from marigold import MarigoldNormalsPipeline
        checkpoint_path = os.getenv('MARIGOLD_MODELS_PATH') + "/marigold-normals-v1-1/"
        pipe: MarigoldNormalsPipeline = MarigoldNormalsPipeline.from_pretrained(checkpoint_path, variant=variant, torch_dtype=dtype).to(device)
    elif pipeType in ["appearance", "lighting"]:
        from marigold import MarigoldIIDPipeline
        checkpoint_path = os.getenv('MARIGOLD_MODELS_PATH') + f"/marigold-iid-{pipeType}-v1-1/"
        pipe: MarigoldIIDPipeline = MarigoldIIDPipeline.from_pretrained(checkpoint_path, variant=variant, torch_dtype=dtype).to(device)
    else:
        return None

    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    # except ImportError:
    #     pass  # run without xformers

    return pipe
