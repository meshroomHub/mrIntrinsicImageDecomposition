import os

def loadPipe(type: str = "depth"):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = None #"fp16"
    dtype = torch.float32 #torch.float16
    pipeType = type.lower()

    if pipeType == "depth":
        from marigold import MarigoldDepthPipeline
        print("Loading Marigold depth model...")
        checkpoint_path = os.getenv('MARIGOLD_MODELS_PATH') + "/marigold-depth-v1-1/"
        pipe: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(checkpoint_path, variant=variant, torch_dtype=dtype)
    elif pipeType == "normals":
        from marigold import MarigoldNormalsPipeline
        print("Loading Marigold normals model...")
        checkpoint_path = os.getenv('MARIGOLD_MODELS_PATH') + "/marigold-normals-v1-1/"
        pipe: MarigoldNormalsPipeline = MarigoldNormalsPipeline.from_pretrained(checkpoint_path, variant=variant, torch_dtype=dtype)
    elif pipeType in ["appearance", "lighting"]:
        from marigold import MarigoldIIDPipeline
        print(f"Loading Marigold {pipeType} model...")
        checkpoint_path = os.getenv('MARIGOLD_MODELS_PATH') + f"/marigold-iid-{pipeType}-v1-1/"
        pipe: MarigoldIIDPipeline = MarigoldIIDPipeline.from_pretrained(checkpoint_path, variant=variant, torch_dtype=dtype)
    else:
        return None

    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    # except ImportError:
    #     pass  # run without xformers

    pipe = pipe.to(device)
    return pipe
