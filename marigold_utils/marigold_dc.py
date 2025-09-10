import logging
import os
import warnings
import argparse

import diffusers
import numpy as np
import torch
from diffusers import MarigoldDepthPipeline
from PIL import Image
from img_proc import image
from pathlib import Path
from pyalicevision import image as avimg


warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()

class MarigoldDepthCompletionPipeline(MarigoldDepthPipeline):
    """
    Pipeline for Marigold Depth Completion.
    Extends the MarigoldDepthPipeline to include depth completion functionality.
    """
    def __call__(
        self, image: Image.Image, sparse_depth: np.ndarray,
        num_inference_steps: int = 50, processing_resolution: int = 768, seed: int = 2024
    ) -> np.ndarray:
        
        """
        Args:
            image (PIL.Image.Image): Input image of shape [H, W] with 3 channels.
            sparse_depth (np.ndarray): Sparse depth guidance of shape [H, W].
            num_inference_steps (int, optional): Number of denoising steps. Defaults to 50.
            processing_resolution (int, optional): Resolution for processing. Defaults to 768.
            seed (int, optional): Random seed. Defaults to 2024.

        Returns:
            np.ndarray: Dense depth prediction of shape [H, W].

        """
        # Resolving variables
        device = self._execution_device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Check inputs.
        if num_inference_steps is None:
            raise ValueError("Invalid num_inference_steps")
        if type(sparse_depth) is not np.ndarray or sparse_depth.ndim != 2:
            raise ValueError("Sparse depth should be a 2D numpy ndarray with zeros at missing positions")

        # Prepare empty text conditioning
        with torch.no_grad():
            if self.empty_text_embedding is None:
                text_inputs = self.tokenizer("", padding="do_not_pad", 
                    max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_input_ids = text_inputs.input_ids.to(device)
                self.empty_text_embedding = self.text_encoder(text_input_ids)[0]  # [1,2,1024]

        # Preprocess input images
        image, padding, original_resolution = self.image_processor.preprocess(
            image, processing_resolution=processing_resolution, device=device, dtype=self.dtype
        )  # [N,3,PPH,PPW]

        # Check sparse depth dimensions
        if sparse_depth.shape != original_resolution:
            raise ValueError(
                f"Sparse depth dimensions ({sparse_depth.shape}) must match that of the image ({image.shape[-2:]})"
            )
        
        # Encode input image into latent space
        with torch.no_grad():
            image_latent, pred_latent = self.prepare_latents(image, None, generator, 1, 1)  # [N*E,4,h,w], [N*E,4,h,w]
        del image

        # Preprocess sparse depth
        sparse_depth = torch.from_numpy(sparse_depth)[None, None].float().to(device)
        sparse_mask = sparse_depth > 0
        logging.info(f"Using {sparse_mask.int().sum().item()} guidance points")

        # Set up optimization targets and compute the range and lower bound of the sparse depth
        scale, shift = torch.nn.Parameter(torch.ones(1, device=device)), torch.nn.Parameter(torch.ones(1, device=device))
        pred_latent = torch.nn.Parameter(pred_latent)
        sparse_range = (sparse_depth[sparse_mask].max() - sparse_depth[sparse_mask].min()).item() # (cmax − cmin)
        sparse_lower = (sparse_depth[sparse_mask].min()).item() # cmin
        
        # Set up optimizer
        optimizer = torch.optim.Adam([ {"params": [scale, shift], "lr": 0.005},
                                       {"params": [pred_latent] , "lr": 0.05 }])

        def affine_to_metric(depth: torch.Tensor) -> torch.Tensor:
            # Convert affine invariant depth predictions to metric depth predictions using the parametrized scale and shift. See Equation 2 of the paper.
            return (scale**2) * sparse_range * depth + (shift**2) * sparse_lower

        def latent_to_metric(latent: torch.Tensor) -> torch.Tensor:
            # Decode latent to affine invariant depth predictions and subsequently to metric depth predictions.
            affine_invariant_prediction = self.decode_prediction(latent)  # [E,1,PPH,PPW]
            prediction = affine_to_metric(affine_invariant_prediction)
            prediction = self.image_processor.unpad_image(prediction, padding)  # [E,1,PH,PW]
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, "bilinear", is_aa=False
            )  # [1,1,H,W]
            return prediction

        def loss_l1l2(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # Compute L1 and L2 loss between input and target.
            out_l1 = torch.nn.functional.l1_loss(input, target)
            out_l2 = torch.nn.functional.mse_loss(input, target)
            out = out_l1 + out_l2
            return out

        # Denoising loop
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for _, t in enumerate(
            #self.progress_bar(self.scheduler.timesteps, desc=f"Marigold-DC steps ({str(device)})...")
            self.scheduler.timesteps
        ):
            optimizer.zero_grad()

            # Forward pass through the U-Net
            batch_latent = torch.cat([image_latent, pred_latent], dim=1)  # [1,8,h,w]
            noise = self.unet(
                batch_latent, t, encoder_hidden_states=self.empty_text_embedding, return_dict=False
            )[0]  # [1,4,h,w]

            # Compute pred_epsilon to later rescale the depth latent gradient
            with torch.no_grad():
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_epsilon = (alpha_prod_t**0.5) * noise + (beta_prod_t**0.5) * pred_latent

            step_output = self.scheduler.step(noise, t, pred_latent, generator=generator)

            # Preview the final output depth with Tweedie's formula (See Equation 1 of the paper)
            pred_original_sample = step_output.pred_original_sample

            # Decode to metric space, compute loss with guidance and backpropagate
            current_metric_estimate = latent_to_metric(pred_original_sample)
            loss = loss_l1l2(current_metric_estimate[sparse_mask], sparse_depth[sparse_mask])
            loss.backward()

            # Scale gradients up
            with torch.no_grad():
                pred_epsilon_norm = torch.linalg.norm(pred_epsilon).item()
                depth_latent_grad_norm = torch.linalg.norm(pred_latent.grad).item()
                scaling_factor = pred_epsilon_norm / max(depth_latent_grad_norm, 1e-8)
                pred_latent.grad *= scaling_factor

            # Execute the update step through guidance backprop
            optimizer.step()

            # Execute update of the latent with regular denoising diffusion step
            with torch.no_grad():
                pred_latent.data = self.scheduler.step(noise, t, pred_latent, generator=generator).prev_sample

            del pred_original_sample, current_metric_estimate, step_output, pred_epsilon, noise
            torch.cuda.empty_cache()

        del image_latent

        # Decode predictions from latent into pixel space
        with torch.no_grad():
            prediction = latent_to_metric(pred_latent.detach())

        # return Numpy array
        prediction = self.image_processor.pt_to_numpy(prediction)  # [N,H,W,1]
        self.maybe_free_model_hooks()

        return prediction.squeeze()


def search_partial_depth(depth_dir, depth_ext, img_path, img_meta, logger):
    # Search for an existing partial depth
    input_depth_out = None
    if Path(depth_dir).is_dir():
        input_depth_map_names = []
        input_depth_map_names.append("depth_" + str(img_path[0].stem) + depth_ext)
        input_depth_map_names.append(str(img_path[0].stem) + "_depth" + depth_ext)
        input_depth_map_names.append("depth_" + img_path[1] + depth_ext)
        input_depth_map_names.append(img_path[1] + "_depth" + depth_ext)
        input_depth_map_names.append("depthMap_" + str(img_path[0].stem) + depth_ext)
        input_depth_map_names.append(str(img_path[0].stem) + "_depthMap" + depth_ext)
        input_depth_map_names.append("depthMap_" + img_path[1] + depth_ext)
        input_depth_map_names.append(img_path[1] + "_depthMap" + depth_ext)
        input_depth_map_file_path = None
        for name in input_depth_map_names:
            file_path = os.path.join(depth_dir, name)
            if os.path.exists(file_path):
                input_depth_map_file_path = file_path
                break
        if input_depth_map_file_path is not None:
            # Load partial depth map
            logger.info(f"Found partial depth map {input_depth_map_file_path}")
            if depth_ext == '.npy':
                input_depth = np.load(input_depth_map_file_path)
            else: # '.exr'
                input_depth, h_depth, w_depth, par_depth, orientation_depth = image.loadImage(input_depth_map_file_path, applyPAR = True, clipHigh = 1000.0, colorSpace = avimg.EImageColorSpace_NO_CONVERSION)
                if (h_depth, w_depth, par_depth, orientation_depth) == img_meta:
                    input_depth = input_depth[:,:,0]

            input_depth_out = (input_depth, input_depth_map_file_path)

    return input_depth_out

