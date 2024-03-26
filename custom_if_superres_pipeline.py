import torch

import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import numpy as np


from typing import Any, Callable, Dict, List, Optional, Tuple, Union


from diffusers import (IFSuperResolutionPipeline)
from diffusers.utils.torch_utils import randn_tensor
from utils import visualize_images, tensor2pil_list



import lpips

import gp_utils

class CustomSuperResolutionPipeline(IFSuperResolutionPipeline):


    def scheduler_step(self, model_output, timestep, sample, p, counter, generator = None,return_dict = False):
        prev_t = self.scheduler.previous_timestep(timestep)
        predicted_variance = None
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output

        if self.scheduler.config.thresholding:
            pred_original_sample = self.scheduler._threshold_sample(pred_original_sample)
        elif self.scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.scheduler.config.clip_sample_range, self.scheduler.config.clip_sample_range
            )


        #

        if self.base_image is not None and counter < 40:
            #perform 5 steps of gradient descent to minimize the distance between the base image and the predicted image

            with torch.enable_grad():
                pred_original_sample_copy = pred_original_sample.clone().to(dtype=torch.float32).requires_grad_(True)
                optimizer = torch.optim.Adam([pred_original_sample_copy], lr=0.1)
                for i in range(5):
                    optimizer.zero_grad()


                    num_levels = pred_original_sample.shape[0]

                    total_l2_loss = 0
                    total_lpips_loss = 0
                    for j in range(num_levels):
                        pred_downsampled, mask = gp_utils.padded_downsample(pred_original_sample_copy[j], p=p ** j)
                        loss = ((pred_downsampled - self.base_image * mask) ** 2).sum()
                        if self.lpips_loss is not None:
                            # crop center pred_downsampled
                            pred_downsampled = gp_utils.downsample(pred_original_sample_copy[j], p=p ** j)
                            base_cropped = gp_utils.zoom_in_image(self.base_image, p=p ** j, resize_to_original=False)
                            lpips_loss = self.lpips_loss(pred_downsampled.unsqueeze(0),
                                                         base_cropped.unsqueeze(0)).mean()
                            total_lpips_loss += lpips_loss
                        total_l2_loss += loss

                    total_loss = total_l2_loss + total_lpips_loss * 1000



                    total_loss.backward()
                    optimizer.step()
                pred_original_sample = pred_original_sample_copy.detach().to(dtype=torch.float16)
                sample_new = pred_original_sample * alpha_prod_t ** (0.5) + beta_prod_t ** (0.5) * model_output
                #
                model_input = torch.cat([sample_new, self.upscaled], dim=1)
                model_input = (torch.cat([model_input] * 2))
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                model_output_new = self.unet_forward(model_input, timestep, self.prompt_embeds, self.guidance_scale, class_labels=self.class_labels)
                pred_original_sample = (sample_new - beta_prod_t ** (0.5) * model_output_new) / alpha_prod_t ** (0.5)


            print('L2 loss: {:.4f}, LPIPS loss: {:.4f}, Loss: {:.4f}'.format(total_l2_loss, total_lpips_loss, total_loss))
        # #
        # #
        # #
        pred_original_sample = gp_utils.multi_resolution_blending(pred_original_sample, p=p)
        pred_original_sample = gp_utils.render_full_zoom_stack(pred_original_sample, is_noise=False, p=p)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        variance = 0
        if timestep > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            #Noise rendering leads to inferior results in the superres model
            #variance_noise = gp_utils.render_full_zoom_stack(variance_noise, is_noise=True, p=p)


            if self.scheduler.variance_type == "fixed_small_log":
                variance = self._get_variance(timestep, predicted_variance=predicted_variance) * variance_noise
            elif self.scheduler.variance_type == "learned_range":
                variance = self.scheduler._get_variance(timestep, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self.scheduler._get_variance(timestep, predicted_variance=predicted_variance) ** 0.5) * variance_noise


        pred_prev_sample = pred_prev_sample + variance

        return (pred_prev_sample, pred_original_sample, variance)




    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: int = None,
            width: int = None,
            image: Union[Image.Image, np.ndarray, torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            guidance_scale: float = 4.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            noise_level: int = 250,
            clean_caption: bool = True,
            p: int = 1,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to None):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to None):
                The width in pixels of the generated image.
            image (`PIL.Image.Image`, `np.ndarray`, `torch.FloatTensor`):
                The image to be upscaled.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*, defaults to None):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            noise_level (`int`, *optional*, defaults to 250):
                The amount of noise to add to the upscaled image. Must be in the range `[0, 1000)`
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images, and the second element is a list
            of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw)
            or watermarked content, according to the `safety_checker`.
        """
        # 1. Check inputs. Raise error if not correct

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.check_inputs(
            prompt,
            image,
            batch_size,
            noise_level,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters

        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 5. Prepare intermediate images
        num_channels = self.unet.config.in_channels // 2
        intermediate_images = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            num_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare upscaled image and noise level
        image = self.preprocess_image(image, num_images_per_prompt, device)
        upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

        noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
        noise = randn_tensor(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
        upscaled = self.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

        if do_classifier_free_guidance:
            noise_level = torch.cat([noise_level] * 2)

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        self.prompt_embeds = prompt_embeds
        self.guidance_scale = guidance_scale
        self.class_labels = noise_level
        self.upscaled = upscaled
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = torch.cat([intermediate_images, upscaled], dim=1)

                model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input
                model_input = self.scheduler.scale_model_input(model_input, t)



                noise_pred = self.unet_forward(model_input, t, prompt_embeds, guidance_scale, class_labels=noise_level)


                intermediate_images, pred_original_samples, _ = self.scheduler_step(
                    noise_pred, t, intermediate_images, p, counter=i, **extra_step_kwargs, return_dict=False
                )

                visualize_images(tensor2pil_list(pred_original_samples.clamp(-1, 1)), row_size=len(prompt)).save(
                    'step_{}.png'.format(i))

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

        image = intermediate_images

        if output_type == "pil":
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)

            # 11. Apply watermark
            if self.watermarker is not None:
                image = self.watermarker.apply_watermark(image, self.unet.config.sample_size)
        elif output_type == "pt":
            nsfw_detected = None
            watermark_detected = None

            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        else:
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # Offload all models
        self.maybe_free_model_hooks()

        return image

    def set_base_image(self, base_image):
        #base_image: PIL image
        if base_image is None:
            self.base_image = None
            return
        base_image = base_image.resize((self.unet.config.sample_size, self.unet.config.sample_size), Image.BILINEAR)
        self.base_image = (TF.to_tensor(base_image) * 2 - 1).cuda()

    def set_lpips(self):
        self.lpips_loss = lpips.LPIPS(net='vgg').cuda()
    def unet_forward(self, model_input, t, prompt_embeds, guidance_scale, class_labels=None):
        do_classifier_free_guidance = guidance_scale > 1.0
        noise_pred = self.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            class_labels=class_labels,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
            noise_pred, _ = noise_pred.split(model_input.shape[1] // 2, dim=1)

        return noise_pred