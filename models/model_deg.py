from transformers import T5EncoderModel, T5TokenizerFast
import torch
from diffusers import FluxTransformer2DModel
from torch import nn
import random
from typing import List, Dict, Tuple, Union, Optional
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import copy
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio
from math import pi
import inspect
import yaml
from deg import AudioEventGraph, parse_audio_event_description, AudioEventGraphTransformer, AudioEventGraphTransformerWithTextEncoder
from model import StableAudioPositionalEmbedding, DurationEmbedder, retrieve_timesteps, TangoFlux


class TangoFluxDEG(TangoFlux):

    def __init__(self, config, event_type_mapping=None, initialize_reference_model=False):
        super().__init__(config, initialize_reference_model=initialize_reference_model)
        
        # DEG configs
        self.use_DEG = config.get("use_DEG", True)
        self.DEG_max_events = config.get("DEG_max_events", 32)
        self.DEG_n_frames = config.get("DEG_n_frames", 16)
        self.use_text_encoder_for_events = config.get("use_text_encoder_for_events", False)
        self.DEG_n_layers = config["DEG_n_layers"]
        print("layers:", self.DEG_n_layers)
        print("frames", self.DEG_n_frames)
        if self.use_text_encoder_for_events:
            print("-----------use_text_encoder_for_events------------")
        self.event_type_mapping = event_type_mapping or {}
        
        if self.use_DEG:
            if self.use_text_encoder_for_events:
                text_encoder_dim = self.text_encoder.config.d_model if hasattr(self.text_encoder.config, 'd_model') else 1024
                
                self.graph_transformer = AudioEventGraphTransformerWithTextEncoder(
                    input_embedding_dim=text_encoder_dim,  
                    hidden_dim=self.text_embedding_dim,
                    n_heads=8,
                    n_layers=self.DEG_n_layers,
                    n_frames=self.DEG_n_frames,
                    pooling_strategy="mean"
                )
            else:
                self.graph_transformer = AudioEventGraphTransformer(
                    event_vocab_size=len(self.event_type_mapping),
                    hidden_dim=self.text_embedding_dim,
                    n_heads=8,
                    n_layers=self.DEG_n_layers,
                    n_frames=self.DEG_n_frames
                )
    
    def encode_event_graph(self, event_description, duration=10.0):

        if not self.use_DEG:
            return None, None
        
        device = next(self.parameters()).device
        graphs = []
        
        if isinstance(event_description, str):
            graph = parse_audio_event_description(event_description, duration)
            graphs = [graph]
        elif isinstance(event_description, AudioEventGraph):
            graphs = [event_description]
        elif isinstance(event_description, list):
            if len(event_description) == 3 and all(isinstance(sublist, list) for sublist in event_description):
                graph = parse_audio_event_description(event_description, duration)
                graphs = [graph]
            elif all(isinstance(item, str) for item in event_description):
                graphs = [parse_audio_event_description(desc, duration) for desc in event_description]
            elif all(isinstance(item, AudioEventGraph) for item in event_description):
                graphs = event_description
            elif all(isinstance(item, list) and len(item) == 3 for item in event_description):
                graphs = [parse_audio_event_description(desc, duration) for desc in event_description]
            else:
                for item in event_description:
                    if isinstance(item, str):
                        graph = parse_audio_event_description(item, duration)
                    elif isinstance(item, AudioEventGraph):
                        graph = item
                    elif isinstance(item, list) and len(item) == 3:
                        graph = parse_audio_event_description(item, duration)
                    else:
                        continue
                    graphs.append(graph)

        if not graphs:
            return None, None

        if self.use_text_encoder_for_events:
            graph_embeddings, graph_mask = self.graph_transformer(
                graphs,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                max_events=self.DEG_max_events
            )
        else:
            graph_embeddings, graph_mask = self.graph_transformer(
                graphs, 
                self.event_type_mapping,
                max_events=self.DEG_max_events
            )
        
        return graph_embeddings, graph_mask
    
    @torch.no_grad()
    def inference_flow_with_DEG(
        self,
        prompt,
        event_description=None,
        num_inference_steps=50,
        timesteps=None,
        guidance_scale=3,
        duration=10,
        disable_progress=False,
        num_samples_per_prompt=1,
    ):

        bsz = num_samples_per_prompt
        device = self.transformer.device
        scheduler = self.noise_scheduler

        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(duration, torch.Tensor):
            duration = torch.tensor([duration], device=device)
            
        classifier_free_guidance = guidance_scale > 1.0

        duration_hidden_states = self.encode_duration(duration)

        if classifier_free_guidance:
            bsz = 2 * num_samples_per_prompt
            encoder_hidden_states, boolean_encoder_mask = self.encode_text_classifier_free(
                prompt, num_samples_per_prompt=num_samples_per_prompt
            )
            duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)
        else:
            encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)

        graph_embeddings = None
        graph_mask = None
        if event_description is not None and self.use_DEG:
            graph_embeddings, graph_mask = self.encode_event_graph(event_description, duration[0].item())
            
            if classifier_free_guidance and graph_embeddings is not None:
                uncond_graph = torch.zeros_like(graph_embeddings[:1]).repeat(num_samples_per_prompt, 1, 1)
                cond_graph = graph_embeddings.repeat(num_samples_per_prompt, 1, 1)
                graph_embeddings = torch.cat([uncond_graph, cond_graph], dim=0)
                uncond_mask = torch.zeros_like(graph_mask[:1], dtype=torch.bool).repeat(num_samples_per_prompt, 1)
                cond_mask = graph_mask.repeat(num_samples_per_prompt, 1)
                graph_mask = torch.cat([uncond_mask, cond_mask], dim=0)
        
        mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
        masked_data = torch.where(
            mask_expanded, encoder_hidden_states, torch.tensor(float("nan"), device=device)
        )
        pooled = torch.nanmean(masked_data, dim=1)
        pooled_projection = self.fc(pooled)
        
        if graph_embeddings is not None and graph_mask is not None:
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, graph_embeddings, duration_hidden_states], dim=1
            )
            duration_mask = torch.ones((boolean_encoder_mask.shape[0], 1), dtype=torch.bool, device=device)
            boolean_encoder_mask = torch.cat([boolean_encoder_mask, graph_mask, duration_mask], dim=1)
        else:
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, duration_hidden_states], dim=1
            )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps, sigmas
        )
        latents = torch.randn(num_samples_per_prompt, self.audio_seq_len, 64)
        progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)
        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
        audio_ids = (
            torch.arange(self.audio_seq_len)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
            .to(device)
        )
        timesteps = timesteps.to(device)
        latents = latents.to(device)
        encoder_hidden_states = encoder_hidden_states.to(device)
        noise_latents = latents
        if classifier_free_guidance:
            noise_latents = torch.cat([noise_latents] * 2)
        for i, t in enumerate(timesteps):
            latents_input = latents
            if classifier_free_guidance:
                latents_input = torch.cat([latents] * 2)
            noise_pred = self.transformer(
                hidden_states=latents_input,
                timestep=torch.tensor([t / 1000], device=device),
                guidance=None,
                pooled_projections=pooled_projection,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=audio_ids,
                return_dict=False,
            )[0]
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            progress_bar.update(1)

        return latents

    def compute_capo_loss(self, audio_latent, text, event_description, duration, calibrated_rewards):
        calibrated_win, calibrated_lose = calibrated_rewards
        batch_size = len(text)
        
        audio_latent_win = audio_latent[:batch_size]
        audio_latent_lose = audio_latent[batch_size:]

        model_output_win = self.transformer(audio_latent_win, text, event_description, duration)
        model_output_lose = self.transformer(audio_latent_lose, text, event_description, duration)

        with torch.no_grad():
            ref_output_win = self.ref_transformer(audio_latent_win, text, event_description, duration)
            ref_output_lose = self.ref_transformer(audio_latent_lose, text, event_description, duration)
        target_preference = calibrated_win - calibrated_lose
        model_preference = model_output_win - model_output_lose
        ref_preference = ref_output_win - ref_output_lose
        preference_diff = model_preference - ref_preference
        loss = (target_preference - self.beta_dpo * preference_diff).pow(2).mean()
        implicit_acc = (torch.sign(target_preference) == torch.sign(preference_diff)).float().mean()
        
        return loss, model_preference.mean().item(), ref_preference.mean().item(), implicit_acc.item()

    def forward(self, latents, prompt, event_description=None, duration=torch.tensor([10]), sft=True, precomputed_preferences=None):
        device = latents.device
        audio_seq_length = self.audio_seq_len
        bsz = latents.shape[0]

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
        duration_hidden_states = self.encode_duration(duration)
        graph_embeddings = None
        graph_mask = None
        if event_description is not None and self.use_DEG:
            graph_embeddings, graph_mask = self.encode_event_graph(event_description, duration[0].item())
        mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
        masked_data = torch.where(
            mask_expanded, encoder_hidden_states, torch.tensor(float("nan"), device=device)
        )
        pooled = torch.nanmean(masked_data, dim=1)
        pooled_projection = self.fc(pooled)
        if graph_embeddings is not None and graph_mask is not None:
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, graph_embeddings, duration_hidden_states], dim=1
            )
            duration_mask = torch.ones((boolean_encoder_mask.shape[0], 1), dtype=torch.bool, device=device)
            boolean_encoder_mask = torch.cat([boolean_encoder_mask, graph_mask, duration_mask], dim=1)
        else:
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, duration_hidden_states], dim=1
            )
        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
        audio_ids = (
            torch.arange(audio_seq_length)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
            .to(device)
        )

        if sft:

            if self.uncondition:
                mask_indices = [k for k in range(len(prompt)) if random.random() < 0.1]
                if len(mask_indices) > 0:
                    encoder_hidden_states[mask_indices] = 0

            noise = torch.randn_like(latents)
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bsz,
                logit_mean=0,
                logit_std=1,
                mode_scale=None,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)

            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

            model_pred = self.transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projection,
                img_ids=audio_ids,
                txt_ids=txt_ids,
                guidance=None,
                timestep=timesteps / 1000,
                return_dict=False,
            )[0]

            target = noise - latents
            loss = torch.mean(
                ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            raw_model_loss, raw_ref_loss, implicit_acc = (0, 0, 0)

        else:
            encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
            pooled_projection = pooled_projection.repeat(2, 1)
            
            noise = (torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1))
            
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bsz // 2,
                logit_mean=0,
                logit_std=1,
                mode_scale=None,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)
            timesteps = timesteps.repeat(2)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)

            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

            model_pred = self.transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projection,
                img_ids=audio_ids,
                txt_ids=txt_ids,
                guidance=None,
                timestep=timesteps / 1000,
                return_dict=False,
            )[0]
            
            target = noise - latents

            model_losses = F.mse_loss(
                model_pred.float(), target.float(), reduction="none"
            )
            model_losses = model_losses.mean(
                dim=list(range(1, len(model_losses.shape)))
            )
            model_losses_w, model_losses_l = model_losses.chunk(2)
            model_diff = model_losses_w - model_losses_l
            raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())

            with torch.no_grad():
                if hasattr(self, 'ref_transformer'):
                    ref_preds = self.ref_transformer(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=pooled_projection,
                        img_ids=audio_ids,
                        txt_ids=txt_ids,
                        guidance=None,
                        timestep=timesteps / 1000,
                        return_dict=False,
                    )[0]

                    ref_loss = F.mse_loss(
                        ref_preds.float(), target.float(), reduction="none"
                    )
                    ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                    ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_loss.mean()
                else:
                    ref_diff = torch.zeros_like(model_diff)
                    raw_ref_loss = 0

            if precomputed_preferences is not None:
                # CaPO
                # precomputed_preferences = win_score - lose_score，
                target_preference = precomputed_preferences.to(device)
                
                model_preference = -(model_losses_w - model_losses_l)
                ref_preference = -(ref_losses_w - ref_losses_l)
                
                preference_diff = model_preference - ref_preference
                capo_loss = (target_preference - self.beta_dpo * preference_diff).pow(2).mean()
                
                loss = capo_loss + 0.1 * model_losses_w.mean()
                
                implicit_acc = (torch.sign(target_preference) == torch.sign(preference_diff)).float().mean()
                
            else:
                # DPO
                scale_term = -0.5 * self.beta_dpo
                inside_term = scale_term * (model_diff - ref_diff)
                implicit_acc = (scale_term * (model_diff - ref_diff) > 0).sum().float() / inside_term.size(0)
                loss = -1 * F.logsigmoid(inside_term).mean() + model_losses_w.mean()

        return loss, raw_model_loss, raw_ref_loss, implicit_acc
