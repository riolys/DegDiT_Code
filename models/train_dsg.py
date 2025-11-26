import time
import argparse
import json
import logging
import math
import os
import yaml
import re
from pathlib import Path
import diffusers
import datasets
import numpy as np
import pandas as pd
import transformers
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import SchedulerType, get_scheduler
from model_deg import TangoFluxDEG
from datasets import load_dataset, Audio
from utils import read_wav_file, pad_wav, augment_wav
from diffusers import AutoencoderOobleck
import torchaudio
from deg import AudioEventGraph, parse_audio_event_description
logger = get_logger(__name__)


class Text2AudioDEGDataset(Dataset):
    
    
    def __init__(self, dataset, prefix, text_column, audio_column, duration_column, 
                 num_examples=-1, time_steps=256):
        self.dataset = dataset
        self.prefix = prefix
        self.text_column = text_column
        self.audio_column = audio_column
        self.duration_column = duration_column
        self.time_steps = time_steps
        
        if num_examples > 0 and num_examples < len(dataset):
            self.dataset = self.dataset.select(range(num_examples))
    
    def __len__(self):
        return len(self.dataset)
    
    def get_num_instances(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data_item = self.dataset[index]
        
        
        text_data = data_item.get("time_captions", data_item[self.text_column])
        if self.prefix:
            text_data = self.prefix + text_data
            
        
        audio_path = data_item.get("location", data_item[self.audio_column])
        
        duration = data_item.get(self.duration_column, 10.0)
        
        event_description = data_item.get("data_numpy", None)
        
        return text_data, audio_path, duration, event_description
    
    def collate_fn(self, batch):
        texts, audio_paths, durations, event_descriptions = zip(*batch)
        return list(texts), list(audio_paths), list(durations), list(event_descriptions)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rectified flow for text to audio generation with DEG control."
    )
    parser.add_argument(
        "--datasetname",
        type=str,
        default="audioset",
        help="The name of the dataset.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="How many examples to use for training and validation.",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="filepath",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tangoflux_config.yaml",
        help="Config file defining the model size as well as other hyper parameter.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Add prefix in text prompts.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay to use."
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save model after every how many epochs when checkpointing_steps is set to best.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="Whether to continue training from a model weight",
    )
    parser.add_argument(
        "--load_graph",
        type=str,
        default=None,
        help="Whether to continue training from a graph model weight",
    )
    parser.add_argument(
        "--augment", action="store_true", default=False,
        help="Augment training data.",
    )
    parser.add_argument(
        "--use_deg", action="store_true", default=True,
        help="Use Dynamic Scene Graph (DEG) for event control.",
    )
    parser.add_argument(
        "--deg_max_events", type=int, default=32,
        help="Maximum number of events in DEG.",
    )
    parser.add_argument(
        "--train_graph_encoder_only", action="store_true", default=False,
        help="Only train the graph encoder part of the model.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    def load_config(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    config = load_config(args.config)
    
    if "model" not in config:
        config["model"] = {}
    config["model"]["use_deg"] = args.use_deg
    config["model"]["deg_max_events"] = args.deg_max_events

    learning_rate = float(config["training"]["learning_rate"])
    num_train_epochs = int(config["training"]["num_train_epochs"])
    num_warmup_steps = int(config["training"]["num_warmup_steps"])
    per_device_batch_size = int(config["training"]["per_device_batch_size"])
    gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])

    output_dir = config["paths"]["output_dir"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if output_dir is None or output_dir == "":
            output_dir = "saved/" + str(int(time.time()))

            if not os.path.exists("saved"):
                os.makedirs("saved")

            os.makedirs(output_dir, exist_ok=True)

        elif output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

    accelerator.wait_for_everyone()

    # Get the datasets
    data_files = {}
    if config["paths"]["train_file"] != "":
        data_files["train"] = config["paths"]["train_file"]
    if config["paths"]["val_file"] != "":
        data_files["validation"] = config["paths"]["val_file"]
    if config["paths"]["test_file"] != "":
        data_files["test"] = config["paths"]["test_file"]
    else:
        data_files["test"] = config["paths"]["val_file"]

    extension = "json"
    raw_datasets = load_dataset(extension, data_files=data_files)
    text_column, audio_column = args.text_column, args.audio_column

    if args.datasetname == "audioset":
        event_type_mapping=EVENT_NAME_MAPPING_AUDIOSET
    model = TangoFluxDEG(config=config["model"], event_type_mapping=event_type_mapping)
    vae = AutoencoderOobleck.from_pretrained(
        "stabelaudio/stable-audio-open-1.0", 
        subfolder="vae"
    )

    ## Freeze vae
    for param in vae.parameters():
        param.requires_grad = False
        vae.eval()

    ## Freeze text encoder param
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        model.text_encoder.eval()

    prefix = args.prefix

    with accelerator.main_process_first():
        train_dataset = Text2AudioDEGDataset(
            raw_datasets["train"],
            prefix,
            text_column,
            audio_column,
            "duration",
            args.num_examples,
        )
        eval_dataset = Text2AudioDEGDataset(
            raw_datasets["validation"],
            prefix,
            text_column,
            audio_column,
            "duration",
            args.num_examples,
        )
        test_dataset = Text2AudioDEGDataset(
            raw_datasets["test"],
            prefix,
            text_column,
            audio_column,
            "duration",
            args.num_examples,
        )

        accelerator.print(
            "Num instances in train: {}, validation: {}, test: {}".format(
                train_dataset.get_num_instances(),
                eval_dataset.get_num_instances(),
                test_dataset.get_num_instances(),
            )
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["training"]["per_device_batch_size"],
        collate_fn=train_dataset.collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=config["training"]["per_device_batch_size"],
        collate_fn=eval_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config["training"]["per_device_batch_size"],
        collate_fn=test_dataset.collate_fn,
    )

    # Optimizer
    if args.train_graph_encoder_only and args.use_deg:
        optimizer_parameters = list(model.graph_transformer.parameters())
        for param in model.transformer.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = False
    else:
        trainable_params = []
        if args.use_deg:
            trainable_params.extend(list(model.graph_transformer.parameters()))
        trainable_params.extend(list(model.transformer.parameters()) + list(model.fc.parameters()))
        optimizer_parameters = trainable_params
        
    num_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    accelerator.print(f"Num trainable parameters: {num_trainable_parameters}")

    if args.load_from_checkpoint:
        from safetensors.torch import load_file

        w1 = load_file(args.load_from_checkpoint)
        model.load_state_dict(w1, strict=False)
        logger.info(f"Weights loaded from {args.load_from_checkpoint}")

    if args.load_graph:
        from safetensors.torch import load_file

        w1 = load_file(args.load_graph)
        model.load_state_dict(w1, strict=False)
        logger.info(f"Graph Weights loaded from {args.load_graph}")

    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * gradient_accumulation_steps,
    )

    vae, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, model, optimizer, lr_scheduler
    )

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    total_batch_size = (
        per_device_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Using DEG = {args.use_deg}")
    if args.train_graph_encoder_only:
        logger.info("  Training only graph encoder")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    completed_steps = 0
    starting_epoch = 0
    
    resume_from_checkpoint = config["paths"].get("resume_from_checkpoint", "")
    if resume_from_checkpoint != "":
        accelerator.load_state(resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {resume_from_checkpoint}")

    best_loss = np.inf
    length = config["training"]["max_audio_duration"]
    sample_rate = config["training"]["sample_rate"]

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                device = next(model.parameters()).device
                text, audios, duration, event_descriptions = batch

                with torch.no_grad():
                    audio_list = []

                    for audio_path in audios:
                        wav = read_wav_file(audio_path, length, sample_rate)
                        if (wav.shape[0] == 1):  
                            wav = wav.repeat(2, 1)
                        audio_list.append(wav)

                    audio_input = torch.stack(audio_list, dim=0)
                    audio_input = audio_input.to(device)

                    if args.augment:
                        mixed_audio, mixed_captions, mixed_duration = augment_wav(audio_input, text, sample_rate)
                        mixed_audio = mixed_audio.to(device)
                        audio_input = torch.cat([audio_input, mixed_audio], 0)
                        text += mixed_captions
                        duration += mixed_duration
                        event_descriptions += [None] * len(mixed_captions)

                    unwrapped_vae = accelerator.unwrap_model(vae)
                    duration = torch.tensor(duration, device=device)
                    duration = torch.clamp(duration, max=length)

                    audio_latent = unwrapped_vae.encode(audio_input).latent_dist.sample()
                    audio_latent = audio_latent.transpose(1, 2)  # (bsz, seq_len, channel)

                loss, _, _, _ = model(audio_latent, text, event_description=event_descriptions, duration=duration)
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    progress_bar.update(1)
                    completed_steps += 1

                optimizer.step()
                lr_scheduler.step()

            if completed_steps % 10 == 0 and accelerator.is_main_process:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm**0.5
                logger.info(
                    f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}"
                )

                lr = lr_scheduler.get_last_lr()[0]
                result = {
                    "train_loss": loss.item(),
                    "grad_norm": total_norm,
                    "learning_rate": lr,
                }
                
                if args.use_deg and hasattr(model, "graph_transformer"):
                    try:
                        graph_norm = torch.norm(torch.stack([p.grad.norm() 
                                       for p in model.graph_transformer.parameters() 
                                       if p.grad is not None])).item()
                        logger.info(f"Graph encoder gradient norm: {graph_norm:.4f}")
                    except:
                        pass

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if output_dir is not None:
                        output_dir = os.path.join(output_dir, output_dir)
                    accelerator.save_state(output_dir)

        if completed_steps >= args.max_train_steps:
            break

        model.eval()
        eval_progress_bar = tqdm(
            range(len(eval_dataloader)), disable=not accelerator.is_local_main_process
        )
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                device = next(model.parameters()).device
                text, audios, duration, event_descriptions = batch

                audio_list = []
                for audio_path in audios:
                    wav = read_wav_file(audio_path, length, sample_rate)
                    if (wav.shape[0] == 1):
                        wav = wav.repeat(2, 1)
                    audio_list.append(wav)

                audio_input = torch.stack(audio_list, dim=0)
                audio_input = audio_input.to(device)
                duration = torch.tensor(duration, device=device)
                unwrapped_vae = accelerator.unwrap_model(vae)
                audio_latent = unwrapped_vae.encode(audio_input).latent_dist.sample()
                audio_latent = audio_latent.transpose(1, 2)

                val_loss, _, _, _ = model(audio_latent, text, event_description=event_descriptions, duration=duration)

                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        if accelerator.is_main_process:
            result = {
                "epoch": float(epoch + 1),
                "epoch/train_loss": round(total_loss.item() / len(train_dataloader), 4),
                "epoch/val_loss": round(total_val_loss.item() / len(eval_dataloader), 4)
            }

            result_string = "Epoch: {}, Loss Train: {}, Val: {}\n".format(
                epoch, result["epoch/train_loss"], result["epoch/val_loss"]
            )

            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["epoch/val_loss"] < best_loss:
                best_loss = result["epoch/val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(output_dir, "best"))
                
                if args.use_deg:
                    unwrapped_model = accelerator.unwrap_model(model)
                    graph_params = {
                        k: v for k, v in unwrapped_model.state_dict().items() 
                        if "graph_transformer" in k
                    }
                    from safetensors.torch import save_file
                    graph_encoder_path = f"{output_dir}/graph_encoder_best.safetensors"
                    save_file(graph_params, graph_encoder_path)
                    logger.info(f"Graph encoder parameters saved to {graph_encoder_path}")
                
                try:
                    if args.use_deg and len(event_descriptions) > 0 and event_descriptions[0]:
                        sample_idx = 0
                        sample_text = [text[sample_idx]]
                        sample_event = event_descriptions[sample_idx]
                        sample_duration = [10.0]
                        
                        generated = unwrapped_model.inference_flow_with_deg(
                            prompt=sample_text,
                            event_description=sample_event,
                            num_inference_steps=50,
                            guidance_scale=3.5,
                            duration=sample_duration,
                            disable_progress=True
                        )
                        
                        with torch.no_grad():
                            decoded_audio = unwrapped_vae.decode(generated.transpose(1, 2)).sample
                        
                        sample_path = f"{output_dir}/outputs/epoch_{epoch+1}_best.wav"
                        torchaudio.save(sample_path, decoded_audio.cpu(), sample_rate)
                        logger.info(f"Generated sample audio saved to {sample_path}")
                except Exception as e:
                    logger.info(f"Failed to generate sample: {str(e)}")

            if (epoch + 1) % args.save_every == 0:
                accelerator.save_state(
                    "{}/{}".format(output_dir, "epoch_" + str(epoch + 1))
                )
                
                if args.use_deg:
                    unwrapped_model = accelerator.unwrap_model(model)
                    graph_params = {
                        k: v for k, v in unwrapped_model.state_dict().items() 
                        if "graph_transformer" in k
                    }
                    from safetensors.torch import save_file
                    graph_encoder_path = f"{output_dir}/graph_encoder_epoch_{epoch+1}.safetensors"
                    save_file(graph_params, graph_encoder_path)

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            accelerator.save_state(
                "{}/{}".format(output_dir, "epoch_" + str(epoch + 1))
            )


if __name__ == "__main__":
    main()
