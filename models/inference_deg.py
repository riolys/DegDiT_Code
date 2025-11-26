import os
import copy
import json
import time
import torch
import argparse
import soundfile as sf
import yaml
import numpy as np
import re
import logging
from tqdm import tqdm
from transformers import AutoProcessor, ClapModel
import torchaudio
from model_deg import TangoFluxDEG
from deg import AudioEventGraph, parse_audio_event_description
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck
from pathlib import Path
torch.manual_seed(42)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation with DEG control.")
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model weights."
    )
    parser.add_argument(
        "--graph_encoder", type=str, default=None,
        help="Path for saved graph encoder weights (optional)."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="JSON file containing the test prompts for generation."
    )
    parser.add_argument(
        "--text_key", type=str, default="time_captions",
        help="Key containing the text in the json file."
    )
    parser.add_argument(
        "--event_key", type=str, default="data_numpy",
        help="Key containing the event description in the json file."
    )
    parser.add_argument(
        "--duration", type=int, default=10,
        help="Duration of generated audio in seconds."
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--num_test_instances", type=int, default=-1,
        help="How many test instances to evaluate.",
    )
    parser.add_argument(
        "--clap_model", type=str, default="laion_clap/630k-audioset-best.pt",
        help="Path to CLAP model for audio-text similarity ranking."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/audiosetS/deg",
        help="Directory to save outputs."
    )
    parser.add_argument(
        "--event_mapping", type=str, default="audioset", choices=["audioset", "pico"],
        help="Which event mapping to use ('audioset' or 'pico')"
    )
    parser.add_argument(
        "--use_text_encoder", action="store_true",
        help="Use text encoder for event encoding instead of event mapping"
    )
    parser.add_argument(
        "--save_graph_visualization", action="store_true",
        help="Save visualization of audio event graph"
    )
    
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class TangoFluxDEGInference:
    
    def __init__(
        self,
        model_path, 
        config=None,
        graph_encoder_path=None,
        event_mapping="audioset",
        use_text_encoder=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.vae = AutoencoderOobleck()
        vae_weights = load_file(f"{model_path}/model.safetensors")
        self.vae.load_state_dict(vae_weights)
        
        if config is None:
            config = {}
        if "use_deg" not in config:
            config["use_deg"] = True
            
        config["use_text_encoder_for_events"] = use_text_encoder
        if use_text_encoder:
            event_type_mapping = None
        else:
            raise ValueError(f"Please set use_text_encoder_for_events as True")

        
        self.model = TangoFluxDEG(config=config, event_type_mapping=event_type_mapping)
        weights = load_file(f"{model_path}/model_1.safetensors")
        self.model.load_state_dict(weights, strict=False)
        
        if graph_encoder_path is not None and os.path.exists(graph_encoder_path):
            logger.info(f"From {graph_encoder_path} load DEG weight")
            
            try:
                graph_weights = load_file(graph_encoder_path)
                
                model_state = self.model.state_dict()
                updated_keys = []
                
                for k, v in graph_weights.items():
                    if k in model_state:
                        if model_state[k].shape == v.shape:
                            model_state[k] = v
                            updated_keys.append(k)
                        else:
                            logger.warning(f"Shape mismatch, skip parameter: {k}, Model shape: {model_state[k].shape}, Weight Shape: {v.shape}")
                
                self.model.load_state_dict(model_state, strict=False)
                logger.info(f"Successfully loaded the graph encoder weights and updated {len(updated_keys)} parameters")
            except Exception as e:
                logger.error(f"Error loading graph encoder weights: {str(e)}")
        
        self.vae.to(device)
        self.model.to(device)
        
        self.vae.eval()
        self.model.eval()

def audio_text_matching(clap_model, clap_processor, waveforms, text, sample_freq=16000, max_len_in_seconds=10, device="cuda"):

    new_freq = 48000
    resampled = []
    
    for wav in waveforms:
        x = torchaudio.functional.resample(torch.tensor(wav, dtype=torch.float).reshape(1, -1), orig_freq=sample_freq, new_freq=new_freq)[0].numpy()
        resampled.append(x[:new_freq*max_len_in_seconds])

    inputs = clap_processor(text=text, audios=resampled, return_tensors="pt", padding=True, sampling_rate=48000)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clap_model(**inputs)

    logits_per_audio = outputs.logits_per_audio
    ranks = torch.argsort(logits_per_audio.flatten(), descending=True).cpu().numpy()
    return ranks

def main():
    args = parse_args()

    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))
    
    config = load_config(train_args["config"])
    
    if torch.cuda.is_available():
        config['model']['text_encoder_name'] = "t5/flan-t5-large"
        
    if "model" not in config:
        config["model"] = {}
    config["model"]["use_deg"] = True
    
    flux_model = TangoFluxDEGInference(
        model_path=args.model, 
        config=config["model"],
        graph_encoder_path=args.graph_encoder,
        event_mapping=args.event_mapping,
        use_text_encoder=args.use_text_encoder
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"use GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("use CPU")
    
    if args.num_samples > 1:
        clap = ClapModel.from_pretrained(args.clap_model).to(device)
        clap.eval()
        clap_processor = AutoProcessor.from_pretrained(args.clap_model)
    
    prefix = train_args.prefix if hasattr(train_args, 'prefix') and train_args.prefix else ""
    

    with open(args.test_file) as f:
        data = []
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Error at line: {line}")
    
    logger.info(f"loaded {len(data)} audio data pieces")
    
    text_prompts = []
    event_descriptions = []
    filepath_list = []
    
    for item in data:
        if args.text_key in item:
            text_prompts.append(prefix + item[args.text_key])
        else:
            raise KeyError(f"Key '{args.text_key}' not found in item: {item}")
        
        if args.event_key in item:
            event_descriptions.append(item[args.event_key])
        elif "data_numpy" in item:
            event_descriptions.append(item["data_numpy"])
        else:
            raise KeyError(f"Key '{args.event_key}' not found in item: {item}")
        
        if "location" in item:
            filepath = item["location"]
            filename = os.path.basename(filepath)
            filepath_list.append(filename)
        else:
            raise KeyError(f"Key 'location' not found in item: {item}")
    
    if args.num_test_instances != -1:
        text_prompts = text_prompts[:args.num_test_instances]
        event_descriptions = event_descriptions[:args.num_test_instances]
        filepath_list = filepath_list[:args.num_test_instances]
    
    num_steps, guidance, batch_size, num_samples, duration = args.num_steps, args.guidance, args.batch_size, args.num_samples, args.duration
    all_outputs = []
    all_graphs = []

    logger.info(f"generating, total {len(text_prompts)} prmopts")
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        events = event_descriptions[k: k+batch_size] if event_descriptions else [None] * len(text)
        

        start_time = time.time()
        
        with torch.no_grad():

            latents = flux_model.model.inference_flow_with_deg(
                prompt=text,
                event_description=events[0], 
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                duration=duration
            )
            
            inference_time = time.time() - start_time
            decode_start = time.time()
            
            wave = flux_model.vae.decode(latents.transpose(1, 2)).sample.cpu()
            
            decode_time = time.time() - decode_start
        
        waveform_end = int(duration * flux_model.vae.config.sampling_rate)
        wave = wave[:, :, :waveform_end]
        all_outputs.append(wave[0])
        
        logger.info(f"batch {k//batch_size + 1}/{(len(text_prompts)-1)//batch_size + 1} finished," 
                   f"inference time {inference_time:.2f} second, decode time: {decode_time:.2f} second")
    

    exp_id = str(int(time.time()))
    method_tag = "text_encoder" if args.use_text_encoder else args.event_mapping
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    if num_samples == 1:
        output_dir = f"{args.output_dir}/{exp_id}_{os.path.basename(args.model)}_{method_tag}_steps_{num_steps}_guidance_{guidance}"
        os.makedirs(output_dir, exist_ok=True)
        
        
        for j, wav in enumerate(all_outputs):
            output_path = os.path.join(output_dir, filepath_list[j])
            sf.write(output_path, np.squeeze(wav[0]), samplerate=44100)
            logger.info(f"saved: {output_path}, shape: {np.shape(wav[0])}")
            
            if event_descriptions and event_descriptions[j]:
                event_file = os.path.join(output_dir, f"{os.path.splitext(filepath_list[j])[0]}_events.txt")
                with open(event_file, 'w') as f:
                    if isinstance(event_descriptions[j], str):
                        f.write(event_descriptions[j])
                    else:
                        try:
                            descriptions = []
                            for i in range(len(event_descriptions[j][0])):
                                event = event_descriptions[j][0][i]
                                start = event_descriptions[j][1][i]
                                end = event_descriptions[j][2][i]
                                descriptions.append(f"{event} from {start} to {end}")
                            f.write(" and ".join(descriptions))
                        except Exception as e:
                            logger.warning(f"Unable to format event description: {str(e)}")
                            f.write(str(event_descriptions[j]))
        
        logger.info(f"Generated {len(all_outputs)} audio files, saved to {output_dir}")
    else:
        for i in range(num_samples):
            output_dir = f"{args.output_dir}/{exp_id}_{os.path.basename(args.model)}_{method_tag}_steps_{num_steps}_guidance_{guidance}/rank_{i+1}"
            os.makedirs(output_dir, exist_ok=True)
        
        groups = list(chunks(all_outputs, num_samples))
        for k in tqdm(range(len(groups))):
            wavs_for_text = groups[k]
            rank = audio_text_matching(clap, clap_processor, wavs_for_text, text_prompts[k], device=device)
            ranked_wavs_for_text = [wavs_for_text[r] for r in rank]
            
            for i, wav in enumerate(ranked_wavs_for_text):
                output_dir = f"{args.output_dir}/{exp_id}_{os.path.basename(args.model)}_{method_tag}_steps_{num_steps}_guidance_{guidance}/rank_{i+1}"
                output_path = os.path.join(output_dir, filepath_list[k])
                sf.write(output_path, wav, samplerate=44100)
                
                if event_descriptions and event_descriptions[k]:
                    event_file = os.path.join(output_dir, f"{os.path.splitext(filepath_list[k])[0]}_events.txt")
                    with open(event_file, 'w') as f:
                        if isinstance(event_descriptions[k], str):
                            f.write(event_descriptions[k])
                        else:
                            try:
                                descriptions = []
                                for i in range(len(event_descriptions[k][0])):
                                    event = event_descriptions[k][0][i]
                                    start = event_descriptions[k][1][i]
                                    end = event_descriptions[k][2][i]
                                    descriptions.append(f"{event} from {start} to {end}")
                                f.write(" and ".join(descriptions))
                            except Exception as e:
                                logger.warning(f"Unable to format event description: {str(e)}")
                                f.write(str(event_descriptions[k]))
        

if __name__ == "__main__":
    logger.info("Start DEG generation")
    main()
    logger.info("DEG generation Finish")
