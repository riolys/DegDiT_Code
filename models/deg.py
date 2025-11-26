import torch
import torch.nn as nn
import numpy as np
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
import torch.nn.functional as F

@dataclass
class AudioEvent:
    event_type: str               
    start_time: float             
    end_time: float                
    intensity: float = 1.0        
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def id(self) -> str:
        return f"{self.event_type}_{self.start_time:.2f}_{self.end_time:.2f}"
    
    def to_dict(self) -> Dict:
        return {
            "type": self.event_type,
            "start": self.start_time,
            "end": self.end_time,
            "intensity": self.intensity,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AudioEvent':
        return cls(
            event_type=data["type"],
            start_time=data["start"],
            end_time=data["end"],
            intensity=data.get("intensity", 1.0),
            properties=data.get("properties", {})
        )

@dataclass
class EventRelation:
    source_id: str                
    target_id: str                 
    relation_type: str             
    overlap: float = 0.0           
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation_type,
            "overlap": self.overlap
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EventRelation':
        return cls(
            source_id=data["source"],
            target_id=data["target"],
            relation_type=data["relation"],
            overlap=data.get("overlap", 0.0)
        )


class AudioEventGraph:
    def __init__(self, duration: float = 10.0):
        self.events: List[AudioEvent] = []
        self.relations: List[EventRelation] = []
        self.duration = duration
        self._event_map: Dict[str, int] = {}
        
    def add_event(self, event_type: str, start_time: float, end_time: float, 
                 intensity: float = 1.0, properties: Dict = None) -> str:
        start_time = max(0, min(start_time, self.duration))
        end_time = max(start_time, min(end_time, self.duration))
        
        event = AudioEvent(event_type, start_time, end_time, intensity, properties or {})
        self.events.append(event)
        event_idx = len(self.events) - 1
        self._event_map[event.id] = event_idx
        
        self._update_relations(event)
        
        return event.id
    
    def _update_relations(self, new_event: AudioEvent) -> None:
        if len(self.events) <= 1:
            return
            
        for existing_event in self.events[:-1]:
            if new_event.start_time >= existing_event.end_time:
                relation = "after"
                overlap = 0.0
            elif new_event.end_time <= existing_event.start_time:
                relation = "before"
                overlap = 0.0
            else:
                overlap_start = max(new_event.start_time, existing_event.start_time)
                overlap_end = min(new_event.end_time, existing_event.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if new_event.start_time <= existing_event.start_time and new_event.end_time >= existing_event.end_time:
                    relation = "contains"
                elif existing_event.start_time <= new_event.start_time and existing_event.end_time >= new_event.end_time:
                    relation = "contained_by"
                else:
                    relation = "overlaps"
                
                overlap = overlap_duration / min(new_event.duration, existing_event.duration)
            
            self.relations.append(EventRelation(
                source_id=existing_event.id,
                target_id=new_event.id,
                relation_type=relation,
                overlap=overlap
            ))
            
            inverse_relation = {
                "before": "after",
                "after": "before",
                "contains": "contained_by",
                "contained_by": "contains",
                "overlaps": "overlaps"
            }[relation]
            
            self.relations.append(EventRelation(
                source_id=new_event.id,
                target_id=existing_event.id,
                relation_type=inverse_relation,
                overlap=overlap
            ))
    
    def get_adjacency_tensor(self) -> torch.Tensor:
        n_events = len(self.events)
        if n_events == 0:
            return torch.zeros((0, 0, 5))
        
        adjacency = torch.zeros((n_events, n_events, 5))
        
        relation_map = {
            "before": 0, "after": 1, "overlaps": 2,
            "contains": 3, "contained_by": 4
        }
        
        for relation in self.relations:
            if relation.source_id in self._event_map and relation.target_id in self._event_map:
                source_idx = self._event_map[relation.source_id]
                target_idx = self._event_map[relation.target_id]
                relation_idx = relation_map[relation.relation_type]
                adjacency[source_idx, target_idx, relation_idx] = relation.overlap
        
        return adjacency
    
    def get_temporal_feature_matrix(self, n_frames: int = 16) -> torch.Tensor:
        n_events = len(self.events)
        if n_events == 0:
            return torch.zeros((0, n_frames))
        
        #  [n_events, n_frames]
        temporal_features = torch.zeros((n_events, n_frames))
        frame_duration = self.duration / n_frames
        
        for event_idx, event in enumerate(self.events):
            start_frame = int(event.start_time / frame_duration)
            end_frame = min(n_frames - 1, int(event.end_time / frame_duration))
            
            for frame in range(start_frame, end_frame + 1):
                frame_start = frame * frame_duration
                frame_end = (frame + 1) * frame_duration
                
                overlap_start = max(event.start_time, frame_start)
                overlap_end = min(event.end_time, frame_end)
                overlap = max(0, overlap_end - overlap_start) / frame_duration

                temporal_features[event_idx, frame] = overlap * event.intensity
        
        return temporal_features
    
    def get_dynamic_scene_graph(self, num_frames: int = 16) -> List[List[Dict]]:
        frame_duration = self.duration / num_frames
        scene_graphs = [[] for _ in range(num_frames)]
        
        for event in self.events:
            start_frame = max(0, min(num_frames-1, int(event.start_time / frame_duration)))
            end_frame = max(0, min(num_frames-1, int(event.end_time / frame_duration)))
            
            for frame in range(start_frame, end_frame + 1):
                frame_start = frame * frame_duration
                frame_end = (frame + 1) * frame_duration
                overlap_start = max(event.start_time, frame_start)
                overlap_end = min(event.end_time, frame_end)
                overlap = max(0, overlap_end - overlap_start) / frame_duration
                
                if overlap > 0:
                    scene_graphs[frame].append({
                        "type": event.event_type,
                        "active": True,
                        "overlap": overlap,
                        "intensity": event.intensity,
                        "properties": event.properties
                    })
        
        return scene_graphs
    
    def to_dict(self) -> Dict:
        return {
            "duration": self.duration,
            "events": [event.to_dict() for event in self.events],
            "relations": [relation.to_dict() for relation in self.relations]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AudioEventGraph':
        graph = cls(duration=data.get("duration", 10.0))
        
        id_map = {}
        for event_data in data.get("events", []):
            event_id = graph.add_event(
                event_type=event_data["type"],
                start_time=event_data["start"],
                end_time=event_data["end"],
                intensity=event_data.get("intensity", 1.0),
                properties=event_data.get("properties", {})
            )
            if "id" in event_data:
                id_map[event_data["id"]] = event_id
        
        return graph
    
    def save(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'AudioEventGraph':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def parse_audio_event_description(description, duration: float = 10.0) -> AudioEventGraph:

    graph = AudioEventGraph(duration=duration)
    
    if not description:
        return graph
    

    if isinstance(description, list) and len(description) == 3:
        event_types = description[0]
        start_times = description[1]
        end_times = description[2]
        

        if len(event_types) == len(start_times) == len(end_times):
            for i in range(len(event_types)):
                try:
                    event_type = event_types[i]
                    start_time = float(start_times[i])
                    end_time = float(end_times[i])
                    
                    graph.add_event(
                        event_type=event_type,
                        start_time=start_time,
                        end_time=end_time
                    )
                except (ValueError, TypeError) as e:
                    print(f"event error: {e}")
                    continue
        return graph
    

    if isinstance(description, str):

        event_sections = description.split(" and ")
        
        for section in event_sections:
            event_match = re.match(r"(.*?) (?:at|from) (.*)", section)
            if not event_match:
                continue
                
            event_type = event_match.group(1).strip()
            time_ranges_str = event_match.group(2).strip()
            

            time_ranges = re.findall(r"(\d+\.\d+)-(\d+\.\d+)", time_ranges_str)
            if not time_ranges:
                time_ranges = re.findall(r"<?([\d\.]+)>? to <?([\d\.]+)>?", time_ranges_str)
            
            for start_time, end_time in time_ranges:
                graph.add_event(
                    event_type=event_type,
                    start_time=float(start_time),
                    end_time=float(end_time)
                )
    
    return graph

class AudioEventGraphTransformer(nn.Module):
    
    def __init__(self, 
                 event_vocab_size: int, 
                 hidden_dim: int = 1024,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 n_frames: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_frames = n_frames
        
        self.event_embedding = nn.Embedding(event_vocab_size, hidden_dim//2)
        
        self.time_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim//4),  # [start, end]
            nn.SiLU(),
            nn.LayerNorm(hidden_dim//4),
            nn.Linear(hidden_dim//4, hidden_dim//2)
        )
        
        self.graph_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim//4),  # 5 relations
            nn.SiLU(),
            nn.LayerNorm(hidden_dim//4),
            nn.Linear(hidden_dim//4, hidden_dim//2)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(n_frames, hidden_dim//2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim//2)
        )
        
        self.register_buffer(
            "position_ids", torch.arange(128).expand((1, -1))
        )
        self.position_embeddings = nn.Embedding(128, hidden_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        self.output_mapping = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, graphs: List[AudioEventGraph], event_type_map: Dict[str, int], max_events: int = 32):

        batch_size = len(graphs)
        device = next(self.parameters()).device
        
        event_embeddings = torch.zeros(batch_size, max_events, self.hidden_dim, device=device)
        event_mask = torch.zeros(batch_size, max_events, dtype=torch.bool, device=device)
        
        for b, graph in enumerate(graphs):
            if not graph.events:
                continue
                
            num_events = min(len(graph.events), max_events)
            
            adjacency = graph.get_adjacency_tensor()[:num_events, :num_events].to(device)
            
            temporal_features = graph.get_temporal_feature_matrix(self.n_frames)[:num_events].to(device)
            
            event_indices = []
            for event in graph.events[:num_events]:
                # event_idx = event_type_map.get(event.event_type, 0)
                
                if event.event_type not in event_type_map:
                    raise ValueError(f"unknown event type: '{event.event_type}'")
                
                event_idx = event_type_map[event.event_type]
                event_indices.append(event_idx)
            
            event_indices = torch.tensor(event_indices, device=device)
            type_embeddings = self.event_embedding(event_indices)
            
            time_features = torch.tensor(
                [[event.start_time, event.end_time] for event in graph.events[:num_events]],
                dtype=torch.float32,
                device=device
            )
            time_embeddings = self.time_encoder(time_features)
            
            graph_embeddings = torch.zeros(num_events, self.hidden_dim//2, device=device)
            for i in range(num_events):
                relation_features = adjacency[i, :num_events, :].float()  # [num_events, 5]
                encoded_relations = self.graph_encoder(relation_features)  # [num_events, hidden_dim//2]
                graph_embeddings[i] = encoded_relations.mean(dim=0)
            
            temporal_embeddings = self.temporal_encoder(temporal_features)
            
            combined_embeddings = torch.cat([
                type_embeddings + time_embeddings, 
                graph_embeddings + temporal_embeddings
            ], dim=-1)
            combined_embeddings = self.fusion_layer(combined_embeddings)
            
            position_embeddings = self.position_embeddings(self.position_ids[:, :num_events])
            combined_embeddings = combined_embeddings + position_embeddings
            
            event_embeddings[b, :num_events] = combined_embeddings
            event_mask[b, :num_events] = True
            
        attention_mask = ~event_mask
        encoded_embeddings = self.transformer(event_embeddings, src_key_padding_mask=attention_mask)
        
        output = self.output_mapping(encoded_embeddings)
        
        return output, event_mask


class AudioEventGraphTransformerWithTextEncoder(nn.Module):
    
    def __init__(self, 
                 input_embedding_dim: int = 1024,  
                 hidden_dim: int = 1024,          
                 n_heads: int = 8,               
                 n_layers: int = 4,               
                 n_frames: int = 16,              
                 max_text_length: int = 32,       
                 pooling_strategy: str = "mean"): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_frames = n_frames
        self.max_text_length = max_text_length
        self.pooling_strategy = pooling_strategy
        
        self.text_projector = nn.Sequential(
            nn.Linear(input_embedding_dim, hidden_dim//2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim//2)
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim//4),  # [start, end]
            nn.SiLU(),
            nn.LayerNorm(hidden_dim//4),
            nn.Linear(hidden_dim//4, hidden_dim//2)
        )
        
        self.graph_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim//4),  
            nn.SiLU(),
            nn.LayerNorm(hidden_dim//4),
            nn.Linear(hidden_dim//4, hidden_dim//2)
        )
        
        # 时序编码
        self.temporal_encoder = nn.Sequential(
            nn.Linear(n_frames, hidden_dim//2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim//2)
        )
        
        self.register_buffer(
            "position_ids", torch.arange(128).expand((1, -1))
        )
        self.position_embeddings = nn.Embedding(128, hidden_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        self.output_mapping = nn.Linear(hidden_dim, hidden_dim)
    
    def encode_event_text(self, event_texts: List[str], text_encoder, tokenizer, device):

        inputs = tokenizer(
            event_texts,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            attention_mask = inputs.attention_mask
            sequence_output = outputs.last_hidden_state
            

            if self.pooling_strategy == "cls" and hasattr(outputs, "pooler_output"):
                text_embeddings = outputs.pooler_output
            elif self.pooling_strategy == "mean":
                masked_output = sequence_output * attention_mask.unsqueeze(-1)
                sum_embeddings = masked_output.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
                text_embeddings = sum_embeddings / sum_mask
            elif self.pooling_strategy == "max":
                masked_output = sequence_output.clone()
                masked_output[~attention_mask.bool()] = -1e9
                text_embeddings = torch.max(masked_output, dim=1)[0]
            elif self.pooling_strategy == "last":
                batch_size = sequence_output.shape[0]
                seq_lengths = attention_mask.sum(dim=1) - 1
                text_embeddings = sequence_output[torch.arange(batch_size), seq_lengths]
            else:
                masked_output = sequence_output * attention_mask.unsqueeze(-1)
                text_embeddings = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        projected_embeddings = self.text_projector(text_embeddings)
        return projected_embeddings
        
    def forward(self, graphs: List[AudioEventGraph], text_encoder, tokenizer, max_events: int = 32):

        batch_size = len(graphs)
        device = next(self.parameters()).device
        
        event_embeddings = torch.zeros(batch_size, max_events, self.hidden_dim, device=device)
        event_mask = torch.zeros(batch_size, max_events, dtype=torch.bool, device=device)
        
        for b, graph in enumerate(graphs):
            if not graph.events:
                continue
                
            num_events = min(len(graph.events), max_events)
            
            adjacency = graph.get_adjacency_tensor()[:num_events, :num_events].to(device)
            
            temporal_features = graph.get_temporal_feature_matrix(self.n_frames)[:num_events].to(device)
            
            event_texts = [event.event_type for event in graph.events[:num_events]]
            type_embeddings = self.encode_event_text(event_texts, text_encoder, tokenizer, device)
            
            time_features = torch.tensor(
                [[event.start_time, event.end_time] for event in graph.events[:num_events]],
                dtype=torch.float32,
                device=device
            )
            time_embeddings = self.time_encoder(time_features)
            
            graph_embeddings = torch.zeros(num_events, self.hidden_dim//2, device=device)
            for i in range(num_events):
                relation_features = adjacency[i, :num_events, :].float()  # [num_events, 5]
                encoded_relations = self.graph_encoder(relation_features)  # [num_events, hidden_dim//2]
                graph_embeddings[i] = encoded_relations.mean(dim=0)
            
            temporal_embeddings = self.temporal_encoder(temporal_features)
            
            combined_embeddings = torch.cat([
                type_embeddings + time_embeddings,  
                graph_embeddings + temporal_embeddings
            ], dim=-1)
            combined_embeddings = self.fusion_layer(combined_embeddings)
            
            position_embeddings = self.position_embeddings(self.position_ids[:, :num_events])
            combined_embeddings = combined_embeddings + position_embeddings
            
            event_embeddings[b, :num_events] = combined_embeddings
            event_mask[b, :num_events] = True
            
        attention_mask = ~event_mask
        encoded_embeddings = self.transformer(event_embeddings, src_key_padding_mask=attention_mask)
        output = self.output_mapping(encoded_embeddings)
        
        return output, event_mask
