export TORCH_DISTRIBUTED_DEBUG=DETAIL

# for audioset
accelerate launch models/train_deg.py \
    --datasetname audioset \
    --config configs/audioset/deg.yaml \
    --use_dsg \
    --dsg_max_events 32 \
    --learning_rate 3e-5 \
    --num_warmup_steps 100 \
    --prefix "" \
    --save_every 20 \
    --text_column='time_captions' \
    --audio_column="location" \
    --load_from_checkpoint="models/declare-lab/TangoFlux/tangoflux.safetensors"

