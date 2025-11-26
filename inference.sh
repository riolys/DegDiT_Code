export CUDA_VISIBLE_DEVICES=0
export model_dir="outputs/your_path_save_models"

# for audiocondition
python3 models/inference_deg.py \
    --original_args "${model_dir}/summary.jsonl" \
    --model "${model_dir}/best" \
    --graph_encoder "${model_dir}/graph_encoder_best.safetensors" \
    --test_file "desed_eval/metadata/eval/puclic_dsg.json" \
    --num_steps 50 \
    --guidance 4 \
    --duration 10 \
    --output_dir "${model_dir}" \
    --use_text_encoder

