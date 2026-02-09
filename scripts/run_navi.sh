set -euo pipefail

export PYTHONPATH=.

# ==================== Configuration ====================
# GPU Configuration
CUDA_DEVICES="0,1,2,3"      # Specify GPU device IDs, e.g. "0,1,2,3" or "4,5,6,7"
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

# Model Configuration
MODEL_PATH="your_model_path"  # Modify to your model path, e.g. /path/to/UI-Venus-1.5-Pro

# Input/Output Configuration
INPUT_FILE="examples/trace/trace.json"
OUTPUT_FILE="./results/navi/saved_trace_$(date +%Y%m%d_%H%M%S).json"

# Prompt Type Configuration
PROMPT_TYPE="mobile"        # Prompt type: "web" for web tasks or "mobile" for mobile tasks (default: mobile)

# vLLM Configuration
TENSOR_PARALLEL_SIZE=4      # Tensor parallel size (should match the number of GPUs in CUDA_DEVICES)
GPU_MEMORY_UTIL=0.8         # GPU memory utilization (adjust based on available memory, reduce if insufficient)
MAX_PIXELS=12845056         # Maximum pixels
MIN_PIXELS=3136             # Minimum pixels
MAX_MODEL_LEN=16192         # Maximum model length
MAX_TOKENS=2048             # Maximum generated tokens
TEMPERATURE=0.0             # Sampling temperature
HISTORY_LENGTH=5            # History length

# ==================== Execution ====================
mkdir -p ./results/navi

echo "UI-Venus 1.5 Navigation Evaluation"
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Prompt Type: ${PROMPT_TYPE}"
echo "Tensor Parallel: ${TENSOR_PARALLEL_SIZE}"

python models/navigation/runner.py \
    --model_path="${MODEL_PATH}" \
    --input_file="${INPUT_FILE}" \
    --output_file="${OUTPUT_FILE}" \
    --tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    --max_pixels=${MAX_PIXELS} \
    --min_pixels=${MIN_PIXELS} \
    --max_model_len=${MAX_MODEL_LEN} \
    --max_tokens=${MAX_TOKENS} \
    --temperature=${TEMPERATURE} \
    --history_length=${HISTORY_LENGTH} \
    --prompt_type=${PROMPT_TYPE}

echo "Navigation evaluation complete! Results saved to: ${OUTPUT_FILE}"
