# Analogical Reasoning

The repo is the source code for [Relevant or Random: Can LLMs Truly Perform Analogical Reasoning?]

## Setup

### 1. Create Conda Environment and Install Dependencies
```
conda env create -f analogical.yml
conda activate analogical
```

### 2. Run code

For gpt-3.5-turbo:
```
#### self_generate_type refers to the type of self-generated examples
## GSM8K
python HandleGSM8K.py --self_generate_type relevant --seed 42 --api_key YOUR_OPENAI_API_KEY
## MATH
python HandleMATH.py --self_generate_type relevant --seed 42 --api_key YOUR_OPENAI_API_KEY
## BBH
python HandleBBH.py --self_generate_type relevant --seed 42 --api_key YOUR_OPENAI_API_KEY
```


For Llama series models:
```
#### self_generate_type refers to the type of self-generated examples
#### start a vLLM server named as args.model before running the following command
python HandleGSM8K_llama.py --self_generate_type relevant --seed 42
```
