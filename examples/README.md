# Easy-Translate Examples

This folder contains examples of how to use Easy-Translate with different models and configurations.
You can adapt these examples to your own use cases. If you have any questions, please feel free to open an issue.

### MT Models

```bash
m2m100-1.2B.sh
m2m100-12B_fp16.sh
nllb200_3B_fp16.sh
opusMT.sh
mbart.sh
small100.sh
```

#### Multi-GPU example
```bash
m2m100-1.2B_2GPUs.sh
```
#### Running large models on customer hardware
```bash
m2m100-12B_8bits.sh
nllb200_3B_8bit.sh
nllb200-moe-54B_1GPU_int8.sh
nllb200-moe-54B_1GPU_int4.sh
```

### Running LLMs with translation prompts

```bash
FlanT5-large.sh
LLaMA.sh
Vicuna.sh
Alpaca-Lora.sh
```