# Model Inference Performance

| data type | QAT (finetune 2 epoch) | PTQ (1 batch data for calibration) |
|-----------|------------------------|------------------------------------|
| int8      | 92.85%                 | 98.80%                             |
| int7      | 84.42%                 | 83.78%                             |
| int6      | 52.20%                 | 47.23%                             |
| int5      | 28.20%                 | 24.84%                             |
| int4      | 14.99%                 | 18.27%                             |
| int3      | 18.00%                 | 12.46%                             |
| int2      | 11.69%                 | 12.56%                             |

