# CDAQ

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Run experiments

```bash
python trainqat.py --data data/coco.yaml --weights path/to/pretrained_fp32_yolov5s.pt --epochs 100 --batch-size 32 --w_bit 4 --a_bit 4 --optimizer AdamW --per_channel True --first_w_bit 8 --first_a_bit 8 --last_w_bit 8 --last_a_bit 8 --method cdaq
```
