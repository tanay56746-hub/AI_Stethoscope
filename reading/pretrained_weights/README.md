# Pretrained Weights

Due to GitHub's 100MB single-file size limit, the official PANNs (Pretrained Audio Neural Networks) checkpoint is not stored directly in this repository.

### Download PANNs CNN14 Backbone:
Download the official pretrained model weights (`Cnn14_mAP=0.431.pth`, ~312 MB) from Zenodo:
- **Download Link**: [Zenodo PANNs Cnn14 Checkpoint](https://zenodo.org/record/3987831/files/Cnn14_mAP=0.431.pth)
- **Destination**: Place the downloaded file into this folder:
  ```
  pretrained_weights/Cnn14_mAP=0.431.pth
  ```

### Final Multi-Task Model Weights:
The final trained multi-task model checkpoints and ablation models are committed and available directly under:
- `experiments/multitask_ablations/cnn_physical_panns/checkpoints/best_checkpoint.pt`
