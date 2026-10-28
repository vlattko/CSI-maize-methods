# Connectivity-aware foreground distillation

The two arms use the same 491,873-parameter U-Net, saved initial weights, training-frame order and five-epoch budget. One arm minimizes equally weighted binary cross-entropy and soft Dice. The other adds 0.25 times soft-clDice with ten morphological iterations. Inputs use a 640 by 481 grid; validation predictions are resized to the original 1,024-pixel processing canvas before scoring against the unchanged teacher mask.

The loss follows [Shit et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html). It was developed for curvilinear structures such as vessels, neurons and roads. Here we test whether that objective preserves thin plant foreground support. The implementation is original code based on the published algorithm. A finite morphological loss against generated masks does not establish botanical topology or correct individual leaves.

The frozen training genotypes are B73, Mo17, Os024445 and Os622448. Validation uses Os6251/15. Os7798/2 is excluded from training and selection. The training and validation manifests bind native source, processing image and original SAM teacher-mask bytes. All visits and views of each genotype remain in the same partition. The dense teacher masks differ from the polygon representation used to train YOLO; comparisons are operational model comparisons, not an isolated architecture experiment.

Validation selects the checkpoint and foreground probability threshold from 0.35, 0.50 and 0.65. This pixel probability is distinct from YOLO object confidence and a SAM proposal score. Models with a high mean area IoU can still lose tips. We therefore retain teacher endpoint and thin-skeleton support at several pixel tolerances, alongside Dice, IoU, nonempty prediction counts and connected components.

```sh
python -m unittest discover -v
python train_topology.py --help
python audit_topology.py --help
python infer_topology.py --help
```

Training requires the explicit local preparation manifest and PyTorch environment. No model weights or private images are included in the public source. Checkpoint cards record the selected threshold and checkpoint hash. Independent audit records confirm identical initialization/orders, original-grid validation, input hashes and exclusion of test images during selection. The common foreground evaluator performs the later held-out comparison without changing those choices.

The connectivity-aware arm increased validation teacher endpoint support by about two percentage points while slightly reducing IoU. This can preserve genuine thin foregrounds and teacher artefacts alike. Expert-reviewed complete-leaf annotations and downstream tracking checks remain separate requirements.
