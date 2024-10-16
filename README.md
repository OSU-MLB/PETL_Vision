
<div align=center>
<h1 style="font-size: 160pt;" align=center><strong>Lessons Learned from a Unifying Empirical Study of Parameter-Efficient Transfer Learning (PETL) in
Visual Recognition</strong></h2>
<p>

 ![GitHub Repo stars](https://img.shields.io/github/stars/OSU-MLB/PETL_Vision?style=for-the-badge&color=red)
![GitHub forks](https://img.shields.io/github/forks/OSU-MLB/PETL_Vision?style=for-the-badge&color=blue)

</p>
</div>

<div align="center">
  <p align="center">
    ·
    <a href="https://arxiv.org/abs/2409.16434">Paper</a>
    ·
    <a href="https://github.com/OSU-MLB/PETL_Vision">Code</a>
<!--     ·
    <a href="">Home</a> -->

  </p>
</div>

## 🔥 <span id="head1"> *News and Updates* </span>

* ✅ [2024/09/25] Codebase is created.

# Environment Setup  
```bash  
source env_setup.sh
```  
  
# Data Preparation
You can put all the data in a folder and pass the path to `--data_path` argument.  

## VTAB-1k  
We provide two ways for preparing VTAB-1k dataset  
  
1. TFDS  
   - Following [VPT](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md)'s instruction to download the data through  
      TFDS  
   - When you pass the data name for `--data` argument, add `tfds_vtab` before each VTAB dataset,  
      e.g. `tfds_vtab-cifar(num_classes=100)`  
   - You can find all the dataset names of the TFDS version VTAB1-k from `TFDS_DATASETS` in `utils/global_var.py`  
2. Processed Version  
   - Download the processed version from [here](https://1drv.ms/u/s!AmrFUyZ_lDVGinv8y53HX3_rLrpq?e=P5EN5G).  
   - When you pass the data name for `--data` argument, add `process_vtab`, e.g. `process_vtab-cifar`  
 - You can find all the dataset names of the processed version VTAB1-k from `VTAB_DATASETS` in `utils/global_var.py`  
  
  
Both ways apply the same transforms to the dataset: Resize, ToTensor and optionally Normalize with mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225].  
  
## Robustness to Domain Shift  
 - Download ImageNet from the [official website](https://www.image-net.org/download.php)  
 - When you pass the data name for `--data` argument, use `imagenet-fs`  
 - Download ImageNet-Sketch from https://github.com/HaohanWang/ImageNet-Sketch  
 - Download ImageNet-A from https://github.com/hendrycks/natural-adv-examples  
 - Download ImageNet-R from https://github.com/hendrycks/imagenet-r  
 - Download ImageNetV2 from https://github.com/modestyachts/ImageNetV2.  
  
## CIFAR100  
Download when you run the code  
  
# Pretrain weights
Download the pretrained weights from the following links and put them in the `pretrained_weights` folder.  
1. [ViT-B-Sup-21k](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) rename it as `ViT-B_16_in21k.npz`  
2. [ViT-B-CLIP](https://huggingface.co/timm/vit_base_patch16_clip_224.openai/tree/main) rename it as `ViT-B_16_clip.bin`  
3. [ViT-B-DINOV2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) rename it as `ViT-B_14_dinov2.pth`  
  
# Running  
Example commands for implemented methods:  
  
## VTAB-1K
```bash  
### SSF  
CUDA_VISIBLE_DEVICES=0  python main.py --ssf --data processed_vtab-dtd  --debug  
  
### VPT  
CUDA_VISIBLE_DEVICES=0  python main.py --vpt_mode deep --vpt_num 10 --data processed_vtab-dtd  --debug  
  
### AdaptFormer  
CUDA_VISIBLE_DEVICES=0  python main.py --ft_mlp_module adapter --ft_mlp_mode parallel --ft_mlp_ln before --adapter_bottleneck 64 --adapter_init lora_kaiming --adapter_scaler 0.1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  
  
### Convpass  
CUDA_VISIBLE_DEVICES=0  python main.py --ft_attn_module convpass --ft_attn_mode parallel --ft_attn_ln after --ft_mlp_module convpass --ft_mlp_mode parallel --ft_mlp_ln after --convpass_scaler 0.1 --data processed_vtab-dtd  --debug --optimizer adamw --data_path data_folder/vtab_processed  
  
### VPT Version (zero-init) Adapter  
CUDA_VISIBLE_DEVICES=0  python main.py --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init zero --adapter_scaler 1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  
  
### lora init Pfeiffer Adapter  
CUDA_VISIBLE_DEVICES=0  python main.py --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init lora_xavier --adapter_scaler 1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  
  
### random init Pfeiffer Adapter  
CUDA_VISIBLE_DEVICES=0  python main.py --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  
  
### random init Houlsby Adapter  
CUDA_VISIBLE_DEVICES=0  python main.py --ft_attn_module adapter --ft_attn_mode sequential_after --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  
  
### LoRA  
CUDA_VISIBLE_DEVICES=0  python main.py --lora_bottleneck 8 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  
  
### FacT TK  
CUDA_VISIBLE_DEVICES=0  python main.py --data processed_vtab-dtd  --optimizer adamw --fact_type tk --drop_path_rate 0.1 --debug --fact_dim 32 --fact_scaler 0.1 --data_path data_folder/vtab_processed  
  
### Fact TT  
CUDA_VISIBLE_DEVICES=0  python main.py --data processed_vtab-dtd  --optimizer adamw --fact_type tt --drop_path_rate 0.1 --data_path data_folder/vtab_processed  
  
### ReAdapter  
CUDA_VISIBLE_DEVICES=0  python main.py --ft_attn_module repadapter --ft_attn_mode sequential_before --ft_mlp_module repadapter --ft_mlp_mode sequential_before --repadapter_bottleneck 8 --repadapter_scaler 10 --data processed_vtab-smallnorb_azi  --optimizer adamw --debug  
  
### BitFit  
CUDA_VISIBLE_DEVICES=0  python main.py --bitfit --data processed_vtab-clevr_count  --debug --data_path data_folder/vtab_processed  
  
### VQT  
CUDA_VISIBLE_DEVICES=0  python main.py --vqt_num 10 --data_path data_folder/vtab_processed  
  
### LayerNorm  
CUDA_VISIBLE_DEVICES=0  python main.py --ln --data processed_vtab-clevr_count  --debug --data_path data_folder/vtab_processed  
  
### DiffFit  
CUDA_VISIBLE_DEVICES=0  python main.py --difffit --data processed_vtab-clevr_count  --debug --data_path data_folder/vtab_processed  
  
### Attention  
CUDA_VISIBLE_DEVICES=0  python main.py --attention_index 9 10 11 --attention_type qkv --data_path data_folder/vtab_processed  
  
### MLP  
CUDA_VISIBLE_DEVICES=0  python main.py --mlp_index 9 10 11 --mlp_type full --data_path data_folder/vtab_processed  
  
### block  
CUDA_VISIBLE_DEVICES=0  python main.py --block_index 9 10 11 --data_path data_folder/vtab_processed  
  
### full  
CUDA_VISIBLE_DEVICES=0  python main.py --full --eval_freq 2 --data_path data_folder/vtab_processed  
```  

## Tuning Code for VTAB-1K
```bash  
CUDA_VISIBLE_DEVICES=0 python main_tune.py --data processed_vtab-caltech101 --default experiment/config/default_vtab_processed.yml --tune experiment/config/method/lora.yml --lrwd experiment/config/lr_wd_vtab_processed.yml  
```  
All the config files can be found in the `experiment/config` folder.   
The `default` config file is used to set the default hyperparameters for the training.  
The `tune` config file is used to set the tunable hyperparameters for the method.  
The `lrwd` config file is used to set the learning rate and weight decay for the training.  
We provide the tuning results `tune_summary.csv` and final run results `final_result.json` in the `tune_output` folder. 
If you want to rerun the final run using the best tuning parameters, you can specify a new output name for the final run using this option `--final_output_name`. 

To tune all methods for one VTAB-1K dataset, here is an example command:  
```bash
dataset='processed_vtab-dsprites_loc'
for METHOD in lora_p_adapter repadapter rand_h_adapter adaptformer convpass fact_tk fact_tt lora difffit full linear ssf bitfit ln vpt_shallow vpt_deep
  do
    CUDA_VISIBLE_DEVICES=0 python main_tune.py --data ${dataset}  --default experiment/config/default_vtab_processed.yml --tune experiment/config/method/$METHOD.yml --lrwd experiment/config/lr_wd_vtab_processed.yml
  done
```


## Robustness to Domain Shift  
We use the CLIP ViT-B/16 model and add an FC layer as the prediction head with zero-initialized bias and initialize weights using the class label text embedded
by the text encoder. Subsequently, we discard the text encoder and apply PETL methods to the visual encoder, fine-tuning only the PETL modules and the head.

The code to generate the prediction head for CLIP can be found at [build_clip_zs_classifier.py](experiment/build_clip_zs_classifier.py).  

This is an example command to run a PETL method for the 100-shot ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0  python main.py --data fs-imagenet --data_path data_folder/imagenet/images --warmup_lr_init 1.0e-7 --lr 0.00003 --wd 0.005 --eval_freq 1 --store_ckp --lora_bottleneck 32  --batch_size 256 --final_acc_hp --early_patience 10
```
This is an example command to tune a few methods for the 100-shot ImageNet dataset:
```bash
for METHOD in rand_h_adapter_64 lora_16 ssf
do
  for dataset in fs-imagenet
    do
      CUDA_VISIBLE_DEVICES=0 python main_tune.py --data ${dataset} --test_bs 2048 --bs 256  --default experiment/config/clip_fs_imagenet.yml --tune experiment/config/method-imagenet/$METHOD.yml --lrwd experiment/config/lr_wd_clip_imagenet.yml
    done
done
```

To evaluate the performance of fine-tuned models on OOD data, here is an example command:
```bash  
for METHOD in rand_h_adapter_64 lora_16 ssf
do
  for dataset in fs-imagenet eval_imagenet-v2 eval_imagenet-r eval_imagenet-a eval_imagenet-s
    do
      CUDA_VISIBLE_DEVICES=0 python main_evaluate.py --test_data ${dataset} --bs 2048 --default experiment/config/clip_fs_imagenet.yml --tune experiment/config/method-imagenet/$METHOD.yml --data_path /research/nfs_chao_209/zheda
    done
done
```  

When it comes to applying WiSE to PETL methods, there are two types of PETL methods. 
Methods that insert additional parameters to the model, such as Adapter, and methods that directly fine-tuned existing parameters, such as bitfit. 
For the former, we use `merge_petl.py` and use `merge_model.py` for the latter. 
Example commands for each type:
```bash
CUDA_VISIBLE_DEVICES=0 python merge_model.py --bs 1024  --default experiment/config/clip_fs_imagenet.yml --tune experiment/config/method-imagenet/ln.yml
CUDA_VISIBLE_DEVICES=0 python merge_petl.py --bs 1024 --default experiment/config/clip_fs_imagenet.yml --tune experiment/config/method-imagenet/fact_tk_64.yml
```

To get the WiSE curve plots, you can use `WiSE_PETL.ipynb`.

## Many-shot (Full) Datasets
We use three datasets for mann-shot experiments: CIFAR100, Clevr-distance and RESISC.
Here is an example command to run the PETL methods for the Clevr-distance dataset. Config files for CIFAR and RESISC can be found in the `experiment/config` folder.
```bash
for METHOD in rand_h_adapter_8 lora_8 fact_tk_8 
do
  for dataset in clevr
    do
      CUDA_VISIBLE_DEVICES=0 python main_tune.py --data ${dataset}  --default experiment/config/default_clevr.yml --tune experiment/config/method_clevr/$METHOD.yml --lrwd experiment/config/lr_wd_clevr.yml
    done
done
```
## Results
We provide the tuning results `tune_summary.csv`, final run results `final_result.json` and other results for all the experiments in the `tune_output` folder.


## To add a new method  
1. add a new module file in `./model/`.   
2. add the module accordingly in [block.py](model/block.py), [mlp.py](model/mlp.py), [patch_embed.py](model/patch_embed.py), [vision_transformer.py](model/vision_transformer.py), [attention.py](model/attention.py).
3. add the name of the added module in TUNE_MODULES and modify `get_model()` in `./experiment/build_model.py` accordingly.
4. add required arguments in `main.py`  

## To add a new dataset
1. add a new dataset file in `/data/dataset`.
2. modify [build_loader.py](experiment/build_loader.py) to include the new dataset.

## To add a new backbone
1. modify `get_base_model()` in [build_model.py](experiment/build_model.py).

## To run new hyperparameter tuning
1. add a general config file and a learning rate and weight decay config file in `./experiment/config/`.
2. add new method config files. You can create a new folder in `./experiment/config/` for your experiment.

## To collect logits and ground truth for PETL methods
Use [main_collect_prediction.py](main_collect_prediction.py). 
  
## Reference:
- SSF: https://github.com/dongzelian/SSF  
- Adaptformer: https://github.com/ShoufaChen/AdaptFormer  
- ConvPass: https://github.com/JieShibo/PETL-ViT  
- LoRA: https://github.com/ZhangYuanhan-AI/NOAH  
- Adapter: https://github.com/ZhangYuanhan-AI/NOAH, https://arxiv.org/pdf/2007.07779.pdf, https://arxiv.org/pdf/1902.00751.pdf 
- VPT: https://github.com/KMnP/vpt  
- FacT: https://github.com/JieShibo/PETL-ViT  
- RepAdpater: https://github.com/luogen1996/RepAdapter
- VQT: https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Visual_Query_Tuning_Towards_Effective_Usage_of_Intermediate_Representations_for_CVPR_2023_paper.pdf
- BitFit: https://arxiv.org/pdf/2106.10199.pdf  
- DiffFit: https://arxiv.org/pdf/2304.06648.pd


