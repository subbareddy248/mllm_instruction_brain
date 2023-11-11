# Brain Activity Prediction

In this task, we will take embeddings from pre-trained instruction-based models and train a linear regressor to take 
these embeddings and output the corresponding brain activity (fMRI).

**Hypothesis**: Here, we would like to use multimodal models because they _should_ be more aligned with our brain.

## Models to consider

|              Model         |                                                Checkpoints                                                                             |
|:--------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| `instructblip-flan-t5-xl`  | [ [HuggingFace](https://huggingface.co/Salesforce/instructblip-flan-t5-xl) ]                                                           |
| `instructblip-flan-t5-xxl` | [ [HuggingFace](https://huggingface.co/Salesforce/instructblip-flan-t5-xxl) ]                                                          |
| `instructblip-vicuna-7b`   | [ [HuggingFace](https://huggingface.co/Salesforce/instructblip-vicuna-7b) ]                                                            |
| `instructblip-vicuna-13b`  | [ [HuggingFace](https://huggingface.co/Salesforce/instructblip-vicuna-13b) ]                                                           |
| `blip2-flan-t5-xl`         | [ [HuggingFace](https://huggingface.co/Salesforce/blip2-flan-t5-xl) ]                                                                  |
| `blip2-flan-t5-xxl`        | [ [HuggingFace](https://huggingface.co/Salesforce/blip2-flan-t5-xxl) ]                                                                 |
| `blip2-vicuna-13b`         | Weights Not Available                                                                                                                  |
| `blip2-vicuna-7b`          | [ [Pretrained Weights](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_vicuna7b.pth) ] |
| `Meta-Transformer-B16`     | [ [Google Drive](https://drive.google.com/file/d/19ahcN2QKknkir_bayhTW5rucuAiX0OXq/view?usp=sharing) ] [ [OpenXLab](https://download.openxlab.org.cn/models/zhangyiyuan/MetaTransformer/weight//Meta-Transformer_base_patch16_encoder) ]                                |
| `Meta-Transformer-L14`     | [ [Google Drive](https://drive.google.com/file/d/15EtzCBAQSqmelhdLz6k880A19_RpcX9B/view?usp=drive_link) ] [ [OpenXLab](https://download.openxlab.org.cn/models/zhangyiyuan/MetaTransformer/weight//Meta-Transformer_large_patch14_encoder) ]                             |


## TODO

- [ ] Add some visual models that also deal with videos
