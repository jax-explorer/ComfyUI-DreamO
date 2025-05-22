# ComfyUI-DreamO

https://github.com/bytedance/DreamO
ComfyUI Warpper

Note:
This only a warpper. if you need full native comfyui impl. you can find other.


### Update:
many user need diffusers>=0.33.0
create support_0.33.0 branch, change code by [https://github.com/bytedance/DreamO/issues/20](https://github.com/bytedance/DreamO/issues/20)
and now this branch support diffusers>=0.33.0

main branch need diffusers==0.31.0

support_0.33.0 branch diffusers>=0.33.0

### Config
open offload && 8bit can run on low vram, no open offload need 40GB.

Flux model auto download to models/diffusers

lora need to download to models/lora

lora:
https://huggingface.co/ByteDance/DreamO/tree/main



IP reference (subject)

![show](./assets/show_1.png)


style reference 

note: prompt need add "generate a same style image."

![show](./assets/show_2.png)


ID (face) + IP  reference

![show](./assets/show_3.png)


## online run:

https://www.comfyonline.app/explore/app/dreamo-image-generate
