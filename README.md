# Object Detection Using Mask R-CNN

本仓库的开发计划见[项目下一步开发计划](https://github.com/guanfuchen/objdet/issues/8)

---
## Datasets

```
cd maskrcnn_simple
tree datasets -L 2

datasets
├── CaltechPedestrians -> $HOME/Data/CaltechPedestrians
├── voc
│   └── VOC2007 -> $HOME/Data/voc/VOC2007
└── wider -> $HOME/Data/wider

CaltechPedestrians数据集
root:$HOME/Data/CaltechPedestrians/convert
format:{set**}_{V***}_{frame_num}.png
.
├── annotations.json
└── images
    ├── set00_V000_0.png
    ├── set00_V000_1000.png
    ├── set00_V000_1001.png
    ├── set00_V000_1002.png
    ├── set00_V000_1003.png
    ├── set00_V000_1004.png
    ├── set00_V000_1005.png

```