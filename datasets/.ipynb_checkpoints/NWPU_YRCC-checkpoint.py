# 同济子豪兄 2023-6-25
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class NWPU_YRCC(BaseSegDataset):

    METAINFO = {
        'classes':['others', 'water', 'shore ice'],
        'palette':[[0,255,255], [0,255,0], [255,0,255]]
    }

    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',   # 标注mask图像的格式
                 reduce_zero_label=True, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)