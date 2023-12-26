from mmdet.apis import DetInferencer

# 初始化模型
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')
# models 是一个模型名称列表，它们将自动打印
models = DetInferencer.list_models('mmdet')

# 推理示例图片
inferencer('demo/demo.jpg', out_dir="output")
