# Mock 演示图片

在已安装后端依赖的环境中运行：

```bash
python scripts/create_sample_images.py
```

脚本生成四张合法 PNG，分别稳定触发放行、中风险复检、高风险剔除、严重风险停线审批。结果来自图片内容散列，而不是文件名或前端硬编码。
