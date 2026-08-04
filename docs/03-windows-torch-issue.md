# Windows 下 pytest 进程内 torch 导入偶发 access violation

> 观察记录，非最终归因。日期 2026-08-04。

## 现象

- 崩溃点固定：`import torch` 执行到 `torch/__init__.py`（约 442 行）时，在 `importlib` 的 `create_module` 阶段触发 `Windows fatal exception: access violation`。
- 仅出现在 pytest 进程内。独立脚本 `python -c "import torch"` 与 `python -c "import cv2; import torch"` 从未复现。
- 间歇性：同一测试文件有时连续多次通过，有时收集期即崩溃。
- 崩溃发生时机与进程内已加载的原生库有关。观察到崩溃案例中，OpenCV（cv2）原生库已在同一进程内加载。

## 受控实验

实验文件：`inference-service/tests/diag_cv2_torch.py`（模块级 `import cv2` 后再 `import torch`）。

| 配置 | 结果 |
|---|---|
| 独立脚本，cv2 先于 torch | 多次运行零崩溃 |
| pytest 最小模块，仅 import torch | 通过 |
| pytest 模块，cv2 先于 torch（受控 8 连跑） | 8/8 通过，零崩溃 |
| pytest 全量单元层（收集阶段零 torch） | 连续 3 次零崩溃 |
| 此前全量套件运行 | 数十次运行中约 4 至 5 次收集期崩溃 |

结论：受控实验无法稳定复现，崩溃仅在全量套件运行中出现且频率低。**加载顺序假设未被证实**，不能作为确定性根因。

## 根因

未确定。崩溃点为 torch 原生扩展的 `create_module` 阶段，表现为 Windows access violation；与进程全局状态相关，间歇性出现。

## 已排除的解释

- opencv-python 5.x 与 torch 的确定性冲突：未做过受控对照实验验证 opencv 5.x 必然触发，且受控 8 连跑（cv2→torch）零崩溃，故文档只保留"当前稳定 pin 为 opencv-python==4.12.0.88"这一经过运行验证的事实。

## 缓解措施（现行）

1. 测试分层：`pytest.ini` 默认 `-m "not integration and not gpu"`，单元层收集阶段不 import torch、不加载模型。
2. `torch` 只在 gpu 层测试的函数体内通过 `pytest.importorskip("torch")` 惰性加载。
3. 不再采用"失败后重试"作为长期策略。若单元层出现崩溃即为回归，需回到本文件排查。

## 后续排查建议

- 在崩溃现场用 `procdump` / WinDbg 抓取 dump，确认冲突的加载基址。
- 实验 `torch._C` 单独预加载（如 conftest 中最先 `import torch`）是否能消除竞态。
- 升级 torch 或切换 opencv 构建（headless 版）后再观察。