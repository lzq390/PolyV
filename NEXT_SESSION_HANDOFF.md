# PolyV Session Handoff

Date: 2026-06-08  
Workspace: `/home/lzq390/gith/PolyV`

## 项目目标

PolyV 当前是在视频中检测反应状态，核心状态包括：

- `LIQUID_STIRRING`
- `FINAL_GEL_ROD_CLIMBING`

当前阶段重点不是最终状态判定逻辑，而是动态 ROI 划分，尤其是 `liquid_roi`。用户要求：

- `liquid_roi` 不再从瓶身边界或搅拌杆位置推导。
- 直接从图像中动态寻找白色/乳白色液体椭圆。
- `liquid_roi` 是该白色椭圆的外接矩形，即椭圆作为矩形内切椭圆。
- 本轮只重做 `liquid_roi`，`rod_roi` 和 `sparse_roi` 暂不重做。

## 当前进度

已完成新版 `liquid_roi` 策略，并生成 v3 批量对比图。用户上一轮反馈：除 trim 外其它视频明显偏差，参考用户手绘红框重新调整。当前 v3 结果相对上一版已明显收紧，基本贴近红框位置。

最终对比图：

`/home/lzq390/gith/PolyV/outputs/liquid_roi_batch_review_v3/liquid_roi_contact_sheet.jpg`

结果 manifest：

`/home/lzq390/gith/PolyV/outputs/liquid_roi_batch_review_v3/manifest.json`

## 关键代码文件

- `/home/lzq390/gith/PolyV/polyv_detector/dynamic_detector.py`
  - 动态 ROI 主逻辑。
  - 当前 `liquid_roi` 检测、EMA 更新、ROI 输出均在这里。
- `/home/lzq390/gith/PolyV/tests/test_dynamic_detector.py`
  - 动态 ROI 单元测试。
  - 已新增 liquid ROI 可靠椭圆后更新的测试。
- `/home/lzq390/gith/PolyV/tools/generate_liquid_roi_batch_review.py`
  - 新增脚本，用已有 manifest 的视频列表重新生成 `liquid_roi` 批量 overlay/contact sheet。

## 当前 liquid_roi 策略

入口函数：

`_detect_liquid_ellipse_bounds(frame, config)`

逻辑分两类候选：

1. compact 候选，默认使用
2. broad 候选，只用于 trim 这类右侧、靠底、较大的椭圆场景

### compact 候选

函数：

`_detect_compact_liquid_ellipse_window(...)`

搜索区域：

- `x = 0.18W .. 0.84W`
- `y = 0.62H .. 0.94H`

mask：

- `core_mask`：更严格的白色/乳白色核心
  - `value >= 175`
  - `sat <= 85`
  - `R >= 175`
  - `G >= 168`
  - `B >= 125`
  - `channel_span <= 85`
- `milk_mask`：较宽松乳白色 mask，见 `_milky_liquid_mask(...)`

候选窗口：

- width ratios: `0.20, 0.24, 0.28, 0.30, 0.32, 0.34`
- height ratios: `0.12, 0.16, 0.18, 0.20`
- step: `0.015W`, `0.015H`

评分因素：

- 白色密度
- 椭圆内部密度
- 椭圆相对角落增益
- 尺寸先验，目标大约 `0.30W x 0.17H`
- 中心位置弱先验
- 垂直位置先验
- 底部惩罚，避免下沉到锅沿/黄色标签

### broad 候选

函数：

`_detect_broad_liquid_ellipse_window(...)`

用途：

- 保留 trim 视频中液体椭圆更靠右、靠底、范围更大的情况。

搜索区域：

- `x = 0.18W .. 0.84W`
- `y = 0.62H .. 0.995H`

只有 `_prefer_broad_liquid_box(...)` 判断通过时才使用 broad：

- broad score 足够高
- left 在 `0.42 .. 0.58`
- right 在 `0.62 .. 0.80`
- bottom `>= 0.94`
- width 在 `0.20 .. 0.36`
- height 在 `0.18 .. 0.30`

这主要是为了让 trim 保持用户认为接近正确的框。

## 动态更新策略

`DetectionConfig` 当前相关默认值：

- `dynamic_roi_ema_alpha = 0.35`
- `dynamic_roi_update_interval_sec = 5.0`

新增/当前关键逻辑：

- `DynamicRoiTracker.update_liquid(frame_rgb)`
  - 只更新 `liquid_x0/y0/x1/y1` 和 `liquid_ellipse_score`
  - 不更新 `rod_roi` / `sparse_roi` 的几何来源
  - 当新检测 `liquid_ellipse_score` 太低时跳过更新
- `GelClimbDetector._dynamic_rois_for_frame(...)`
  - 默认每 5 秒触发一次 `update_liquid`
  - 解决 #1 视频前 20 秒标定框偏上、61 秒实际液体在下方的问题
  - 避免每秒完整搜索导致批量视频处理非常慢

注意：

- 之前尝试每帧/每秒都更新 liquid ROI，会导致 7 个视频批处理明显变慢，已改成 5 秒限频。

## v3 批量结果

生成命令：

```bash
/home/lzq390/miniconda3/bin/conda run -n CV python tools/generate_liquid_roi_batch_review.py --outputs outputs/liquid_roi_batch_review_v3 --time-sec 61
```

结果摘要：

| # | video | liquid_roi | white_coverage | ellipse_score |
|---|---|---:|---:|---:|
| 1 | `VID_20260601_142527.mp4` | `[0.34601, 0.72858, 0.54994, 0.88876]` | `0.8930` | `0.8782` |
| 2 | `VID_20260601_144516.mp4` | `[0.28594, 0.70926, 0.58594, 0.86944]` | `0.8608` | `0.9215` |
| 3 | `VID_20260601_150311.mp4` | `[0.37685, 0.65870, 0.63879, 0.82691]` | `0.8981` | `0.9383` |
| 4 | `VID_20260603_151103 - Trim.mp4` | `[0.46830, 0.76852, 0.73547, 0.99537]` | `0.7179` | `0.8058` |
| 5 | `飞书20260604-111405.mp4` | `[0.37562, 0.64694, 0.62066, 0.81472]` | `0.9348` | `0.9342` |
| 6 | `飞书20260604-111732.mp4` | `[0.37562, 0.64694, 0.62066, 0.81472]` | `0.9348` | `0.9342` |
| 7 | `飞书20260604-111744.mp4` | `[0.37685, 0.65870, 0.63879, 0.82691]` | `0.8981` | `0.9383` |

## 验证状态

当前测试命令：

```bash
/home/lzq390/miniconda3/bin/conda run -n CV python -m unittest discover -s tests -p 'test*.py' -v
```

最近结果：

- 14 tests
- 全部 OK

新增测试重点：

- 白色椭圆横向移动时 `liquid_roi` 跟随。
- 干扰白色区域存在时优先选下方液体椭圆。
- 检测失败时回退到 `config.liquid_roi`。
- 标定后出现可靠下方椭圆时，只让 `liquid_roi` 更新，`rod_roi` 保持稳定。

## 环境注意

使用 conda 环境：

```bash
/home/lzq390/miniconda3/bin/conda run -n CV ...
```

已知情况：

- 系统 Python 可能缺 `cv2` / `numpy` / `PIL`。
- `CV` 环境有 `cv2` 和 `numpy`，但之前没有 `PIL`，所以新增 batch review 脚本使用 OpenCV 画图。
- 当前目录不是 git 仓库，`git status` 会报 `fatal: not a git repository`。

## 待确认 / 下一步

1. 让用户确认 v3 contact sheet 是否满足肉眼红框标准。
2. 如果用户认为 #3/#5/#6/#7 框仍偏窄或偏高，可优先调整 compact 的目标尺寸：
   - `width target`: 当前约 `0.30W`
   - `height target`: 当前约 `0.17H`
   - 底部惩罚会影响是否覆盖更低的白色区域
3. 如果用户认为 #2 右侧仍应更宽，需要谨慎放宽：
   - 右扩过多容易重新吃到锅底/黄色标签。
   - 建议只调 `width_ratios` 或尺寸先验，不要恢复旧的 broad 默认策略。
4. trim 当前依赖 broad 特例。如果后续 trim 被改坏，优先检查 `_prefer_broad_liquid_box(...)` 的阈值。
5. `rod_roi` 和 `sparse_roi` 本轮没有重新设计。后续如果用户继续要求三 ROI 都准确，需要单独重做它们，不要让它们依赖新的 `liquid_roi` 包含关系。

