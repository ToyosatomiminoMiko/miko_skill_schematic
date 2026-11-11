---
name: miko-skill-schematic
description: 用纯 Python+Pillow 画白底线框技术图:电路原理图(逻辑门/数字电路/框图)与神经网络示意图(前向传播、反向传播、梯度)。当用户要"画电路图/原理图/门级图/逻辑门/接线图",或"画神经网络结构/前向传播/反向传播/梯度/模型架构图",或修改已有此类图时使用。
whenToUse: 用户要求生成或修改电路原理图、逻辑门图、数字电路接线图、神经网络结构图、前向/反向传播示意图时;只做线框/矢量风示意图,不做照片级、风景、ASCII 艺术。
---

# 线框示意图:电路原理图 + 神经网络

白底、线框、可直接进讲义/网页的高分辨率 PNG。纯 Python3 + Pillow,无其他依赖。
共享画布底座 + 两个可跑范例 + 风格规范 + 几何自检;新图**照范例改**,不要从零画。

## 快速开始

技能根记为 `S`(全局装:`~/.dsh/skills/miko-skill-schematic`;项目级装:`<项目>/.dsh/skills/miko-skill-schematic`)。

```sh
python3 "$S/scripts/circuit_cla.py"           # 8 位 CLA 电路原理图 -> $S/assets/circuit_cla.png
python3 "$S/scripts/nn_forward_backward.py"   # MLP 前向/反向传播示意图 -> $S/assets/nn_forward_backward.png
python3 "$S/scripts/verify.py" all            # 几何 + 逻辑自检,0 问题才合格
python3 "$S/scripts/verify.py" 我的图.py       # 检查照范例写的新图(需 build() -> Canvas)
python3 "$S/scripts/circuit_cla.py" out.png   # 也可指定输出路径
```

成图参考:`assets/circuit_cla.png`(1950×3120)、`assets/nn_forward_backward.png`。

## 最小模板(新图从这 21 行起步)

```python
import sys
sys.path.insert(0, "你的技能目录/scripts")     # 换成实际路径
from canvas import Canvas, d_and, title_block

def build():
    c = Canvas(1200, 540)
    title_block(c, 44, 50, "最小示例：A 与 B 相与", "照这个骨架换拓扑即可")
    cx, cy, gw, gh = 560, 320, 140, 80
    in_a, in_b = (cx - gw / 2, cy - 14), (cx - gw / 2, cy + 14)
    out = (cx + gw / 2, cy)
    for y, name in ((in_a[1], "A"), (in_b[1], "B")):
        c.wire([(120, y), (cx - gw / 2, y)], (70, 70, 70), net=name)
        c.terminal(120, y)                  # 输入端点登记,自检才不会当成断线
        c.label(112, y, name, (70, 70, 70), 22, "rm")
    c.arrow(out, (980, cy), (12, 92, 196), w=3, head=18)
    c.label(996, cy, "Y = A 与 B", (12, 92, 196), 24, "lm")
    c.gate(d_and(cx, cy, gw, gh))           # 门体最后登记,压住线头
    return c

if __name__ == "__main__":
    build().save("minimal.png")
```

存成 `minimal.py` → `python3 verify.py minimal.py` 应为 0 严重问题。这 21 行只够"一个门 + 输入 + 输出";图再复杂就照对应的完整范例改,别把骨架撑大。

## 工作流

1. **定图种**,读对应规范:
   - 电路/逻辑门 → `references/circuit-style.md`(配色、栅格、门符号、通道布线)
   - 神经网络/传播 → `references/nn-style.md`(两栏对照、节点几何、公式写法)
2. **新建自己的工作脚本**(别改技能里的范例):把 `$S/scripts/` 整个复制进项目(至少含 `canvas.py`),再拿范例当模板改 `build()` 里的拓扑/坐标/文案;或直接在脚本开头 `sys.path.insert(0, "$S/scripts")` 后 `from canvas import ...`。只复制单个范例 `.py` 会因缺 `canvas.py` 报 ModuleNotFoundError。
3. 图元只用 `scripts/canvas.py` 的 `Canvas` API 和公共助手;符号(与/或/异或门、节点)已封装,不要手画。
4. 渲染 → 先跑 `verify.py`(机器检查最便宜)→ `read_image` 目检(大图会被自动缩放,只看整体与文字)→ 迭代,直到 0 问题且肉眼干净。同一张图别反复读,目检 1~2 次即可。
5. 交付 PNG(唯一输出格式,无损;不要再写 PPM)。

## canvas 速查

```python
import sys, os; sys.path.insert(0, "$S/scripts")
from canvas import Canvas, INK, FILL, d_and, or_poly, xor_arc, pin_on_top, pin_on_bot, pin_on_back, or_width

c = Canvas(1600, 1200)              # 参数是最终分辨率像素;内部 SS=2 超采样抗锯齿
c.wire([(x1,y1),(x2,y2)], color, w=3, net="A0")   # 折线;net 仅供自检
c.dashed(pts, color, w=3, on=10, off=7)           # 虚线(反向传播/梯度)
c.arrow(p0, p1, color, w=3, head=16, dash=None, filled=True)  # 实线/虚线箭头,自动登记端子;虚线传 dash=(on,off)
c.poly(pts, fill=FILL, edge=INK, w=3)             # 门体(别名 c.gate);不透明填充压住线头
c.circle(cx, cy, r=24, fill=FILL, edge=INK)       # 神经网络节点
c.rect(x0, y0, x1, y1, fill=None, edge=INK, r=12) # 公式框/图例块;r 省略=直角,r>0 圆角
c.junction(x, y, color, r=4)                      # 三线以上交汇处实心结点
c.terminal(x, y)                                  # 登记端子,自检才不会把图例/注释当断线
c.label(x, y, "文本", color=INK, size=20, anchor="mm", weight="Bold")  # anchor 用 Pillow 语义
c.hat_label(x, y, "y", color=INK, size=20)        # 只画单个 ŷ 时的组字;长字符串里的 ŷ 由 label 自动组字
c.save("out.png")                                 # 坐标即最终像素,SS 自动处理
```

公共助手(别在范例里重写这些):

```python
from canvas import title_block, gate_legend, edge_arrow, node_to_point, point_to_node, layer_ys, text_width, trim, save_asset
title_block(c, 44, 50, "标题", "副标题")        # 左上标题块;副标题可省,自动排在 1.4 倍字号处
gate_legend(c, 792, 132, (("与门","and"),("或门","or"),("异或门","xor")), title="门符号图例")
edge_arrow(c, n0, n1, color, dash=None)        # 节点圆心 -> 节点圆心,贴圆外缘 r+1.5;半径变了传 r=R
node_to_point(c, node, p, color)               # 节点圆心 -> 任意点;point_to_node 反向
layer_ys(n, cy, pitch)                         # 一列 n 个节点居中的 y 坐标列表
text_width("文本", 20, "Bold")                 # 文字宽度:下划线、排版对齐用
trim(p0, p1, r0=0, r1=0)                      # 底层几何:两端各缩进半径+gap,返回贴圆端点
save_asset(build(), "my.png")                  # 写技能 ASSETS_DIR(assets/)并打印;out= 可覆盖
```

绘制顺序固定:线/形按登记顺序 → 结点 → 文字。**线先画、门/节点后画、字最后**,不要自己在 PIL 上调。
**风格只改一处**:纸色/墨色/填充/线宽在 `canvas.STYLE`(键 `paper/ink/fill/sub/wire_w`;导入时绑定,改完重新运行);各图的语义色(`FWD/BWD`、`NET_*`)在各自文件开头,布局坐标不要跟着配色一起动。

## 硬约束与常见坑

- **共线=短路**:同一 y 上两段不同网络的水平线重叠,读图就是短接。通道内用 `assign_lanes(spans)` 分轨道,再逐段画。
- **扇出交点**画 `junction()` 实心圆;没有圆点的交叉线约定为不相连。
- **同网络同色、不同网络异色**(≤7 色);颜色是读图的锚,不要随意换。
- **断线**是最严重错误:每个端点必须接引脚/别的线段/登记端子;`verify.py` 的"悬空端点"必须为 0。
- 中文标签由 Noto Sans CJK 自动挑选;`.ttc` 按 index=2 取简体字面并设字重。公式上标用 `^(l)`。
- **禁用豆腐块字符**(Noto CJK 实测缺字):`ᵀ ⊤ ⟵ ⟶ ˡ ⁻ ⁱ ₀₁₂`。`ŷ` 直接写即可:`Canvas.label` 会自动拆成 y+手画帽子(Noto CJK 无 U+0177),不要自己重写排版。
- 版面:标题左上、图例右上、说明按 30px 行距换行;正文字号 ≥16(次要注释可 15),标题 36~40 Bold。
- 只做示意/线框;不做照片级、风景、ASCII 艺术。

## 目录

| 路径 | 内容 |
| --- | --- |
| `scripts/canvas.py` | 画布底座 + 公共助手:超采样、门符号、节点箭头、标题块、字体、PNG 导出、轨道分配 |
| `scripts/circuit_cla.py` | 电路范例:8 位超前进位加法器门级原理图(布局套路可照搬) |
| `scripts/nn_forward_backward.py` | 神经网络范例:MLP 前向/反向传播双栏对照图 |
| `scripts/verify.py` | 自检:穿门体/节点、共线、断线、文字重叠/出界、CLA 逻辑仿真 |
| `references/circuit-style.md` | 电路图规范:配色、栅格、门尺寸、布线原则、自检清单 |
| `references/nn-style.md` | 神经网络图规范:两栏版式、节点几何、公式书写、图种要点 |
| `assets/` | 两张参考成图,改图时先看它们 |

自检命令:`python3 scripts/verify.py [circuit|nn|all|你的图.py] [--logic]`;退出码非 0 即有严重问题,先修再交付。文字重叠只算警告(不计入退出码),但交付前仍应处理。
