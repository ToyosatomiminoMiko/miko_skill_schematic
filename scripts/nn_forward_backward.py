#!/usr/bin/env python3
r"""神经网络“前向传播 / 反向传播”示意图:3-4-4-2 全连接 MLP 的双栏对照。

是什么:
    miko-skill-schematic 技能的“神经网络示意图”范例。左栏画前向传播:蓝色实线箭头从
    输入层经两个隐藏层流向输出层,最后进 loss 框;右栏画反向传播:橙色虚线箭头从
    loss 框沿同一拓扑向左回流,把误差 δ 与参数梯度 ∂L/∂W 一级一级传回输入层。
    底部通栏灰底卡片回顾链式法则的四个基本式子:前向链、误差项、梯度、更新。

为什么这样画(教学版式原理):
    - 两栏共用一套节点坐标:输入 x、激活 a^{(1)} / a^{(2)}、预测 ŷ 在两栏里 y 完全一致,
      读者可以横着对照“同一层在前向里算什么、在反向里回传什么”,这是双栏对照图的
      核心。层头公式与底部回顾带统一走 mathtext 真 TeX(^{(l)}、^\top、\sigma、\odot、
      \partial、\hat{y}、\leftarrow),不再把 ^(l) 当普通字符写。
    - 方向不能只靠颜色:颜色 + 线型(实线/虚线)+ 箭头朝向三重区分,黑白打印或色弱
      读者也能读出“数据向前、梯度向后”。
    - 3-4-4-2 的全连接边是 3×4+4×4+4×2=36 条/栏。先登记边、后登记节点圆(canvas
      的固定绘制顺序),圆的不透明填充会压住伸进圆里的线头,节点边缘才干净;箭头
      端点落在圆外缘(r+1.5),不插进圆里。
    - 每层上方只放两行公式(size 17):列距 340,mathtext 排出来最宽一条约 137px,
      两行以内不会与邻列打架;再长就必须拆行,而不是把字号减到 16px 以下(会糊)。

坑(实测,写新图时注意):
    - 公式里的 ŷ 一律写 mathtext 的 \hat{y},帽子由 mathtext 自己排,不受 CJK 字体
      缺字影响。Noto Sans CJK 的全部字面都没有 U+0177(ŷ),裸字体渲染会变豆腐块;
      Canvas.label 的组字(拆成 y + hat_label,帽子按墨迹 bbox 定位)是给纯文字图
      兜底的,本图公式行不再依赖它。帽子用 stroke 登记,自检不会误报“悬空端点”。
    - 中文只能留在 label / rich_line 的 "t" 段:mathtext 不支持 CJK,写成 \text{中文}
      会渲染成豆腐块。公式里的多字母函数名用 \mathrm{softmax} / \mathrm{CE} 排正体,
      单字母变量保持斜体。
    - mathtext 不支持 \begin{aligned}、\displaystyle、\stackrel 与矩阵环境;本图只
      用单行公式,分式优先斜杠式 \partial L/\partial W(17~18px 正文里 \frac 会挤)。
    - label 文本仍受 CJK 字体限制:η δ × 等已逐字校验可用,ᵀ ⊤ ⟵ ⟶ ˡ ⁻ ⁱ ₀₁₂ 缺字
      禁止使用;公式里的符号交给 mathtext(^\top、\leftarrow、\odot 等)。
    - 画布 2900×1800、SS=2 超采样:所有坐标都是最终像素,不要再自己乘 2。

元信息:作者 deepseek-harness,2026-11-11(2026-11-12 公式换 mathtext)。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from canvas import (
    Canvas, FILL, INK, formula_width, rich_width, text_baseline, title_block,
    save_asset, edge_arrow, node_to_point, point_to_node, layer_ys, text_width,
)

# ---------------------------------------------------------------- 画布与配色

W_IMG, H_IMG = 2900, 1800

FWD = (31, 111, 205)        # 前向 / 预测:蓝(与 references/nn-style.md 一致)
BWD = (198, 80, 38)         # 反向 / 梯度:橙
FORMULA = (74, 81, 94)      # 层公式:深灰,比墨色浅一档
NOTE = (82, 88, 100)        # 栏内说明
MUTED = (118, 124, 136)     # 维度、约定:弱化信息
LINE_LIGHT = (208, 214, 224)  # 分隔线、卡片描边

IN_FILL = (240, 243, 248)   # 输入节点:灰
OUT_FILL = (226, 244, 233)  # 输出节点:淡绿
LOSS_FILL = (253, 236, 230)  # loss 框:淡橙红
BAND_FILL = (247, 249, 252)  # 底部公式带:极浅灰

# ---------------------------------------------------------------- 版图参数

CY = 850                    # 节点层中心 y(两栏完全相同,便于横向对照)
PITCH = 90                  # 层内节点间距
R = 24                      # 节点半径
LAYER_DX = (140, 480, 820, 1160)  # 4 个层列相对栏左边缘的 x 偏移
LOSS_DX = 1300              # loss 框中心相对栏左边缘的 x 偏移
LOSS_W, LOSS_H = 130, 100
PANEL_X = (60, 1460)        # 两栏左边缘

# ---------------------------------------------------------------- 公式条目
# 图的公式分三种行:纯公式、纯文本、中文与公式混排。统一成 (kind, 内容) 条目,
# 渲染函数只写一份“居中对齐 + 基线对齐”的逻辑,三种行混在同一条版式里不会各偏各的。
#   _F(tex)  纯公式,tex 必须带 $...$,交给 Canvas.formula(mathtext);
#   _T(s)    纯中文/文本,交给 Canvas.label(中文进 mathtext 会变豆腐块);
#   _M(...)  中文 + 公式混排,交给 Canvas.rich_line(按数学基线对齐)。


def _F(tex):
    """一条纯公式条目。"""
    return ("f", tex)


def _T(s):
    """一条纯文本条目。"""
    return ("t", s)


def _M(*parts):
    """一条中文 + 公式混排条目;parts 是 [("t", 文本), ("f", 公式), ...]。"""
    return ("m", list(parts))


def _draw_row(sch, x, y, item, color, size, align="l"):
    """画一行公式条目:纯公式 / 纯文本 / 混排,垂直都对齐到行中线 y。

    align="l" 以 x 为左端(底部回顾带);align="m" 以 x 为水平中线(层头、参数
    标签、loss 框)。纯公式与混排用数学基线定位,纯文本保持 label 的行中线语义。
    """
    kind, val = item
    base = text_baseline(y, size)
    if kind == "f":
        sch.formula(x, base, val, color, size,
                    "lm" if align == "l" else "mm", "baseline")
    elif kind == "t":
        sch.label(x, y, val, color, size, "lm" if align == "l" else "mm")
    elif kind == "m":
        start = x if align == "l" else x - rich_width(val, size) / 2.0
        sch.rich_line(start, base, val, color, size)
    else:
        raise ValueError("公式条目 kind 只能是 f/t/m,收到 %r" % (kind,))


# 每层的节点数、名字、填充色、前向/反向两行公式。
# 公式条目见上面的 _F/_T/_M:前向 z=Wa+b → a=σ(z) → softmax 得 ŷ;反向
# δ^{(3)}=ŷ-y,隐藏层 δ^{(l)}=(W^{(l+1)})^\top δ^{(l+1)} ⊙ σ'(z^{(l)})。
LAYERS = (
    dict(n=3, name="输入层", fill=IN_FILL,
         fwd=(_F(r"$x=(x_1,\,x_2,\,x_3)$"), _T("3 维输入向量")),
         bwd=(_F(r"$\partial L/\partial x=(W^{(1)})^\top\delta^{(1)}$"),
              _T("梯度回传到输入层"))),
    dict(n=4, name="隐藏层 1", fill=FILL,
         fwd=(_F(r"$z^{(1)}=W^{(1)}x+b^{(1)}$"),
              _F(r"$a^{(1)}=\sigma(z^{(1)})$")),
         bwd=(_F(r"$\delta^{(1)}=(W^{(2)})^\top\delta^{(2)}$"),
              _F(r"$\odot\,\sigma'(z^{(1)})$"))),
    dict(n=4, name="隐藏层 2", fill=FILL,
         fwd=(_F(r"$z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}$"),
              _F(r"$a^{(2)}=\sigma(z^{(2)})$")),
         bwd=(_F(r"$\delta^{(2)}=(W^{(3)})^\top\delta^{(3)}$"),
              _F(r"$\odot\,\sigma'(z^{(2)})$"))),
    dict(n=2, name="输出层", fill=OUT_FILL,
         fwd=(_F(r"$z^{(3)}=W^{(3)}a^{(2)}+b^{(3)}$"),
              _F(r"$\hat{y}=\mathrm{softmax}(z^{(3)})$")),
         bwd=(_F(r"$\delta^{(3)}=\hat{y}-y$"),
              _M(("t", "("), ("f", r"$\mathrm{softmax}$"),
                 ("t", "+交叉熵)")))),
)

# 连线束旁的参数标签:矩阵形状按 (n_out × n_in) 标注,和底部公式带对得上。
WEIGHTS_FWD = (
    _F(r"$W^{(1)}\,(4\times3)$"),
    _F(r"$W^{(2)}\,(4\times4)$"),
    _F(r"$W^{(3)}\,(2\times4)$"),
)
WEIGHTS_BWD = (
    _F(r"$\partial L/\partial W^{(1)}\,(4\times3)$"),
    _F(r"$\partial L/\partial W^{(2)}\,(4\times4)$"),
    _F(r"$\partial L/\partial W^{(3)}\,(2\times4)$"),
)

NOTES_FWD = (
    _M(("t", "前向:每层先做仿射变换 "), ("f", r"$z=Wa+b$"),
       ("t", ",再过激活 "), ("f", r"$\sigma$"),
       ("t", ";输出层用 "), ("f", r"$\mathrm{softmax}$"),
       ("t", " 得到预测分布 "), ("f", r"$\hat{y}$"), ("t", "。")),
    _T("蓝色实线箭头 = 激活值(数据)沿层向前流动的方向。"),
    _M(("t", "参数:"), ("f", r"$W^{(1)}\,(4\times3)$"), ("t", "、"),
       ("f", r"$W^{(2)}\,(4\times4)$"), ("t", "、"),
       ("f", r"$W^{(3)}\,(2\times4)$"),
       ("t", " 共 36 个权重,再加 10 个偏置。")),
)
NOTES_BWD = (
    _M(("t", "反向:从标量损失 "), ("f", r"$L$"),
       ("t", " 出发,按链式法则逐层回传误差 "), ("f", r"$\delta$"),
       ("t", ",同时算出各参数梯度 "), ("f", r"$\partial L/\partial W$"),
       ("t", "。")),
    _T("橙色虚线箭头 = 梯度(误差信号)沿层向后回流的方向。"),
    _M(("f", r"$\partial L/\partial W^{(l)}$"), ("t", " 与 "),
       ("f", r"$W^{(l)}$"), ("t", " 同形状,"),
       ("f", r"$\partial L/\partial b^{(l)}$"), ("t", " 与 "),
       ("f", r"$b^{(l)}$"), ("t", " 同形状,可直接交给优化器。")),
)

# 底部“链式法则回顾”卡片:四列一排,按训练一步的顺序 前向 → 误差项 → 梯度 → 更新,
# 从左到右读。每列 = 关键词 + 公式(逐条一行,避免大字号挤成一行看不清)+ 一句解释。
# 为什么用四列而不是四行:四条公式单独成行只占不到 1/3 画布宽,通栏卡片会显得左重右空;
# 四列平分整条带,顺序与阅读方向一致,还能给每条公式配一行解释。
CHAIN_COLUMNS = (
    (150, "前向:", FWD,
     (_F(r"$z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$"),
      _F(r"$a^{(l)}=\sigma(z^{(l)})$"),
      _M(("f", r"$L=$"), ("t", "交叉熵("),
         ("f", r"$\hat{y},\,y$"), ("t", ")"))),
     "把输入逐层变成激活值,最后算出标量损失 L"),
    (820, "误差项:", BWD,
     (_F(r"$\delta^{(L)}=\partial L/\partial z^{(L)}$"),
      _F(r"$\delta^{(l)}=(W^{(l+1)})^\top\delta^{(l+1)}$"),
      _F(r"$\odot\,\sigma'(z^{(l)})$")),
     "从输出层向输入层递推,本层 δ 来自下一层"),
    (1490, "梯度:", BWD,
     (_F(r"$\partial L/\partial W^{(l)}=\delta^{(l)}(a^{(l-1)})^\top$"),
      _F(r"$\partial L/\partial b^{(l)}=\delta^{(l)}$"),
      ""),
     "权重梯度 = 本层 δ × 上一层激活的转置"),
    (2160, "更新:", BWD,
     (_F(r"$W^{(l)}\leftarrow W^{(l)}-\eta\,\partial L/\partial W^{(l)}$"),
      _F(r"$b^{(l)}\leftarrow b^{(l)}-\eta\,\partial L/\partial b^{(l)}$"),
      ""),
     "沿负梯度方向走一步,η 控制步长"),
)


# ---------------------------------------------------------------- 图例

def _legend(sch):
    """右上角图例:只放一次,不占两栏的地方。"""
    x0, y0, x1, y1 = 2136, 72, 2834, 248
    sch.rect(x0, y0, x1, y1, (249, 251, 254), LINE_LIGHT, 2, r=16)
    ax0, ax1 = 2168, 2248
    sch.arrow((ax0, 118), (ax1, 118), FWD, w=3, head=12)
    sch.label(2262, 118, "前向传播 / 预测", FWD, 20, "lm")
    # 反向样本的箭头也要朝左,与图里真实的梯度方向一致(否则读者会读反)。
    sch.arrow((ax1, 174), (ax0, 174), BWD, w=3, head=12, dash=(9, 6))
    sch.label(2262, 174, "反向传播 / 梯度", BWD, 20, "lm")
    sch.label(2168, 218, "实线 = 数据向前流动,虚线 = 梯度向后回传", MUTED, 16, "lm")


# ---------------------------------------------------------------- 单栏

def _panel(sch, x0, forward):
    """画一栏。两栏拓扑相同,只有箭头方向、线型、公式与说明不同。"""
    color = FWD if forward else BWD
    dash = None if forward else (9, 6)
    xs = [x0 + dx for dx in LAYER_DX]
    ys = [layer_ys(layer["n"], CY, PITCH) for layer in LAYERS]

    # 1) 全连接边:36 条/栏。先登记线、后登记节点圆(圆会盖住线头)。
    for li in range(len(xs) - 1):
        for ay in ys[li]:
            for by in ys[li + 1]:
                if forward:
                    edge_arrow(sch, (xs[li], ay), (xs[li + 1], by), color, dash)
                else:
                    edge_arrow(sch, (xs[li + 1], by), (xs[li], ay), color, dash)

    # 2) 输入侧:左栏数据从左边进入;右栏梯度向左边流出(∂L/∂x)。
    for ay in ys[0]:
        if forward:
            edge_arrow(sch, (x0 + 64, ay), (xs[0], ay), color, dash, head=12)
        else:
            edge_arrow(sch, (xs[0], ay), (x0 + 64, ay), color, dash, head=12)

    # 3) loss 框 + 输出层与 loss 之间的箭头
    lx = x0 + LOSS_DX
    box_l = lx - LOSS_W / 2.0
    sch.rect(box_l, CY - LOSS_H / 2.0, lx + LOSS_W / 2.0, CY + LOSS_H / 2.0,
             LOSS_FILL, INK, 3, r=14)
    targets = ((CY - 26, ys[3][0]), (CY + 26, ys[3][1]))   # (loss 侧 y, 输出节点 y)
    if forward:                     # 左栏:输出层 ŷ -> loss
        for ty, oy in targets:
            node_to_point(sch, (xs[3], oy), (box_l - 2, ty), color, dash)
    else:                           # 右栏:loss -> 输出层(∂L/∂ŷ 回流)
        for ty, oy in targets:
            point_to_node(sch, (box_l - 2, ty), (xs[3], oy), color, dash)

    # 4) 节点圆:盖住线头
    for li, layer in enumerate(LAYERS):
        for y in ys[li]:
            sch.circle(xs[li], y, R, layer["fill"], INK, w=3)

    # 5) 栏头 + 一句话点题 + 每层“层名 + 两行公式”
    head = "① 前向传播 forward pass" if forward else "② 反向传播 backward pass"
    hx, hy = x0 + 40, 298
    sch.label(hx, hy, head, color, 28, "lm", "Bold")
    ux1 = hx + text_width(head, 28, "Bold")
    sch.wire([(hx, hy + 25), (ux1, hy + 25)], color, w=4)
    sch.terminal(hx, hy + 25)
    sch.terminal(ux1, hy + 25)
    tag = ("数据流:输入 → 隐藏层 1 → 隐藏层 2 → 输出 → 损失"
           if forward else
           "梯度流:损失 → 输出 → 隐藏层 2 → 隐藏层 1 → 输入")
    sch.label(hx, 356, tag, NOTE, 18, anchor="lm")
    for li, layer in enumerate(LAYERS):
        sch.label(xs[li], 460, layer["name"], INK, 22, "mm", "Bold")
        for j, item in enumerate(layer["fwd" if forward else "bwd"]):
            _draw_row(sch, xs[li], 496 + j * 34, item, FORMULA, 17, align="m")

    # 6) 连线束旁的参数标签:束顶在 y≈715,标签放上方 29px,不压线。
    for k, item in enumerate(WEIGHTS_FWD if forward else WEIGHTS_BWD):
        _draw_row(sch, (xs[k] + xs[k + 1]) / 2.0, 686, item, color, 18, align="m")

    # 7) 层维度 + 栏内说明
    for li, layer in enumerate(LAYERS):
        sch.label(xs[li], 1050, "%d 维" % layer["n"], MUTED, 16)
    for j, item in enumerate(NOTES_FWD if forward else NOTES_BWD):
        _draw_row(sch, x0 + 40, 1112 + j * 42, item, NOTE, 19, align="l")

    # 8) loss 框里的文字(先画框后写字,文字永远在最上层)
    sch.label(lx, CY - 24, "损失函数", INK, 19, "mm", "Bold")
    _draw_row(sch, lx, CY + 6, _F(r"$L(\hat{y},\,y)$"), INK, 18, align="m")
    sch.label(lx, CY + 31, "交叉熵", MUTED, 15)

    # 9) 输出侧的小注:左栏写 ŷ,右栏写 ∂L/∂ŷ,说明差异在哪
    if forward:
        _draw_row(sch, 1272, 764, _F(r"$\hat{y}$"), color, 17, align="m")
    else:
        _draw_row(sch, 2668, 764, _F(r"$\partial L/\partial\hat{y}$"), color, 17,
                  align="m")


# ---------------------------------------------------------------- 底部公式带

def _chain_rule_band(sch):
    """底部通栏卡片:链式法则四式,四列一排,从左到右就是一次训练步的顺序。"""
    sch.rect(60, 1400, 2840, 1745, BAND_FILL, LINE_LIGHT, 2, r=18)
    sch.label(100, 1444, "链式法则回顾", INK, 26, "lm", "Bold")
    convention = _M(("t", "约定:"), ("f", r"$a^{(0)}=x$"), ("t", ","),
                    ("f", r"$\delta^{(l)}=\partial L/\partial z^{(l)}$"),
                    ("t", ","), ("f", r"$\odot$"), ("t", " 表示逐元素相乘"))
    _draw_row(sch, 2800 - rich_width(convention[1], 18), 1444,
              convention, MUTED, 18, align="l")
    sch.wire([(100, 1476), (2800, 1476)], (216, 221, 230), w=2)
    sch.terminal(100, 1476)
    sch.terminal(2800, 1476)
    # 列间细分隔线:让“四列”一眼可辨,又比全框格更轻;两端登记成 terminal
    for sx in (785, 1455, 2125):
        sch.wire([(sx, 1496), (sx, 1672)], (230, 233, 240), w=2)
        sch.terminal(sx, 1496)
        sch.terminal(sx, 1672)
    for cx, key, kc, formulas, note in CHAIN_COLUMNS:
        sch.label(cx, 1516, key, kc, 21, "lm", "Bold")
        for j, item in enumerate(formulas):
            if item:
                _draw_row(sch, cx, 1554 + j * 32, item, INK, 21, align="l")
        sch.label(cx, 1666, note, MUTED, 17, anchor="lm")


# ---------------------------------------------------------------- 主体

def build():
    """登记全部图元,返回 Canvas(供 scripts/verify.py 做几何自检)。"""
    sch = Canvas(W_IMG, H_IMG)

    # 标题 / 副标题(左上)
    title_block(sch, 60, 64, "神经网络的前向传播与反向传播",
                "3-4-4-2 全连接 MLP:蓝色实线 = 前向传播,"
                "橙色虚线 = 反向传播(梯度)", size=40)

    # 两栏之间的分隔虚线:只起视觉分区作用,不参与连线
    sch.dashed([(1450, 258), (1450, 1345)], LINE_LIGHT, w=2, on=10, off=8)

    _legend(sch)
    _panel(sch, PANEL_X[0], True)
    _panel(sch, PANEL_X[1], False)
    _chain_rule_band(sch)
    return sch


def main(out=None):
    """渲染并写出 PNG。out 省略时写到技能目录的 assets/ 下。"""
    return save_asset(build(), "nn_forward_backward.png", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
