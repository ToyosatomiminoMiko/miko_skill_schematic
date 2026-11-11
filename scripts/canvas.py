#!/usr/bin/env python3
"""示意图画布底座:超采样抗锯齿 + 逻辑门 / 节点 / 箭头 / 虚线 / 中文标注 / PNG 导出。

是什么:
    本文件是 miko-skill-schematic 技能的共享绘图底座。电路原理图与神经网络示意图都
    只往 Canvas 上登记图元,不直接碰 PIL;所有坐标都用"最终分辨率像素",内部统一
    乘 SS 倍绘制,再 LANCZOS 缩小。

为什么这样设计:
    - SS 超采样:Pillow 的线/圆没有抗锯齿,直接按最终分辨率画会有明显锯齿;
      2 倍分辨率绘制再缩小,等于免费的 4 倍样本抗锯齿,线宽也能取整。
    - 绘制顺序 = 登记顺序(线/形混排),结点其次,文字最后;门体要压住线头必须
      自己延后登记(circuit_cla.py 的 gate_queue),canvas 不会自动把门体排到最后。
    - 虚线自己采样实现:Pillow 没有虚线 API;正交折线按累计弧长切段,可以让虚实
      节奏跨拐点连续。反向传播的梯度线必须虚线,否则与正向传播实线混在一起。
    - .ttc 中文字体要按 index 取"简体中文"字面,可变字体还要 set_variation_by_name,
      否则默认 Thin 字重细到看不清;找不到中文字体时退化为默认字体并写方块。
    - 只存 PNG:Pillow 的 PNG 是无损压缩,像素与裸 RGB 完全一致,体积却只有 P6 PPM
      的约 1/30(CLA 图 0.5MB vs 17MB),且浏览器、read_image 都能直接看。手写 PPM
      只是"不借图像库"的教学演示,放进主线只会多一条要维护的存盘分支。

元信息:作者 deepseek-harness,2026-11-11。
"""

import math
import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 风格配置
# 换风格(纸色/墨色/填充/线宽)只改 STYLE 这一段的数值,布局代码不用动。
# STYLE 在导入时绑定到下面的别名,所以必须改源文件后重新运行;运行期改 dict 不生效。
# 各图自己的语义色(网络颜色 FWD/BWD/NET_*)与版图尺寸,放在各图文件开头。

STYLE = dict(
    paper=(255, 255, 255),    # 纸色:白底
    ink=(26, 34, 50),         # 墨色:线、门框、正文,比纯黑柔和
    fill=(233, 240, 253),     # 门体/节点默认填充:极淡蓝
    sub=(96, 103, 116),       # 副标题、弱化文字
    wire_w=3,                 # 默认线宽(最终分辨率像素)
)

SS = 2                        # 超采样倍率:只影响抗锯齿质量,一般不要动
BG = STYLE["paper"]           # 下面的别名是给代码读的,改风格请改上面的 STYLE
INK = STYLE["ink"]
FILL = STYLE["fill"]
SUB = STYLE["sub"]
WIRE_W = STYLE["wire_w"]

# 字体候选:优先中日韩 CJK 字体(要画中文),都没有时退到拉丁字体
_FONT_CANDIDATES = (
    "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
)


@lru_cache(maxsize=1)
def _font_path():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _find_font_file():
    """返回 (字体路径, 是否需要按简体中文字面 index=2 取)。"""
    p = _font_path()
    if p is None:
        return None, False
    # 可变 CJK 字体是 .ttc 集合,index 0 常见是日文,index 2 才是简体中文
    return p, p.endswith(".ttc")


@lru_cache(maxsize=64)
def font(size, weight="Regular"):
    """取尺寸为 size(最终分辨率)的字号,内部乘 SS。weight 只对可变字体生效。"""
    path, is_ttc = _find_font_file()
    px = max(1, int(round(size * SS)))
    if path is None:
        return ImageFont.load_default(size=px)
    if is_ttc:
        f = ImageFont.truetype(path, px, index=2)
        try:
            f.set_variation_by_name(weight)   # 可变字体:Regular / Bold / Medium
        except Exception:
            pass
        return f
    return ImageFont.truetype(path, px)


# ---------------------------------------------------------------- 几何工具

# 量文字墨迹用的探针画布(Pillow 的 textbbox 需要 ImageDraw 实例)
_PROBE = ImageDraw.Draw(Image.new("RGB", (4, 4)))


def bez(p0, p1, p2, t):
    """二次贝塞尔取点,用来画或门/异或门的弧线轮廓。"""
    u = 1.0 - t
    return (u * u * p0[0] + 2 * t * u * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * t * u * p1[1] + t * t * p2[1])


def sample(p0, p1, p2, n=18):
    return [bez(p0, p1, p2, i / float(n)) for i in range(n + 1)]


def dashes(pts, on=10.0, off=7.0):
    """把折线 pts 切成虚线采样点列表(相邻两点一段,只保留实线段)。

    为什么按弧长切:直接按每段独立切会让拐角处的虚实节奏断掉,看起来像两段线;
    按累计长度切能跨拐点保持节奏。
    """
    out = []
    phase = 0.0          # 0..on+off 的累计相位
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = math.hypot(dx, dy)
        if seg <= 0:
            continue
        ux, uy = dx / seg, dy / seg
        t = 0.0
        while t < seg:
            if phase < on:                 # 当前处在实线相位
                end = min(t + (on - phase), seg)
                out.append((a[0] + ux * t, a[1] + uy * t))
                out.append((a[0] + ux * end, a[1] + uy * end))
                phase += end - t
                t = end
            else:                          # 空白相位:只推进,不画
                end = min(t + (on + off - phase), seg)
                phase = (phase + end - t) % (on + off)
                t = end
    return out


def assign_lanes(segs):
    """区间图贪心染色:重叠的水平段分到不同轨道,返回 (每段轨道号, 轨道总数)。

    为什么必须做:同一 y 上两段线属于不同网络时一旦共线,视觉上就是短接;
    给重叠区间分配不同 lane,布线就永远不会在几何上重叠。
    """
    order = sorted(range(len(segs)), key=lambda j: min(segs[j]))
    lanes = []
    out = [0] * len(segs)
    for j in order:
        lo, hi = min(segs[j]), max(segs[j])
        for li, occ in enumerate(lanes):
            if all(not (lo < b and a < hi) for a, b in occ):
                occ.append((lo, hi))
                out[j] = li
                break
        else:
            lanes.append([(lo, hi)])
            out[j] = len(lanes) - 1
    return out, len(lanes)


# ---------------------------------------------------------------- 逻辑门符号
# 门符号统一约定:输入引脚在左,输出引脚在右;或门/异或门是三段二次贝塞尔弧。


def d_and(cx, cy, w=80, h=52):
    """D 形与门轮廓:左边平直(输入侧),右边半圆(输出侧)。"""
    r = h / 2.0
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    pts = [(x0, cy - r), (x1 - r, cy - r)]
    for i in range(1, 25):
        a = -math.pi / 2.0 + math.pi * i / 24.0
        pts.append((x1 - r + r * math.cos(a), cy + r * math.sin(a)))
    pts.append((x0, cy + r))
    return pts


def or_ctrl(cx, cy, w=80, h=52, bow=0.42, tip=0.62):
    """或门/异或门的三条边界曲线控制点。

    上边:(x0,y0)->(x1,cy);下边:(x1,cy)->(x0,y1);后背:(x0,y1)->(x0,y0)。
    后背曲线向右凹(控制点在 x0+bow*w),这正是或门标志性的凹背弧。
    """
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    top = ((x0, y0), (x0 + tip * w, y0), (x1, cy))
    bot = ((x1, cy), (x0 + tip * w, y1), (x0, y1))
    back = ((x0, y1), (x0 + bow * w, cy), (x0, y0))
    return top, bot, back


def or_poly(cx, cy, w=80, h=52, bow=0.42, tip=0.62):
    top, bot, back = or_ctrl(cx, cy, w, h, bow, tip)
    return sample(*top) + sample(*bot)[1:] + sample(*back)[1:]


def xor_arc(cx, cy, w=80, h=52, dx=12, bow=0.42):
    """异或门标志性的第二道背弧:把后背曲线整体左移 dx。

    为什么必须单独画:异或门 = 或门 + 多一道背弧;少了它,图上分不出异或门
    与普通或门,读图的人就会把 S_i = P_i xor C_i 那级看错。
    """
    _, _, back = or_ctrl(cx, cy, w, h, bow=bow)
    return [(x - dx, y) for x, y in sample(*back)]


def pin_on_back(cx, cy, w, h, t, dx=0.0, bow=0.42):
    """取或门/异或门后背弧上的引脚位置(t=0 底,t=1 顶)。"""
    _, _, back = or_ctrl(cx, cy, w, h, bow=bow)
    x, y = bez(*back, t)
    return (x + dx, y)


def pin_on_top(cx, cy, w, h, t, tip=0.62):
    top, _, _ = or_ctrl(cx, cy, w, h, tip=tip)
    return bez(*top, t)


def pin_on_bot(cx, cy, w, h, t, tip=0.62):
    _, bot, _ = or_ctrl(cx, cy, w, h, tip=tip)
    return bez(*bot, t)


def or_width(n_top):
    """或门按扇入数加宽:多输入引脚沿顶部弧线均匀分布。"""
    return 56 + n_top * 18.0


# ---------------------------------------------------------------- 节点连线
# 神经网络/框图类图的通用几何:端点落在圆外缘,箭头不插进节点。


def trim(p0, p1, r0=0.0, r1=0.0, gap=1.5):
    """把 p0->p1 两端各缩进 r+gap,返回贴着圆外缘的两个端点。

    退化处理:两点重合时没有方向,原样返回(否则除零);两圆相交(d < r0+r1+2gap)
    时两端夹到中点,避免端点越过对面、画出反向箭头。
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return p0, p1
    ux, uy = dx / d, dy / d
    a, b = r0 + gap, r1 + gap
    if a + b >= d:
        mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        return mid, mid
    return ((p0[0] + ux * a, p0[1] + uy * a),
            (p1[0] - ux * b, p1[1] - uy * b))


def edge_arrow(sch, src, dst, color, dash=None, r=24, w=2, head=13):
    """节点圆外缘之间的箭头(src/dst 都是圆心)。"""
    p0, p1 = trim(src, dst, r, r)
    sch.arrow(p0, p1, color, w=w, head=head, dash=dash)


def node_to_point(sch, node, p1, color, dash=None, r=24, w=2.5, head=12):
    """从节点圆外缘指向任意点(节点 -> 框/文字)。"""
    p0, _ = trim(node, p1, r, 0.0)
    sch.arrow(p0, p1, color, w=w, head=head, dash=dash)


def point_to_node(sch, p0, node, color, dash=None, r=24, w=2.5, head=12):
    """从任意点指向节点圆外缘(框/文字 -> 节点)。"""
    _, p1 = trim(p0, node, 0.0, r)
    sch.arrow(p0, p1, color, w=w, head=head, dash=dash)


def layer_ys(n, cy, pitch):
    """一列 n 个节点的 y:以 cy 为中线、pitch 为间距,居中排布。"""
    return [cy + (i - (n - 1) / 2.0) * pitch for i in range(n)]


def text_width(s, size, weight="Regular"):
    """文字宽度(最终分辨率像素),用于下划线、排版对齐。"""
    return font(size, weight).getlength(s) / SS


# ---------------------------------------------------------------- 画布


class Canvas(object):
    """图元登记 + 一次渲染。坐标全部是最终分辨率像素,内部乘 SS。"""

    def __init__(self, width, height, bg=BG):
        self.w, self.h = width, height
        self.bg = bg
        self._ops = []       # 线 / 形:按登记顺序画
        self._dots = []      # 结点:画在形之后,避免被门体填充盖住
        self._labels = []    # 文字:永远最后画
        self.terminals = []  # 端子坐标:几何自检用来区分"端子"与"断线"

    # ---- 登记接口

    def wire(self, pts, color, w=WIRE_W, net=None):
        """折线。net 只是给自检用的网络名标签,不参与绘制。"""
        self._ops.append(("line", self._sc(pts), color, int(w * SS), "wire", net))

    def stroke(self, pts, color=INK, w=WIRE_W):
        """不闭合描边,用于异或门第二道背弧这类开曲线。"""
        self._ops.append(("line", self._sc(pts), color, int(w * SS), "stroke", None))

    def dashed(self, pts, color, w=WIRE_W, on=10, off=7, net=None):
        """折线虚线段。on/off 是实线段长度与间隙长度(最终分辨率像素)。"""
        segs = dashes(pts, on, off)
        for a, b in zip(segs[0::2], segs[1::2]):
            self._ops.append(("line", self._sc([a, b]), color, int(w * SS), "dash", net))

    def arrow(self, p0, p1, color, w=WIRE_W, head=16, dash=None, filled=True):
        """箭头:直线(可虚线)+ 箭头。箭头尖端登记为端子。

        filled=True 画实心三角箭头(示意图常用);False 画两撇开口箭头(原理图注释常用)。
        """
        if dash:
            self.dashed([p0, p1], color, w, on=dash[0], off=dash[1])
        else:
            self.wire([p0, p1], color, w)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        if filled:
            half = head * 0.40
            base = (p1[0] - head * math.cos(ang), p1[1] - head * math.sin(ang))
            nx, ny = -math.sin(ang), math.cos(ang)
            tri = [(p1[0], p1[1]),
                   (base[0] + nx * half, base[1] + ny * half),
                   (base[0] - nx * half, base[1] - ny * half)]
            self._ops.append(("triangle", self._sc(tri), color))
        else:
            for d in (math.radians(150), math.radians(-150)):
                tip = (p1[0] + head * math.cos(ang + d), p1[1] + head * math.sin(ang + d))
                self.wire([p1, tip], color, w)
                self.terminals.append((tip[0] * SS, tip[1] * SS))
        self.terminals.append((p0[0] * SS, p0[1] * SS))
        self.terminals.append((p1[0] * SS, p1[1] * SS))

    def poly(self, pts, fill=FILL, edge=INK, w=WIRE_W):
        """闭合多边形:先填充再描边(门体、提示框都用它)。"""
        self._ops.append(("poly", self._sc(pts), fill, edge, int(w * SS)))

    gate = poly   # 电路图里"门"就是填充的多边形,给出别名让语义更直白

    def circle(self, cx, cy, r, fill=FILL, edge=INK, w=WIRE_W):
        """实心圆:神经网络节点。"""
        box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
        self._ops.append(("ellipse", box, fill, edge, int(w * SS)))
        return (cx, cy)

    def rect(self, x0, y0, x1, y1, fill=None, edge=INK, w=WIRE_W, r=0):
        """矩形/圆角矩形:公式框、图例块。"""
        if r > 0:
            self._ops.append(("rrect", self._sc([(x0, y0), (x1, y1)]), fill, edge,
                              int(w * SS), int(r * SS)))
        else:
            self._ops.append(("rect", self._sc([(x0, y0), (x1, y1)]), fill, edge,
                              int(w * SS)))

    def junction(self, x, y, color, r=4):
        """实心结点:表示三条以上导线在此相连。"""
        self._dots.append(((x * SS, y * SS), color, r * SS))

    def terminal(self, x, y):
        """登记一个端子(输入/输出端点),供几何自检区分"端子"与"断线"。"""
        self.terminals.append((x * SS, y * SS))

    def label(self, x, y, s, color=INK, size=20, anchor="mm", weight="Regular"):
        """文字。anchor 用 Pillow 语义:mm 居中、lm 左中、rm 右中。

        字符串里含 ŷ(U+0177)时自动组字:拆成普通片段 + "y+帽子",宽度按字体实测
        advance 计算,所以居中/对齐与普通字符串完全一致。为什么不在调用方处理:
        Noto Sans CJK 全字面都没有 U+0177,每个画 ŷ 的脚本都得重写一遍排版,不如收进这里。
        """
        if "ŷ" not in s:
            self._labels.append((x, y, s, color, size, anchor, weight))
            return
        self._label_hat(x, y, s, color, size, anchor, weight)

    @staticmethod
    def _tw(s, size, weight):
        """文字宽度(最终像素)。font() 返回 SS 倍字号,所以要除回 SS。"""
        return font(size, weight).getlength(s) / SS

    def _label_hat(self, x, y, s, color, size, anchor, weight):
        """把含 ŷ 的字符串拆段排版:普通片段用 label,ŷ 用 hat_label 组字。

        为什么不用调用方自己排:排版规则(逐段 advance、帽子定位)只应有一份实现;
        hat_label 已封装"y + 墨迹 bbox 帽子",这里只负责把宽度和锚点算对。
        """
        hatch_gap = 2.0
        parts = s.split("ŷ")
        adv_y = self._tw("y", size, weight)
        total = (sum(self._tw(p, size, weight) for p in parts)
                 + (adv_y + hatch_gap) * (len(parts) - 1))
        if anchor == "mm":
            x0 = x - total / 2.0
        elif anchor == "rm":
            x0 = x - total
        else:
            x0 = x
        cx = x0
        for i, part in enumerate(parts):
            if part:
                self._labels.append((cx, y, part, color, size, "lm", weight))
                cx += self._tw(part, size, weight)
            if i < len(parts) - 1:
                self.hat_label(cx + adv_y / 2.0, y, "y", color, size, weight)
                cx += adv_y + hatch_gap

    def hat_label(self, x, y, base="y", color=INK, size=20, weight="Regular",
                  lift=None, half_ratio=0.9, thickness=None):
        """组字:写字母 + 在正上方补一个 ^ 帽,拼出 ŷ 这类字体缺字的字符。

        为什么需要:Noto Sans CJK 的全部字面(JP/KR/SC/TC/HK)都没有 U+0177(ŷ),
        直接 label("ŷ") 会渲染成豆腐块。帽子按实际墨迹 bbox 定位,不是拍脑袋的偏移;
        返回帽尖坐标,需要的话可以登记为端子。
        """
        f = font(size, weight)
        if lift is None:                      # 帽底与字母墨迹之间留约 0.3 字号的空隙
            lift = max(4.0, size * 0.30)
        box = _PROBE.textbbox((x * SS, y * SS), base, font=f, anchor="mm")
        cx = (box[0] + box[2]) / 2.0 / SS
        half = (box[2] - box[0]) / 2.0 / SS * half_ratio
        top = box[1] / SS - lift
        w = thickness if thickness is not None else max(2, int(round(size * 0.09)))
        self.label(x, y, base, color, size, "mm", weight)
        self.stroke([(cx - half, top + 3), (cx, top - 3), (cx + half, top + 3)], color, w)
        return (cx, top - 3)

    # ---- 渲染

    @staticmethod
    def _sc(pts):
        return [(x * SS, y * SS) for x, y in pts]

    def render(self):
        img = Image.new("RGB", (self.w * SS, self.h * SS), self.bg)
        d = ImageDraw.Draw(img)
        for op in self._ops:
            kind = op[0]
            if kind == "line":
                pts, color, w = op[1], op[2], op[3]
                d.line(pts, fill=color, width=w, joint="curve")
            elif kind == "poly":
                _, pts, fill, edge, w = op
                if fill is not None:
                    d.polygon(pts, fill=fill)
                d.line(pts + [pts[0]], fill=edge, width=w, joint="curve")
            elif kind == "triangle":
                _, pts, color = op
                d.polygon(pts, fill=color)
            elif kind == "ellipse":
                _, box, fill, edge, w = op
                if fill is not None:
                    d.ellipse(box, fill=fill)
                d.ellipse(box, outline=edge, width=w)
            elif kind == "rect":
                _, pts, fill, edge, w = op
                (x0, y0), (x1, y1) = pts
                if fill is not None:
                    d.rectangle([x0, y0, x1, y1], fill=fill)
                d.rectangle([x0, y0, x1, y1], outline=edge, width=w)
            elif kind == "rrect":
                _, pts, fill, edge, w, rr = op
                (x0, y0), (x1, y1) = pts
                if fill is not None:
                    d.rounded_rectangle([x0, y0, x1, y1], radius=rr, fill=fill)
                d.rounded_rectangle([x0, y0, x1, y1], radius=rr, outline=edge, width=w)
        for (x, y), color, r in self._dots:
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)
        for x, y, s, color, size, anchor, weight in self._labels:
            d.text((x * SS, y * SS), s, font=font(size, weight), fill=color, anchor=anchor)
        return img.resize((self.w, self.h), Image.LANCZOS)

    # ---- 导出

    def save(self, path):
        """渲染并保存为 PNG(唯一输出格式,无损)。返回路径。"""
        img = self.render()
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        img.save(path)
        return path


# ---------------------------------------------------------------- 版式与导出


def title_block(sch, x, y, text, sub=None, color=INK, size=38, weight="Bold",
                sub_color=SUB, sub_size=20):
    """左上角标题 + 可选副标题;副标题在标题下方 1.4 倍字号处,不用另算坐标。"""
    sch.label(x, y, text, color, size, "lm", weight)
    if sub:
        sch.label(x, y + size * 1.4, sub, sub_color, sub_size, "lm")


def gate_legend(sch, x, y, items, title=None, step=122, size=22,
                title_color=(90, 90, 90)):
    """逻辑门符号图例:一行画出与/或/异或门形状,下面写名字。

    items 是 (名字, kind) 列表,kind 取 "and" / "or" / "xor";title 写在整行上方居中。
    为什么收进底座:图例的门尺寸/间距与正文门统一,每个电路图都手写一遍容易走形。
    """
    w, h = 66, 46
    for j, (name, kind) in enumerate(items):
        if kind not in ("and", "or", "xor"):
            raise ValueError("未知门类型 %r:只能是 and/or/xor" % (kind,))
        cx = x + j * step
        if kind == "and":
            sch.gate(d_and(cx, y, w, h))
        else:
            sch.gate(or_poly(cx, y, w, h))
            if kind == "xor":
                sch.stroke(xor_arc(cx, y, w, h, dx=10))
        sch.label(cx, y + 44, name, INK, size, "mm")
    if title:
        sch.label(x + (len(items) - 1) / 2.0 * step, y - 46, title,
                  title_color, 20, "mm")


ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets")


def save_asset(sch, name, out=None):
    """渲染并写出 PNG;out 省略时写到技能目录 assets/<name>。返回路径。"""
    out = out or os.path.join(ASSETS_DIR, name)
    sch.save(out)
    print("wrote", out)
    return out
