#!/usr/bin/env python3
"""几何自检:线穿门体/节点、不同网络共线、悬空端点、文字重叠/出界;电路图另加逻辑仿真。

是什么:
    读取 canvas.Canvas 上登记的全部图元,检查"画错了但肉眼在密集区域看不出来"的几类错误:
      1) 导线(含虚线段)从门体/节点内部穿过 => 视觉上等于接到了不该接的引脚;
      2) 两条不同网络的线共线重叠(同 y 或同 x 且区间重叠)=> 看起来就是短接;
      3) 悬空端点:端点既不接别的线、也不贴门体、也不是登记过的端子 => 断线;
      4) 文字互相重叠、文字超出画布边界 => 标注被压住或截断;
      5) (仅电路图)按图上的拓扑仿真全部输入,确认画出来的电路逻辑正确。

为什么:
    原理图/示意图一旦密集,靠眼睛很难发现"差几像素的短接"或"少一截的连线";
    把错误变成可判定的几何关系,在出图前跑一遍,比反复放大看图可靠得多。
    文字重叠用 Pillow 的 textbbox 实测字宽(不是估字符数),中西文混排也准。
    已知折衷:穿体判定按 2px(最终)采样、命中 >=2 点才算,极浅(<~2px)擦边可能漏;
    虚线只查穿体、不查断线(虚线本来就一段一段,间隙不是错误)。

用法:
    python3 verify.py circuit            # 自检电路图范例 circuit_cla.py(含逻辑仿真)
    python3 verify.py nn                 # 自检神经网络示意图 nn_forward_backward.py
    python3 verify.py all                # 两个都查
    python3 verify.py path/to/myfig.py   # 自检照范例写的任意图(模块需有 build() -> Canvas)
    python3 verify.py path/to/myfig.py --logic   # 电路图可额外做逻辑仿真
    退出码:0 = 未发现严重问题;1 = 有严重问题(穿体/共线/断线/逻辑不符/文字出界);
           2 = 用法错误。文字重叠只是警告,不计入退出码。

元信息:作者 deepseek-harness,2026-11-11;几何判据沿用旧 check_cla.py。
"""

import importlib
import importlib.util
import os
import sys

from PIL import Image, ImageDraw

from canvas import SS

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ---------------------------------------------------------------- 取图元模型


class Model(object):
    """把 Canvas 上登记的图元拆成便于几何检查的列表。坐标都是 SS 倍超采样坐标。"""

    def __init__(self, canvas):
        self.canvas = canvas
        self.w, self.h = canvas.w, canvas.h
        self.wires = []      # (pts, color, w, role, net)
        self.polys = []      # 门体多边形
        self.strokes = []    # 开曲线描边(异或门第二道背弧),也是门体的一部分
        self.circles = []    # 神经网络节点 (cx, cy, r)
        for op in canvas._ops:
            kind = op[0]
            if kind == "line":
                if op[4] == "stroke":
                    self.strokes.append(op[1])
                else:
                    self.wires.append((op[1], op[2], op[3], op[4], op[5]))
            elif kind == "poly":
                self.polys.append(op[1])
            elif kind == "ellipse":
                (x0, y0), (x1, y1) = (op[1][0], op[1][1]), (op[1][2], op[1][3])
                self.circles.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0))
        self.terminals = canvas.terminals
        self.labels = canvas._labels


# ---------------------------------------------------------------- 基础判定


def point_in_poly(pt, poly):
    """射线法判断点是否在多边形内。"""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xin = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < xin:
                inside = not inside
    return inside


def on_segment(p, a, b, tol=1.5):
    """点 p 是否落在线段 ab 上(容差 tol)。"""
    px, py = p
    ax, ay = a
    bx, by = b
    if abs(ax - bx) < 1:
        return abs(px - ax) <= tol and min(ay, by) - tol <= py <= max(ay, by) + tol
    if abs(ay - by) < 1:
        return abs(py - ay) <= tol and min(ax, bx) - tol <= px <= max(ax, bx) + tol
    # 斜线:投影参数在 [0,1] 且点到线距离小于 tol
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 == 0:
        return abs(px - ax) <= tol and abs(py - ay) <= tol
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    qx, qy = ax + t * vx, ay + t * vy
    return ((qx - px) ** 2 + (qy - py) ** 2) ** 0.5 <= tol


def near_body(model, p, tol=6.0):
    """点是否落在门体/节点内部或贴边(tol 为贴边容差)。"""
    for poly in model.polys:
        if point_in_poly(p, poly):
            return True
        for i in range(len(poly)):
            if on_segment(p, poly[i], poly[(i + 1) % len(poly)], tol):
                return True
    for pts in model.strokes:
        for a, b in zip(pts, pts[1:]):
            if on_segment(p, a, b, tol):
                return True
    for cx, cy, r in model.circles:
        if ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 <= r + tol:
            return True
    return False


# ---------------------------------------------------------------- 各项检查


def check_wire_through_body(model):
    """导线穿过门体/节点内部。端点贴在门体边上不算穿(原理图引脚本来就在边上)。"""
    bad = {}
    for wid, (pts, color, w, role, net) in enumerate(model.wires):
        if role not in ("wire", "dash"):   # 虚线段也要查穿体,只是不参与"断线"判定
            continue
        hits = 0
        for a, b in zip(pts, pts[1:]):
            L = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
            if L < 8:                       # 太短的段不采样
                continue
            n = max(2, int(L / 4))          # 每 4 SS px(最终 2px)一个采样点,短虚线段也能查
            for j in range(n + 1):
                t = j / float(n)
                if t * L < 4 or (1 - t) * L < 4:   # 端点 2px(最终)内是引脚贴边,不算穿
                    continue
                p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                inside = False
                for poly in model.polys:
                    if point_in_poly(p, poly):
                        inside = True
                        break
                if not inside:
                    for cx, cy, r in model.circles:
                        if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= (r - 1) ** 2:
                            inside = True
                            break
                if inside:
                    hits += 1
        if hits >= 2:                       # 单个采样点可能是擦边;>=2 点才算真的穿过去
            bad[wid] = hits
    return bad


def check_collinear_overlap(model):
    """不同网络的线共线重叠。交叉(垂直穿过)不算错:没有结点就是不相连。"""
    segs = []
    for wid, (pts, color, w, role, net) in enumerate(model.wires):
        for a, b in zip(pts, pts[1:]):
            segs.append((wid, a, b, net))
    found = []
    for i in range(len(segs)):
        p1, a1, b1, n1 = segs[i]
        for j in range(i + 1, len(segs)):
            p2, a2, b2, n2 = segs[j]
            if p1 == p2 or (n1 is not None and n1 == n2):
                continue
            h1 = abs(a1[1] - b1[1]) < 1
            h2 = abs(a2[1] - b2[1]) < 1
            v1 = abs(a1[0] - b1[0]) < 1
            v2 = abs(a2[0] - b2[0]) < 1
            if h1 and h2 and abs(a1[1] - a2[1]) < 3:
                lo = max(min(a1[0], b1[0]), min(a2[0], b2[0]))
                hi = min(max(a1[0], b1[0]), max(a2[0], b2[0]))
                if hi - lo > 10:
                    found.append(("H", a1[1], lo, hi, n1, n2))
            if v1 and v2 and abs(a1[0] - a2[0]) < 3:
                lo = max(min(a1[1], b1[1]), min(a2[1], b2[1]))
                hi = min(max(a1[1], b1[1]), max(a2[1], b2[1]))
                if hi - lo > 10:
                    found.append(("V", a1[0], lo, hi, n1, n2))
    return found


def check_dangling(model):
    """悬空端点:既不接别的线、不贴门体/节点、也不是登记过的端子 => 断线。"""
    bad = []
    for wid, (pts, color, w, role, net) in enumerate(model.wires):
        if role != "wire":
            continue
        for end, pt in (("头", pts[0]), ("尾", pts[-1])):
            if any(abs(pt[0] - tx) <= 3 and abs(pt[1] - ty) <= 3 for tx, ty in model.terminals):
                continue
            touched = False
            for qid, (qpts, _, _, qrole, _) in enumerate(model.wires):
                if qid == wid or qrole != "wire":
                    continue
                if any(on_segment(pt, a, b, 1.5) for a, b in zip(qpts, qpts[1:])):
                    touched = True
                    break
            if touched or near_body(model, pt):
                continue
            bad.append((wid, end, net, round(pt[0] / 2.0), round(pt[1] / 2.0)))
    return bad


def check_labels(model):
    """文字重叠 / 文字超出画布。返回 (重叠对, 出界的标签)。"""
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    boxes = []
    for x, y, s, color, size, anchor, weight in model.labels:
        from canvas import font
        f = font(size, weight)
        try:
            box = probe.textbbox((x * SS, y * SS), s, font=f, anchor=anchor)
        except Exception:
            continue
        boxes.append((s, box))
    W, H = model.w * SS, model.h * SS
    overlaps = []
    for i in range(len(boxes)):
        s1, b1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            s2, b2 = boxes[j]
            ox = min(b1[2], b2[2]) - max(b1[0], b2[0])
            oy = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if ox > 2 and oy > 2:
                overlaps.append((s1, s2, ox, oy))
    out = []
    for s, (x0, y0, x1, y1) in boxes:
        if x0 < -2 or y0 < -2 or x1 > W + 2 or y1 > H + 2:
            out.append((s, (x0, y0, x1, y1), (W, H)))
    return overlaps, out


# ---------------------------------------------------------------- 电路逻辑自检


def logic_check():
    """按图上的递推结构仿真全量输入,确认画出来的确实是 8 位加法器。

    结构(与图一一对应):
      P_i = A_i xor B_i ; G_i = A_i and B_i
      T(i,k) = P_i and (T(i-1,k-1) 若 k>=2 否则 G_{i-1}/C0)
      C(i+1) = G_i or T(i,1) or ... or T(i,i+1)
      S_i = P_i xor C_i
    """
    bad = 0
    for A in range(256):
        for B in range(256):
            for cin in (0, 1):
                P = [((A >> i) & 1) ^ ((B >> i) & 1) for i in range(8)]
                G = [((A >> i) & 1) & ((B >> i) & 1) for i in range(8)]
                C = [cin]
                prev = {0: cin}          # 第 -1 行的 T(-1,0) 就是进位输入 C0
                for i in range(8):
                    cur = {0: G[i]}      # T(i,0) = G_i
                    acc = G[i]
                    for k in range(1, i + 2):
                        t = P[i] & prev[k - 1]   # T(i,k) = P_i AND T(i-1,k-1)
                        cur[k] = t
                        acc |= t
                    C.append(acc)
                    prev = cur
                S = sum(((P[i] ^ C[i]) & 1) << i for i in range(8))
                if S + (C[8] << 8) != A + B + cin:
                    bad += 1
    return bad


# ---------------------------------------------------------------- 入口


def load_module(which):
    """加载被检绘图模块:内置名(circuit_cla / nn_forward_backward)或 .py 路径。

    路径模式下把模块所在目录加入 sys.path,让脚本能 import 同目录的 canvas.py。
    """
    if which.endswith(".py") or os.sep in which:
        path = os.path.abspath(which)
        spec = importlib.util.spec_from_file_location("diagram_under_test", path)
        if spec is None or spec.loader is None:
            raise ImportError("无法加载 %r" % path)
        mod = importlib.util.module_from_spec(spec)
        mod_dir = os.path.dirname(path)
        if mod_dir not in sys.path:
            sys.path.insert(0, mod_dir)
        spec.loader.exec_module(mod)
        return mod
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    return importlib.import_module(which)


def verify_module(which, check_logic=False):
    """对绘图模块的 build() 结果做全套检查,打印报告并返回 (严重问题数, 警告数)。"""
    mod = load_module(which)
    canvas = mod.build()
    model = Model(canvas)
    errors = 0
    warnings = 0

    print("== 模块 %s: wires=%d polys=%d circles=%d labels=%d =="
          % (which, len(model.wires), len(model.polys), len(model.circles), len(model.labels)))

    through = check_wire_through_body(model)
    print("-- 线穿门体/节点: %s" % (("无" if not through else "%d 处" % len(through))))
    for wid in sorted(through, key=lambda p: -through[p])[:20]:
        pts, color, w, role, net = model.wires[wid]
        print("   wire#%d net=%s hits=%d" % (wid, net, through[wid]))
    errors += len(through)

    overlap = check_collinear_overlap(model)
    print("-- 不同网络共线重叠: %s" % ("无" if not overlap else "%d 处" % len(overlap)))
    for kind, at, lo, hi, n1, n2 in overlap[:20]:
        print("   %s %s=%.0f 区间=[%.0f,%.0f] %s vs %s"
              % (kind, "y" if kind == "H" else "x", at, lo, hi, n1, n2))
    errors += len(overlap)

    dangling = check_dangling(model)
    print("-- 悬空端点(断线): %s" % ("无" if not dangling else "%d 处" % len(dangling)))
    for wid, end, net, x, y in dangling[:20]:
        print("   wire#%d %s net=%s 端点=(%d,%d)" % (wid, end, net, x, y))
    errors += len(dangling)

    overlaps, outside = check_labels(model)
    print("-- 文字重叠(警告,不计入退出码): %s" % ("无" if not overlaps else "%d 对" % len(overlaps)))
    for s1, s2, ox, oy in overlaps[:20]:
        print("   %r 与 %r 重叠 %dx%d" % (s1, s2, ox, oy))
    print("-- 文字超出画布: %s" % ("无" if not outside else "%d 条" % len(outside)))
    for s, box, size in outside[:20]:
        print("   %r bbox=%s 画布=%s" % (s, box, size))
    errors += len(outside)
    warnings += len(overlaps)

    if check_logic:
        bad = logic_check()
        print("-- 逻辑仿真(按 CLA 结构;仅对 8 位加法器类图有意义): 不符 %d 组" % bad)
        errors += bad
    return errors, warnings


def main():
    args = [a for a in sys.argv[1:] if a != "--logic"]
    logic_flag = "--logic" in sys.argv[1:]
    which = args[0] if args else "circuit"
    total, warned = 0, 0
    if which.lower() == "circuit":
        e, w = verify_module("circuit_cla", check_logic=True)
        total, warned = total + e, warned + w
    elif which.lower() == "nn":
        e, w = verify_module("nn_forward_backward", check_logic=False)
        total, warned = total + e, warned + w
    elif which.lower() == "all":
        e, w = verify_module("circuit_cla", check_logic=True)
        total, warned = total + e, warned + w
        e, w = verify_module("nn_forward_backward", check_logic=False)
        total, warned = total + e, warned + w
    elif which.endswith(".py") or os.sep in which:
        try:
            e, w = verify_module(which, check_logic=logic_flag)
        except Exception as exc:
            print("无法加载/检查 %r: %s" % (which, exc))
            return 2
        total, warned = total + e, warned + w
    else:
        print("用法: python3 verify.py [circuit|nn|all|路径/to/你的图.py] [--logic]")
        return 2
    print("\n严重问题合计: %d   警告合计: %d(文字重叠)" % (total, warned))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
