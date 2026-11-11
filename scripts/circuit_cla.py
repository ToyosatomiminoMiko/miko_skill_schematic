#!/usr/bin/env python3
"""8 位超前进位加法器(CLA)门级原理图 —— 电路原理图模板/范例。

是什么:
    本文件是 miko-skill-schematic 技能里"电路原理图"的完整范例:画出 8 位单级
    超前进位加法器的门级线框图,输出 PNG。它同时是写新电路图的模板:
    布局常量 + build() 的写法可以直接照搬,只换拓扑与坐标。

为什么这样画(布局原理,换别的电路也适用):
    CLA 的进位展开式为
        C(i+1) = G_i + P_i*G_{i-1} + P_i*P_{i-1}*G_{i-2} + ... + P_i..P_0*C_0
    其中 P_i = A_i xor B_i(传播),G_i = A_i and B_i(产生)。
    直接按上式画,单项的与门扇入会涨到 9,连线呈乱麻。改用递推
        T(i,k) = P_i * T(i-1,k-1),  T(i,0) = G_i,  T(-1,0) = C_0
    则每一项都退化成"两个输入":上一级的项 + 本级的 P_i。
    于是 36 个乘积项成为一棵三角阵列,门全部是 2 输入与门,
    连线只沿两个方向:项沿对角线级联,P_i 沿行内水平总线扇出。
    这就是本图布线为什么规整的根本原因。

    每行(bit i)从左到右依次是:
      A_i,B_i 输入 -> XOR(得 P_i) / AND(得 G_i) -> OR(汇出 C(i+1)) -> XOR(得 S_i)
    右侧是 T 三角阵列;进位 C_i 由上一行的 OR 输出经行间通道引回本行求和异或门。

可能的坑:
    - 行间通道用区间图贪心染色分配轨道(lane),重叠的水平段必须分到不同 lane,
      否则两条不同网络的线会共线重叠,视觉上等同于短路(canvas.assign_lanes)。
    - 同名文字要错开:C0、G_i 等标签放进通道后要留出与相邻行的间距。

元信息:作者 deepseek-harness,2026-11-11;布局与门符号逻辑沿用 cla8.py。
"""

from canvas import (
    Canvas,
    d_and, or_poly, xor_arc, pin_on_back as back_pin,
    pin_on_top as top_pin, pin_on_bot as bot_pin,
    or_width, assign_lanes,
    title_block, gate_legend, save_asset,
)

# ---------------------------------------------------------------- 语义配色
# 纸色/墨色/门体填充/线宽统一在 canvas.STYLE;这里只放本图各网络的语义色。
# 颜色是读图的锚:同一网络全程同色,不同网络异色,总数不超过 7 种。

NET_P = (12, 92, 196)     # 传播 P 线:蓝
NET_G = (198, 40, 40)     # 产生 G 线:红
NET_C = (0, 128, 78)      # 进位 C 线:绿
NET_S = (108, 40, 158)    # 和 S 线:紫
NET_IN = (70, 70, 70)     # 原始输入 A/B/Cin:深灰
TERM_C = (150, 150, 150)  # 乘积项 T(i,k):中性灰(既非 P 也非 G)

# ---------------------------------------------------------------- 版图参数

Q = 340          # 行距(每个 bit 一行)
Y0 = 600         # 第 0 行中心 y(顶部留出中文标题与门符号图例)
P = 150          # 三角阵列列距
X_G1 = 150       # P/G 生成门中心 x
X_S = 700        # 求和异或门中心 x
X_OR = 520       # 进位或门中心 x
X1 = 830         # 三角阵列第 1 列中心 x
GW, GH = 80, 52  # 小门宽高
XOR_DX = 12      # 异或门第二道背弧的左移量

W_IMG = X1 + 7 * P + 70          # 图像宽度
H_IMG = Y0 + 7 * Q + 140         # 图像高度

# 行内纵向基准
RAIL_DY = -74        # P 总线相对行中心的 y 偏移
GDET_DY = -48        # G 绕行线相对行中心的 y 偏移
CH_TOP_DY = 86       # 行间通道上边界相对行中心的 y 偏移
LANE_PITCH = 13
LANE_PAD = 14
PJOG_X = 198         # P 线从异或门引出的竖直拐点 x
GJOG_X = 214         # G 线引出的竖直拐点 x(与 PJOG_X 拉开,避免贴线)

# 三角阵列门
TW, TH = 80, 58
T_TERM_DX = -54      # 项输入的垂直引线相对列中心的 x 偏移
T_UP_DX = 46         # 输出向上分支的 x 偏移
T_DN_DX = 62         # 输出向下分支的 x 偏移
T_PDROP_DX = -15     # P 总线落到门顶的 x 偏移


def rowy(i):
    return Y0 + i * Q


def colx(k):
    return X1 + (k - 1) * P


# ---------------------------------------------------------------- 主体


def build():
    sch = Canvas(W_IMG, H_IMG)
    # 门体延后登记:渲染顺序按登记顺序,门体(不透明填充)必须画在所有导线之后,
    # 这样才能压住伸进图形里的线头;若边画边登记,后画的导线会盖在门框上。
    gate_queue = []
    stroke_queue = []

    # ---- 标题与公式(全部中文)
    title_block(sch, 40, 46, "8 位超前进位加法器（单级进位前瞻 CLA）", size=36)
    eqs = [
        "P_i = A_i 异或 B_i            G_i = A_i 与 B_i",
        "T(i,k) = P_i 与 T(i-1,k-1)，T(i,0) = G_i，T(-1,0) = C0",
        "C(i+1) = G_i 或 T(i,1) 或 T(i,2) 或 … 或 T(i,i+1)",
        "S_i = P_i 异或 C_i",
    ]
    for j, s in enumerate(eqs):
        sch.label(44, 106 + j * 30, s, (55, 55, 55), size=21, anchor="lm")

    # ---- 逻辑门符号图例(把门形状本身列出来)
    gate_legend(sch, 792, 132,
                (("与门", "and"), ("或门", "or"), ("异或门", "xor")),
                title="逻辑门符号图例")

    # ---- 连线的颜色图例
    legend = [("传播 P_i（A_i 异或 B_i）", NET_P),
              ("产生 G_i（A_i 与 B_i）", NET_G),
              ("进位 C_i", NET_C),
              ("和 S_i", NET_S),
              ("输入 A_i / B_i / C0", NET_IN),
              ("乘积项 T(i,k)", TERM_C)]
    for j, (s, c) in enumerate(legend):
        yy = 92 + j * 28
        sch.wire([(1210, yy), (1260, yy)], c, 4)
        sch.terminal(1210, yy)      # 图例色条是示意线,登记端子避免被自检当断线
        sch.terminal(1260, yy)
        sch.label(1272, yy, s, c, size=20, anchor="lm")

    # ---- 说明
    notes = [
        "灰线＝乘积项 T(i,k)：由三角阵列里的每个与门输出。它向右上进本行或门，汇总出该行进位 C(i+1)；"
        "同时向右下进下一行，作为 T(i+1,k+1) 的一个输入。",
        "灰线是同一个网络的两个扇出分支：上去汇总、下去级联。两条分支共用一段竖直干线，交点画实心结点。",
        "共 68 个门：与门 44（乘积项阵列 36 ＋ 产生 G 8）、或门 8（进位）、异或门 16（传播 P 8 ＋ 求和 S 8）。",
        "宽或门＝同一个或门符号按扇入加宽，多输入引脚沿顶部弧线均匀分布。",
    ]
    for j, s in enumerate(notes):
        sch.label(44, 300 + j * 30, s, (70, 70, 70), size=20, anchor="lm")

    # ---- 每行的门与端口
    rows = []
    for i in range(8):
        Y = rowy(i)
        xp = (X_G1, Y - 34)                       # P 生成异或门
        xg = (X_G1, Y + 34)                       # G 生成与门
        xs = (X_S, Y)                             # 求和异或门
        n_top = i + 1
        wor = or_width(n_top)
        xo = (X_OR, Y)                            # 进位或门

        p_poly = or_poly(xp[0], xp[1], GW, GH)
        # 异或门的输入接在外侧那道弧上,所以引脚整体左移
        p_pin_a = back_pin(xp[0], xp[1], GW, GH, 2.0 / 3, dx=-XOR_DX)
        p_pin_b = back_pin(xp[0], xp[1], GW, GH, 1.0 / 3, dx=-XOR_DX)
        p_out = (X_G1 + GW / 2.0, Y - 34)

        g_poly = d_and(xg[0], xg[1], GW, GH)
        g_pin_a = (X_G1 - GW / 2.0, Y + 34 - 11)   # 与门左边平直,直接取边上的点
        g_pin_b = (X_G1 - GW / 2.0, Y + 34 + 11)
        g_out = (X_G1 + GW / 2.0, Y + 34)

        s_poly = or_poly(xs[0], xs[1], GW, GH)
        s_pin_p = top_pin(xs[0], xs[1], GW, GH, 0.40)
        s_pin_c = top_pin(xs[0], xs[1], GW, GH, 0.70)
        s_out = bot_pin(xs[0], xs[1], GW, GH, 0.50)

        o_poly = or_poly(xo[0], xo[1], wor, GH)
        o_pin_g = back_pin(xo[0], xo[1], wor, GH, 0.5)
        o_pins_t = [top_pin(xo[0], xo[1], wor, GH, (k + 1.0) / (n_top + 1.0))
                    for k in range(n_top)]
        o_out = (X_OR + wor / 2.0, Y)

        gate_queue.extend((p_poly, g_poly, s_poly, o_poly))
        # XOR 门多一道背弧
        stroke_queue.append(xor_arc(xp[0], xp[1], GW, GH, dx=XOR_DX))
        stroke_queue.append(xor_arc(xs[0], xs[1], GW, GH, dx=XOR_DX))

        rows.append(dict(i=i, Y=Y, p_pin=(p_pin_a, p_pin_b), p_out=p_out,
                         g_pin=(g_pin_a, g_pin_b), g_out=g_out,
                         s_pin=(s_pin_p, s_pin_c), s_out=s_out,
                         o_pin_g=o_pin_g, o_pins_t=o_pins_t, o_out=o_out,
                         wor=wor, n_top=n_top))

    # ---- 三角阵列的与门
    tri = {}
    for i in range(8):
        Y = rowy(i)
        for k in range(1, i + 2):
            X = colx(k)
            gate_queue.append(d_and(X, Y, TW, TH))
            tri[(i, k)] = dict(
                p_pin=(X + T_PDROP_DX, Y - TH / 2.0),   # P 从顶部进入
                t_pin=(X - TW / 2.0, Y + 11),           # 项从左边进入
                out=(X + TW / 2.0, Y),
            )

    # ---- 位片内部连线:A_i,B_i -> P/G 门
    for r in rows:
        i, Y = r["i"], r["Y"]
        pa, pb = r["p_pin"]
        ga, gb = r["g_pin"]
        sch.label(58, pa[1] - 12, "A%d" % i, NET_IN, size=20, anchor="rm")
        sch.label(58, pb[1] + 12, "B%d" % i, NET_IN, size=20, anchor="rm")
        sch.wire([(62, pa[1]), (pa[0], pa[1])], NET_IN, net="A%d" % i)
        sch.wire([(62, pb[1]), (pb[0], pb[1])], NET_IN, net="B%d" % i)
        sch.terminal(62, pa[1])
        sch.terminal(62, pb[1])
        # 同一条 A/B 还要喂给下面的与门
        ax, bx = 95, 82
        sch.wire([(ax, pa[1]), (ax, ga[1]), (ga[0], ga[1])], NET_IN, net="A%d" % i)
        sch.junction(ax, pa[1], NET_IN, r=4)
        sch.wire([(bx, pb[1]), (bx, gb[1]), (gb[0], gb[1])], NET_IN, net="B%d" % i)
        sch.junction(bx, pb[1], NET_IN, r=4)

    # ---- P 总线(水平轨)+ 落入门顶的支线
    for r in rows:
        i, Y = r["i"], r["Y"]
        rail_y = Y + RAIL_DY
        last_k = i + 1
        rail_end = colx(last_k) + T_PDROP_DX
        sch.wire([(r["p_out"][0], r["p_out"][1]), (PJOG_X, r["p_out"][1]),
                  (PJOG_X, rail_y), (rail_end, rail_y)], NET_P, net="P%d" % i)
        sch.label(300, rail_y - 14, "P%d" % i, NET_P, size=18, anchor="mm")
        # 落到求和异或门的 P 输入
        sch.wire([(r["s_pin"][0][0], rail_y), (r["s_pin"][0][0], r["s_pin"][0][1])],
                 NET_P, net="P%d" % i)
        sch.junction(r["s_pin"][0][0], rail_y, NET_P, r=4)
        # 落到各三角门的 P 输入
        for k in range(1, i + 2):
            px, py = tri[(i, k)]["p_pin"]
            sch.wire([(px, rail_y), (px, py)], NET_P, net="P%d" % i)
            sch.junction(px, rail_y, NET_P, r=4)

    # ---- G 线:去或门 + 去下一行第 1 列的项输入
    for r in rows:
        i, Y = r["i"], r["Y"]
        gx, gy = r["g_out"]
        det_y = Y + GDET_DY
        g_pin_x = r["o_pin_g"][0]
        # G 的竖直干线:向上绕行去或门,向下穿过本行进通道(去下一行第 1 列)。
        # 关键:下端必须真正接到通道里的 gdn 水平段,否则会留一段悬空短线。
        g_bot = (rowy(i) + CH_TOP_DY) if i < 7 else gy
        sch.wire([(gx, gy), (GJOG_X, gy)], NET_G, net="G%d" % i)
        # 下潜点放在加宽或门的左边界之外,再在行中心水平进入或门背引脚;
        # 固定偏移 -20 在宽或门上会切到门体左上角(背弧向左收),所以按门宽算 x。
        drop_x = X_OR - r["wor"] / 2.0 - 8
        sch.wire([(GJOG_X, g_bot), (GJOG_X, det_y), (drop_x, det_y),
                  (drop_x, r["o_pin_g"][1]), (g_pin_x, r["o_pin_g"][1])],
                 NET_G, net="G%d" % i)
        sch.label(300, det_y + 14, "G%d" % i, NET_G, size=18, anchor="mm")
        sch.junction(GJOG_X, gy, NET_G, r=4)

    # ---- 和输出 S_i
    for r in rows:
        i, Y = r["i"], r["Y"]
        sx, sy = r["s_out"]
        sch.wire([(sx, sy), (sx, Y + 55)], NET_S, net="S%d" % i)
        sch.terminal(sx, Y + 55)
        sch.label(sx, Y + 63, "S%d" % i, NET_S, size=21, anchor="mm")

    # ---- 三角阵列连线
    # 通道 c 位于第 c 行与第 c+1 行之间;通道 -1 在第 0 行之上
    def chan_top(c):
        return (rowy(c) + CH_TOP_DY) if c >= 0 else (rowy(0) - 190)

    segs = {c: [] for c in range(-1, 7)}   # 通道号 -> 该通道内的水平段

    def add(c, kind, i, k, xa, xb):
        segs[c].append(dict(kind=kind, i=i, k=k, xa=xa, xb=xb))

    # 收集通道内的水平段(通道 c 夹在第 c 行与第 c+1 行之间;通道 -1 在第 0 行之上)
    # 1) 上行:T(i,k) -> OR_i 顶部引脚
    for (i, k) in tri:
        c = -1 if i == 0 else i - 1
        add(c, "up", i, k, colx(k) + T_UP_DX, rows[i]["o_pins_t"][k - 1][0])
    # 2) 下行:T(i,k) -> T(i+1,k+1) 项输入
    for (i, k) in tri:
        if i >= 7:
            continue
        add(i, "dn", i, k, colx(k) + T_DN_DX, colx(k + 1) + T_TERM_DX)
    # 3) G_i 下行 -> T(i+1,1) 项输入
    for i in range(7):
        add(i, "gdn", i, 0, GJOG_X, colx(1) + T_TERM_DX)
    # 4) C(i+1) 从 OR_i 出来 -> 下一行求和异或门 C 输入
    for i in range(7):
        add(i, "cdn", i, 0, X_OR + rows[i]["wor"] / 2.0 + 25,
            rows[i + 1]["s_pin"][1][0])
    # 5) C0 在上方通道里的水平段
    add(-1, "cin", -1, 0, 62, tri[(0, 1)]["t_pin"][0])

    # T(i,k) 输出:向上支路(去 OR_i)与向下支路(去下一行 T(i+1,k+1) 的项输入)
    for (i, k), t in tri.items():
        X, Y = colx(k), rowy(i)
        ox, oy = t["out"]
        up_x = X + T_UP_DX
        if i < 7:
            dn_x = X + T_DN_DX
            sch.wire([(ox, oy), (dn_x, oy), (dn_x, Y + CH_TOP_DY)], TERM_C,
                     net="T%d_%d" % (i, k))
        # 向上的支路只画到竖直干线起点;竖直段由通道布线统一画,
        # 否则这里会多出一截伸到 Y+CH_TOP_DY 的悬空短线(看起来就是断线)。
        sch.wire([(ox, oy), (up_x, oy)], TERM_C, net="T%d_%d" % (i, k))
        sch.junction(up_x, oy, TERM_C, r=4)

    for c in sorted(segs):
        spans = [(s["xa"], s["xb"]) for s in segs[c]]
        lanes, n = assign_lanes(spans)
        top = chan_top(c)
        for idx, s in enumerate(segs[c]):
            ly = top + LANE_PAD + lanes[idx] * LANE_PITCH
            key = s["kind"]
            i, k = s["i"], s["k"]
            if key == "up":
                up_x = colx(k) + T_UP_DX
                pinx, piny = rows[i]["o_pins_t"][k - 1]
                # 从与门输出(行中心)直接向上进通道,接上那段水平轨道
                sch.wire([(up_x, rowy(i)), (up_x, ly), (pinx, ly),
                          (pinx, piny)], TERM_C, net="T%d_%d" % (i, k))
            elif key == "dn":
                dn_x = colx(k) + T_DN_DX
                tx = colx(k + 1) + T_TERM_DX
                ty = rowy(i + 1) + 11
                sch.wire([(dn_x, rowy(i) + CH_TOP_DY), (dn_x, ly), (tx, ly),
                          (tx, ty), (colx(k + 1) - TW / 2.0, ty)], TERM_C,
                         net="T%d_%d" % (i, k))
            elif key == "gdn":
                ty = rowy(i + 1) + 11
                tx = colx(1) + T_TERM_DX
                sch.wire([(GJOG_X, rowy(i) + CH_TOP_DY), (GJOG_X, ly), (tx, ly),
                          (tx, ty), (colx(1) - TW / 2.0, ty)], NET_G, net="G%d" % i)
            elif key == "cdn":
                xa = X_OR + rows[i]["wor"] / 2.0 + 25
                pinx, piny = rows[i + 1]["s_pin"][1]
                sch.wire([(xa, rowy(i)), (xa, ly), (pinx, ly), (pinx, piny)], NET_C,
                         net="C%d" % (i + 1))
            elif key == "cin":
                tx = colx(1) + T_TERM_DX
                pinx, piny = tri[(0, 1)]["t_pin"]
                scx, scy = rows[0]["s_pin"][1]
                sch.wire([(62, ly), (tx, ly), (tx, piny), (pinx, piny)], NET_C, net="C0")
                sch.terminal(62, ly)
                sch.wire([(scx, ly), (scx, scy)], NET_C, net="C0")
                sch.junction(scx, ly, NET_C, r=4)
                sch.label(58, ly, "C0", NET_C, size=21, anchor="rm")

    # ---- C(i+1) 从 OR 顶端引出
    for i in range(7):
        xa = X_OR + rows[i]["wor"] / 2.0 + 25
        sch.wire([rows[i]["o_out"], (xa, rowy(i))], NET_C, net="C%d" % (i + 1))
    # C8 输出:向下引出,标签避开 S7
    o7 = rows[7]
    xa = X_OR + o7["wor"] / 2.0 + 25
    sch.wire([o7["o_out"], (xa, rowy(7)), (xa, rowy(7) + 108)], NET_C, net="C8")
    sch.label(58, rowy(7) + 108, "C8（进位输出）", NET_C, size=22, anchor="lm")
    sch.wire([(248, rowy(7) + 108), (xa, rowy(7) + 108)], NET_C, net="C8")
    sch.terminal(248, rowy(7) + 108)
    sch.junction(xa, rowy(7) + 108, NET_C, r=4)

    # ---- 注释:指着三角阵列某个与门的输出,说明那条灰线是什么
    sch.arrow((1250, 560), (886, 604), (176, 98, 20), w=3, head=17, filled=False)
    sch.terminal(1250, 560)
    sch.terminal(886, 604)
    sch.label(1262, 560, "每个与门输出的灰线 ＝ 乘积项 T(i,k)", (176, 98, 20),
              size=22, anchor="lm")

    # 最后登记电路门体与异或门背弧:导线全部画完再压门,接头干净。
    # (顶部图例的 3 个门符号没有导线经过,仍在开头登记,不受影响。)
    for poly in gate_queue:
        sch.gate(poly)
    for arc in stroke_queue:
        sch.stroke(arc)

    return sch


def main(out=None):
    """渲染并写出 PNG。out 省略时写到技能目录的 assets/ 下。"""
    return save_asset(build(), "circuit_cla.png", out)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
