import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import signal
import io

st.set_page_config(page_title="自动控制原理虚拟仿真平台", layout="wide")
st.title("🎛️ 自动控制理论 · 虚拟仿真实验平台")
st.markdown("适用于经典控制理论教学：二阶系统、根轨迹、频域分析与PID控制器设计")

# ---------- 侧边栏：选择实验 ----------
experiment = st.sidebar.selectbox(
    "选择实验模块",
    ["📈 时域分析（阶跃/脉冲/斜坡）",
     "🌿 根轨迹分析",
     "📊 频域分析（Bode & Nyquist）",
     "🔧 PID控制器设计"]
)

# ---------- 通用函数：传递函数生成 ----------
def create_tf(num, den):
    """根据分子/分母系数列表创建传递函数"""
    return ct.tf(num, den)

# ---------- 实验一：时域分析 ----------
if experiment == "📈 时域分析（阶跃/脉冲/斜坡）":
    st.header("时域响应分析")
    col1, col2 = st.columns(2)
    with col1:
        system_type = st.selectbox("系统模型", ["自定义传递函数", "典型二阶系统 G(s)=ωn²/(s²+2ζωn s+ωn²)"])
    if "二阶" in system_type:
        with col1:
            zeta = st.slider("阻尼比 ζ", 0.0, 2.0, 0.5, 0.01)
            wn = st.slider("自然频率 ωn (rad/s)", 0.5, 10.0, 2.0, 0.1)
        num = [wn**2]
        den = [1, 2*zeta*wn, wn**2]
    else:
        with col1:
            num_input = st.text_input("分子系数（空格分隔）", "1")
            den_input = st.text_input("分母系数（空格分隔）", "1 2 1")
        num = [float(x) for x in num_input.split()]
        den = [float(x) for x in den_input.split()]

    sys = create_tf(num, den)

    with col2:
        input_type = st.radio("输入信号", ["阶跃", "脉冲", "斜坡"])
        sim_time = st.slider("仿真时间 (s)", 2.0, 30.0, 10.0, 0.5)

    # 生成响应
    t = np.linspace(0, sim_time, 1000)
    if input_type == "阶跃":
        t, y = ct.step_response(sys, T=t)
        title_str = "阶跃响应"
    elif input_type == "脉冲":
        t, y = ct.impulse_response(sys, T=t)
        title_str = "脉冲响应"
    else:
        # 斜坡响应：使用 lsim
        u = t  # 斜坡输入
        t, y, _ = ct.forced_response(sys, T=t, U=u)
        title_str = "斜坡响应"

    # 绘图
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t, y, linewidth=2)
    ax.set_title(f"{title_str}  {input_type}输入")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("输出")
    ax.grid(True)
    st.pyplot(fig)

    # 时域指标（仅阶跃）
    if input_type == "阶跃":
        info = ct.step_info(sys)
        if info is not None:
            st.subheader("📋 时域性能指标")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("上升时间 (Rise Time)", f"{info.get('RiseTime', np.nan):.3f} s")
            col_b.metric("峰值时间 (Peak Time)", f"{info.get('PeakTime', np.nan):.3f} s")
            col_c.metric("超调量 (Overshoot)", f"{info.get('Overshoot', 0):.2f} %")
            col_d.metric("调节时间 (Settling Time)", f"{info.get('SettlingTime', np.nan):.3f} s")
        else:
            st.warning("无法计算时域指标，可能是无超调或响应未收敛。")

# ---------- 实验二：根轨迹 ----------
elif experiment == "🌿 根轨迹分析":
    st.header("根轨迹分析")
    col1, _ = st.columns(2)
    with col1:
        num_input = st.text_input("开环分子系数（空格分隔）", "1")
        den_input = st.text_input("开环分母系数（空格分隔）", "1 2 2 0")
    num = [float(x) for x in num_input.split()]
    den = [float(x) for x in den_input.split()]
    sys = create_tf(num, den)

    # 绘制根轨迹
    fig, ax = plt.subplots(figsize=(8,6))
    # python-control 新版 root_locus 直接接收 ax
    try:
        ct.root_locus(sys, ax=ax)
    except TypeError:
        # 旧版兼容
        ct.rlocus(sys)
        ax = plt.gca()
    ax.set_title(f"根轨迹图  G(s)H(s)={sys}")
    ax.grid(True)
    st.pyplot(fig)

    # 交互：选择增益查看闭环极点
    st.subheader("🔍 选择增益 K 查看闭环极点")
    K = st.slider("增益 K", 0.1, 50.0, 1.0, 0.1)
    cl_sys = ct.feedback(K * sys, 1)
    poles = ct.poles(cl_sys)
    st.write(f"**K = {K:.2f}** 时闭环极点：")
    for i, p in enumerate(poles):
        st.latex(f"s_{i+1} = {p.real:.3f} + j{p.imag:.3f}")

    # 可选：显示阶跃响应
    if st.checkbox("显示此增益下的闭环阶跃响应"):
        t, y = ct.step_response(cl_sys)
        fig2, ax2 = plt.subplots()
        ax2.plot(t, y)
        ax2.set_title(f"闭环阶跃响应 (K={K})")
        ax2.grid(True)
        st.pyplot(fig2)

# ---------- 实验三：频域分析 ----------
elif experiment == "📊 频域分析（Bode & Nyquist）":
    st.header("频域响应分析")
    col1, _ = st.columns(2)
    with col1:
        num_input = st.text_input("分子系数（空格分隔）", "1")
        den_input = st.text_input("分母系数（空格分隔）", "1 2 1")
    num = [float(x) for x in num_input.split()]
    den = [float(x) for x in den_input.split()]
    sys = create_tf(num, den)

    plot_type = st.radio("选择图类型", ["Bode 图", "Nyquist 图", "两者并排"])

    if "Bode" in plot_type or "两者" in plot_type:
        st.subheader("Bode 图")
        fig_bode, axes = plt.subplots(2,1, figsize=(8,6))
        mag, phase, omega = ct.bode_plot(sys, plot=False)  # 获取数据
        # 幅度 dB
        axes[0].semilogx(omega, 20*np.log10(mag))
        axes[0].set_ylabel('Magnitude (dB)')
        axes[0].grid(True, which='both')
        # 相位 度
        axes[1].semilogx(omega, phase * 180/np.pi)
        axes[1].set_ylabel('Phase (deg)')
        axes[1].set_xlabel('Frequency (rad/s)')
        axes[1].grid(True, which='both')
        st.pyplot(fig_bode)

        # 计算稳定裕度
        gm, pm, wg, wp = ct.margin(sys)
        st.write(f"**增益裕度 GM**: {gm:.3f} (≈{20*np.log10(gm):.2f} dB) @ {wg:.3f} rad/s")
        st.write(f"**相位裕度 PM**: {pm:.3f} deg @ {wp:.3f} rad/s")

    if "Nyquist" in plot_type or "两者" in plot_type:
        st.subheader("Nyquist 图")
        fig_nyq, ax_nyq = plt.subplots(figsize=(6,6))
        ct.nyquist_plot(sys, ax=ax_nyq)
        ax_nyq.set_title("Nyquist Diagram")
        ax_nyq.grid(True)
        st.pyplot(fig_nyq)

# ---------- 实验四：PID控制器设计 ----------
elif experiment == "🔧 PID控制器设计":
    st.header("PID控制器设计与闭环仿真")
    st.markdown("被控对象 $G(s)$ 与 PID 控制器 $C(s)=K_p + K_i/s + K_d s$ 串联")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("被控对象模型")
        num_input = st.text_input("分子系数", "1")
        den_input = st.text_input("分母系数", "1 2 1")
    num_p = [float(x) for x in num_input.split()]
    den_p = [float(x) for x in den_input.split()]
    G = create_tf(num_p, den_p)

    with col2:
        st.subheader("PID参数")
        Kp = st.slider("Kp", 0.0, 20.0, 1.0, 0.1)
        Ki = st.slider("Ki", 0.0, 10.0, 0.0, 0.1)
        Kd = st.slider("Kd", 0.0, 5.0, 0.0, 0.1)
        setpoint = st.number_input("设定值", 0.0, 10.0, 1.0, 0.1)

    # 构建 PID 传递函数 C(s) = Kp + Ki/s + Kd*s
    # 注意：纯微分不可实现，通常加一个低通滤波器，这里简化成理想PID
    # 使用 parrarel 或直接构造
    C_num = [Kd, Kp, Ki]  # 注意顺序：control.tf 期望降幂
    C_den = [1, 0]        # 积分项提供了s在分母，但这里为了简化，实际是 (Kd*s^2 + Kp*s + Ki)/s
    # 正确表示 PID：C(s) = (Kd s^2 + Kp s + Ki) / s
    # 由于 control 库不支持分母阶次高于分子？这里没问题，分子2阶分母1阶。
    C = ct.tf([Kd, Kp, Ki], [1, 0])

    # 闭环系统（单位反馈）
    L = ct.series(C, G)
    T = ct.feedback(L, 1)

    # 仿真
    t = np.linspace(0, 15, 1000)
    t, y = ct.step_response(T * setpoint, T=t)  # 乘以设定值进行缩放

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t, y, label=f'PID (Kp={Kp}, Ki={Ki}, Kd={Kd})')
    ax.axhline(setpoint, color='r', linestyle='--', label='设定值')
    ax.set_title("闭环阶跃响应")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("输出")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 性能
    info = ct.step_info(T)
    if info:
        st.metric("超调量", f"{info.get('Overshoot', 0):.2f} %")
        st.metric("调节时间", f"{info.get('SettlingTime', np.nan):.3f} s")
