import numpy as np
import cv2

# ==========================================
# 1. 核心色彩与噪声工具
# ==========================================
def srgb_to_linear(img):
    if img.dtype == np.uint8 or img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)
    linear = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)

def linear_to_srgb(img):
    img = np.clip(img, 0.0, 1.0)
    srgb = np.where(img <= 0.0031308, img * 12.92, 1.055 * (img ** (1 / 2.4)) - 0.055)
    return np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)

def add_camera_noise(img, noise_level=0.005):
    """模拟真实的偏振传感器噪声 (泊松-高斯混合)"""
    noise = np.random.normal(0, noise_level, img.shape)
    poisson_noise = noise * np.sqrt(np.clip(img, 0.01, 1.0))
    return np.clip(img + poisson_noise, 0, 1)

# ==========================================
# 2. 偏振水下模拟器
# ==========================================
class PolarizedUnderwaterSimulator_v2:
    def __init__(self):
        # 物理边界定义 (BGR 顺序)

        # ==========================================
        # 模式1：海洋模式 (Ocean Mode)
        # 物理特性：吸收严重依赖波长(红光极速衰减)，背景呈蓝/绿色
        # ==========================================
        # 极端清澈 (Oceanic I): 红光被水剧烈吸收, 蓝绿光透射
        self.ocean_clear_beta  = np.array([0.05, 0.05, 0.08]) # 吸收红光，透射蓝绿光
        self.ocean_clear_Binf  = np.array([0.05, 0.05, 0.15]) # 纯净深蓝色背景
        # 极限浑浊 (模拟 UPBD 的牛奶+叶绿素): 蓝红光剧烈衰减, 绿光主导
        self.ocean_turbid_beta = np.array([1.50, 1.20, 1.60]) # 模拟富营养化(叶绿素)
        self.ocean_turbid_Binf = np.array([0.20, 0.40, 0.10]) # 黄绿色背景

        # ==========================================
        # 模式2：水箱模式 (Tank Mode - UPBD等)
        # 物理特性：通常使用自来水+牛奶，Mie散射主导(各波段衰减均匀)
        # 背景光受限于水箱壁(通常是黑色或暗灰色)和人工白色LED光源
        # ==========================================
        # 自来水：衰减极小，背景主要是暗色水箱壁
        self.tank_clear_beta  = np.array([0.02, 0.02, 0.02])
        self.tank_clear_Binf  = np.array([0.08, 0.08, 0.08]) # 暗灰背景
        
        # 浑浊水箱(加牛奶)：均匀强散射，背景被白色LED和牛奶反射光主导，呈灰白色
        self.tank_turbid_beta = np.array([1.30, 1.30, 1.30]) # 牛奶导致各通道均匀衰减
        self.tank_turbid_Binf = np.array([0.45, 0.45, 0.45]) # 灰白色高亮背景

        # 偏振特性 (水体本身的偏振能力)
        self.clear_dop = 0.70  # 清水单次散射偏振度高
        self.turbid_dop = 0.10 # 浑浊水多次散射导致退偏振

    def sample_parameters(self, mode="ocean"):
        """
        mode: "ocean" 或 "tank"
        """
        """连续域随机化：生成无限种水质参数"""
        # 浑浊度因子 K，使用 beta 分布让多数数据集中在中等浑浊，少部分极端
        k = np.random.beta(1.5, 1.5)

        # 线性插值生成基础物理参数
        if mode == "tank":
            base_beta = self.tank_clear_beta + k * (self.tank_turbid_beta - self.tank_clear_beta)
            base_Binf = self.tank_clear_Binf + k * (self.tank_turbid_Binf - self.tank_clear_Binf)
            # 水箱环境光受人工 LED 和周围环境影响，引入微小的白平衡漂移
            B_inf_perturb = np.random.uniform(0.9, 1.1, size=3)
        else: # "ocean"
            base_beta = self.ocean_clear_beta + k * (self.ocean_turbid_beta - self.ocean_clear_beta)
            base_Binf = self.ocean_clear_Binf + k * (self.ocean_turbid_Binf - self.ocean_clear_Binf)
            # 自然界水体颜色变化大，引入较大的光谱微扰
            B_inf_perturb = np.random.uniform(0.5, 1.2, size=3)

        # 引入物理微扰 (Domain Randomization)
        beta = base_beta * np.random.uniform(0.8, 1.2, size=3)
        B_inf = np.clip(base_Binf * B_inf_perturb, 0, 0.9) # 防止物理界限外的死白

        dop_water = self.clear_dop - k * (self.clear_dop - self.turbid_dop)
        dop_water = np.clip(dop_water + np.random.normal(0, 0.05), 0.05, 0.85)
        aop_water = np.random.uniform(0, np.pi)

        return {
            "mode": mode,
            "turbidity_k": k,
            "beta": beta,
            "B_inf": B_inf,
            "dop_water": dop_water,
            "aop_water": aop_water
        }

    def render(self, J, depth, params, semantic_mask=None, is_tank_dataset=False):
        """
        核心渲染引擎
        is_tank_dataset: True 时缩放深度模拟 UPBD 水箱(0.1~0.5m)；False 时模拟真实海洋(1~15m)
        """
        J_linear = srgb_to_linear(J)

        #1. D深度：动态深度缩放  解决强行归一化导致的几何比例丢失，并且控制尺度
        if len(depth.shape) == 2:
            depth = depth[..., np.newaxis]

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
         # D百分位统计 (踢掉 2% 的最远和最近点，防止噪点干扰)
        valid = depth[depth > 0]
        p_min = np.percentile(valid, 2) if len(valid) > 0 else 0
        p_max = np.percentile(valid, 98) if len(valid) > 0 else 1
       # 截断异常值
        depth_clipped = np.clip(depth, p_min, p_max)

        # D计算场景原始跨度 (Scene Thickness)
        original_thickness = p_max - p_min + 1e-6
        
        # D根据场景类型进行【比例缩放】
        if is_tank_dataset:
            # --- 模拟水箱 (UPBD 风格) ---
            # 假设水箱物理跨度只有 0.2m - 0.5m
            target_thickness = np.random.uniform(0.15, 0.4) 
            # 比例因子：保持物体间的相对远近关系
            scale_factor = target_thickness / original_thickness
            # 平移：物体离镜头多远？
            offset = np.random.uniform(0.08, 0.20) 
        else:
            # --- 模拟开放海域 ---
            # 海洋跨度大，比如 3m - 10m
            target_thickness = np.random.uniform(3.0, 8.0)
            scale_factor = target_thickness / original_thickness
            offset = np.random.uniform(1.0, 3.0)

        # D生成最终物理深度：(原始深度 - 最小值) * 缩放系数 + 平移偏移
        # 这样物体之间的远近比例（比如 A 比 B 远两倍）得到了最大程度的保留
        actual_depth = (depth_clipped - p_min) * scale_factor + offset

        beta = params["beta"].reshape(1, 1, 3)
        B_inf = params["B_inf"].reshape(1, 1, 3)
        p_w = params["dop_water"]
        theta_w = params["aop_water"]

        # 2. 光学衰减计算
        t = np.exp(-beta * actual_depth)
        t = np.clip(t, 0.1, 1.0)
        Direct = J_linear * t
        Backscatter = B_inf * (1 - t)

        # # 记录前向传播和后向散射
        # cv2.imwrite("debug_direct_v2.png", linear_to_srgb(Direct))
        # cv2.imwrite("debug_backscatter_v2.png", linear_to_srgb(Backscatter))

        # 3. 语义目标偏振计算 
        # 如果有语义分割图，高光/金属赋高偏振，普通物体赋低偏振
        if semantic_mask is not None:
            # 假设 semantic_mask 中 255 是高 DOP 目标(金属)，128 是低 DOP 目标(塑料)
            p_t = np.zeros_like(actual_depth)
            p_t[semantic_mask >= 200] = np.random.uniform(0.3, 0.5)
            p_t[(semantic_mask < 200) & (semantic_mask > 50)] = np.random.uniform(0.05, 0.15)
        else:
            # 如果没有语义图，生成一个基础的弱目标偏振 (0.05~0.15)，打破零偏振假设
            p_t = np.random.uniform(0.05, 0.15)

        # 目标的偏振角通常与光源或表面法线相关，这里加一个轻微的相位差
        theta_t = np.mod(theta_w + np.random.uniform(-0.5, 0.5), np.pi)

        # 4. 生成 0, 60, 120 三角度偏振图 (对齐 UPBD 相机)
        alphas = [0, np.pi/3, 2*np.pi/3]
        angles_deg = ["0", "60", "120"]

        polarized_images = {}
        for alpha, name in zip(alphas, angles_deg):
            # 广义马吕斯定律：直接透射光（目标）和 后向散射光（水体）都有各自的偏振态！
            I_direct_pol = 0.5 * Direct * (1 + p_t * np.cos(2*alpha - 2*theta_t))
            I_back_pol   = 0.5 * Backscatter * (1 + p_w * np.cos(2*alpha - 2*theta_w))

            # 物理叠加
            I_alpha_linear = I_direct_pol + I_back_pol

            # 加入传感器噪声
            I_alpha_linear = add_camera_noise(I_alpha_linear)

            # 转回 sRGB 供网络训练或保存
            polarized_images[name] = linear_to_srgb(I_alpha_linear)

        return polarized_images
