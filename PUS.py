import numpy as np
import cv2 # 假设用 opencv 处理图像


def srgb_to_linear(img):
    """
    将 sRGB 图像转换为 线性 RGB。
    兼容输入为 [0, 255] 的 uint8 图像或 [0, 1] 的 float 图像。
    """
    # 动态检测输入范围：如果最大值大于1，或者类型是整数，则除以255
    if img.dtype == np.uint8 or img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    
    img = np.clip(img, 0.0, 1.0)
    
    # 使用 np.where 提升计算效率
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4
    )
    
    return linear.astype(np.float32)


def linear_to_srgb(img):
    """
    将 线性 RGB 转换为 sRGB。
    输入 img 应为 [0, 1] 的浮点数矩阵。
    """
    img = np.clip(img, 0.0, 1.0)
    
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * (img ** (1 / 2.4)) - 0.055
    )
    
    # 转换为 uint8 格式以便保存或显示
    return np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)


# 散粒噪声模拟 (信号越弱，噪声相对越大)
def add_camera_noise(img, noise_level=0.02):
    # img 是 0-1 之间的图像
    noise = np.random.normal(0, noise_level, img.shape)
    # 暗区噪声更明显 (泊松特性近似)
    poisson_noise = noise * np.sqrt(np.clip(img, 0.01, 1.0))
    return np.clip(img + poisson_noise, 0, 1)


class PolarizedUnderwaterSampler:
    def __init__(self):
        # 注意：OpenCV 读取图像为 BGR 顺序，所以参数也调整为 BGR
        self.jerlov_types = {
            "Oceanic_I": {
                "a":[(0.005,0.02),(0.01,0.03),(0.02,0.05)],  
                "b":[(0.02,0.04),(0.02,0.04),(0.02,0.05)],    
                "dop_range": (0.40, 0.70)  # 清水：偏振度高
            },
            "Coastal_3": {
                "a":[(0.05,0.15),(0.08,0.20),(0.15,0.35)],   
                "b":[(0.08,0.18),(0.08,0.20),(0.10,0.25)],   
                "dop_range": (0.15, 0.40)  # 中等浑浊
            },
            "Coastal_7": {
                "a":[(0.10,0.30),(0.20,0.50),(0.30,0.80)],  
                "b":[(0.20,0.50),(0.25,0.60),(0.30,0.80)],   
                "dop_range": (0.05, 0.20)  # 极浑浊：多次散射导致退偏振
            }
        }

    def sample_parameters(self):
        water = np.random.choice(list(self.jerlov_types.keys()))
        params = self.jerlov_types[water]

        a = np.array([np.random.uniform(*params["a"][i]) for i in range(3)])
        b = np.array([np.random.uniform(*params["b"][i]) for i in range(3)])

        beta = a + b
        B_inf = b / (beta + 1e-6)

        denom = B_inf.max()
        max_brightness = np.random.uniform(0.1, 0.5)
        if denom > 0:
            B_inf = B_inf / denom * max_brightness
    
        noise = np.random.normal(0, 0.02, 3)
        B_inf = np.clip(B_inf + noise, 0, 1)

        # 1. 偏振度 (Degree of Polarization, p) - 依赖水质
        dop = np.random.uniform(*params["dop_range"])

        # 2. 偏振角 (Angle of Polarization, theta) - 全局随机，范围 [0, pi]
        aop = np.random.uniform(0, np.pi)

        return {
            "water_type": water,
            "beta": beta,
            "B_inf": B_inf,
            "dop": dop,     # p
            "aop": aop      # theta
        }

    def render_polarized_images(self, J, depth, params):
        """
        输入:
            J: 干净的RGB图像, shape (H, W, 3), 范围 [0, 1]
            depth: 深度图, shape (H, W, 1) 或 (H, W), 真实物理距离(米)
            params: sample_parameters() 生成的参数字典
        输出:
            包含 0, 45, 90, 135 四个角度偏振图的字典
        """
        J_linear = srgb_to_linear(J)
        beta = params["beta"]       # (3,)
        B_inf = params["B_inf"]     # (3,)
        p = params["dop"]
        theta = params["aop"]

        # 确保维度匹配广播机制 (H, W, 3)
        if len(depth.shape) == 2:
            depth = depth[..., np.newaxis]

        beta = beta.reshape(1, 1, 3)
        B_inf = B_inf.reshape(1, 1, 3)

        # === 调试：打印物理参数 ===
        print(f"beta: {beta[0,0,:]}")
        print(f"B_inf: {B_inf[0,0,:]}")
        print(f"dop: {p:.4f}")
        print(f"aop: {theta:.4f} ")
        # ============================

        # 计算透射率 t = exp(-beta * z)
        t = np.exp(-beta * depth)

        # # === 调试：打印深度统计 ===
        # print(f"depth min: {depth.min():.4f}")
        # print(f"depth max: {depth.max():.4f}")
        # print(f"depth mean: {depth.mean():.4f}")
        # # ============================

        # 1. 直接透射光 (假设物体本身非偏振)
        Direct_unpol = J_linear * t

        # 2. 背景散射光 (非偏振基底)
        Backscatter_base = B_inf * (1 - t)

        # === 调试：保存中间结果 ===
        cv2.imwrite("debug_direct.png", linear_to_srgb(Direct_unpol))
        cv2.imwrite("debug_backscatter.png", linear_to_srgb(Backscatter_base))
        # ============================

        # 3. 渲染不同角度的偏振片图像
        # 偏振片角度 alpha 集合 (弧度)
        alphas =[0, np.pi/4, np.pi/2, 3*np.pi/4] # 0°, 45°, 90°, 135°
        angles_deg = ["0", "45", "90", "135"]

        polarized_images = {}
        for alpha, name in zip(alphas, angles_deg):
            # 依据马吕斯定律叠加偏振效应
            # 公式：I(alpha) = 0.5 * Direct + 0.5 * Backscatter * [1 + p * cos(2alpha - 2theta)]
            I_alpha_linear = 0.5 * Direct_unpol + 0.5 * Backscatter_base * (1 + p * np.cos(2*alpha - 2*theta))
            I_alpha_linear = add_camera_noise(I_alpha_linear)
            polarized_images[name] = linear_to_srgb(I_alpha_linear)
        
        return polarized_images
