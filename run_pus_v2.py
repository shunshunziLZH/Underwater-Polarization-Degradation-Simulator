import os
import argparse
import numpy as np
import cv2
from PUS_v2 import PolarizedUnderwaterSimulator_v2


def get_matched_pairs(data_dir, limit=None):
    """获取配对的RGB和深度图文件"""
    files = os.listdir(data_dir)

    # 提取数字编号作为匹配键
    pairs = []
    for f in files:
        if not f.endswith('.png'):
            continue
        # 解析编号: 00000_colors.png -> 00000
        if '_colors' in f:
            idx = f.split('_')[0]
            depth_file = f.replace('_colors', '_depth')
            if depth_file in files:
                pairs.append((os.path.join(data_dir, f),
                             os.path.join(data_dir, depth_file)))

    # 按编号排序
    pairs.sort(key=lambda x: x[0])

    if limit:
        pairs = pairs[:limit]

    return pairs


def process_image(rgb_path, depth_path, sampler, is_tank_dataset=False):
    """处理单张图像"""
    # 读取RGB
    rgb_img = cv2.imread(rgb_path)

    # 读取深度图
    depth= cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 归一化
    rgb = rgb_img.astype(np.float32) / 255.0
    # depth = depth_img.astype(np.float32) / 1000.0  # mm -> 米

    # # 处理无效值 (0表示无效深度)
    # depth[depth <= 0.01] = 0.1  # 替换为最小有效深度0.1米
    # depth = np.clip(depth, 0.1, 15)

    # 验证深度范围
    print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")

    # 采样参数 (根据模式选择 ocean 或 tank)
    mode = "tank" if is_tank_dataset else "ocean"
    params = sampler.sample_parameters(mode=mode)

    # 渲染 (PUS_v2 使用 render 方法)
    polarized_imgs = sampler.render(rgb, depth, params, semantic_mask=None, is_tank_dataset=is_tank_dataset)

    return polarized_imgs, params


def main():
    parser = argparse.ArgumentParser(description="水下偏振图像生成 v2")
    parser.add_argument("-n", "--num", type=int, default=None, help="处理图片数量，不指定则处理全部")
    parser.add_argument("--tank", action="store_true", help="模拟UPBD水箱模式(0.2~0.5m)，否则为深海模式(1~15m)")
    args = parser.parse_args()

    is_tank_dataset = args.tank

    data_dir = "test_images/test"
    output_dir = "output_v2"

    os.makedirs(output_dir, exist_ok=True)

    # 获取配对文件
    print("正在扫描配对图像...")
    pairs = get_matched_pairs(data_dir)
    total = len(pairs)

    # 根据命令行参数决定处理数量
    if args.num:
        pairs = pairs[:args.num]
        print(f"处理前 {len(pairs)} 张图片")
    else:
        print(f"共 {total} 张图片")

    mode_name = "水箱模式 (0.2~0.5m)" if is_tank_dataset else "深海模式 (1~15m)"
    print(f"深度模式: {mode_name}")
    print(f"偏振角度: 0°, 60°, 120° (对齐UPBD相机)")

    # 创建采样器
    sampler = PolarizedUnderwaterSimulator_v2()

    # 逐张处理
    for i, (rgb_path, depth_path) in enumerate(pairs):
        filename = os.path.basename(rgb_path)

        try:
            polarized_imgs, params = process_image(rgb_path, depth_path, sampler, is_tank_dataset)

            # 保存
            base_name = filename.replace('.png', '')
            for angle, img in polarized_imgs.items():
                cv2.imwrite(f"{output_dir}/{base_name}_pol{angle}.png", img)

            print(f"[{i+1}/{len(pairs)}] {filename}")
            print(f"  浑浊度K={params['turbidity_k']:.3f} | 水体DoP={params['dop_water']:.2f} | 水体AoP={params['aop_water']:.2f}")
            print(f"  beta: {params['beta']}")
            print(f"  B_inf: {params['B_inf']}")

        except Exception as e:
            print(f"[{i+1}/{len(pairs)}] {filename} | 错误: {e}")

    print(f"\n完成！共处理 {len(pairs)} 张图片")
    print(f"结果保存在: {output_dir}/")


if __name__ == "__main__":
    main()
