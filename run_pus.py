import os
import argparse
import numpy as np
import cv2
from PUS import PolarizedUnderwaterSampler


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


def process_image(rgb_path, depth_path, sampler, use_black=False):
    """处理单张图像"""
    # 读取RGB
    if use_black:
        rgb_img = cv2.imread("black.png")
    else:
        rgb_img = cv2.imread(rgb_path)

    # 读取深度图
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 归一化
    rgb = rgb_img.astype(np.float32) / 255.0
    depth = depth_img.astype(np.float32) / 1000.0  # mm -> 米

    # 处理无效值 (0表示无效深度)
    depth[depth <= 0.01] = 0.1  # 替换为最小有效深度0.1米
    depth = np.clip(depth, 0.1, 15)

    # 验证深度范围
    print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")

    # 采样参数
    params = sampler.sample_parameters()

    # 渲染
    polarized_imgs = sampler.render_polarized_images(rgb, depth, params)

    return polarized_imgs, params


def main():
    parser = argparse.ArgumentParser(description="水下偏振图像生成")
    parser.add_argument("-n", "--num", type=int, default=None, help="处理图片数量，不指定则处理全部")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "black"], help="RGB输入模式: normal=正常RGB, black=纯黑图")
    args = parser.parse_args()

    use_black = (args.mode == "black")

    data_dir = "test_images/test"
    output_dir = "output"

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

    print(f"RGB输入模式: {args.mode}")
    print(f"水体类型: Oceanic_I(清水) / Coastal_3(中浊) / Coastal_7(极浊)")

    # 创建采样器
    sampler = PolarizedUnderwaterSampler()

    # 逐张处理
    for i, (rgb_path, depth_path) in enumerate(pairs):
        filename = os.path.basename(rgb_path)

        try:
            polarized_imgs, params = process_image(rgb_path, depth_path, sampler, use_black)

            # 保存
            base_name = filename.replace('.png', '')
            for angle, img in polarized_imgs.items():
                cv2.imwrite(f"{output_dir}/{base_name}_pol{angle}.png", img)

            print(f"[{i+1}/{len(pairs)}] {filename}")
            print(f"  水体: {params['water_type']} | DOP={params['dop']:.2f} | AOP={params['aop']:.2f}")
            print(f"  beta: {params['beta']}")
            print(f"  B_inf: {params['B_inf']}")

        except Exception as e:
            print(f"[{i+1}/{len(pairs)}] {filename} | 错误: {e}")

    print(f"\n完成！共处理 {len(pairs)} 张图片")
    print(f"结果保存在: {output_dir}/")


if __name__ == "__main__":
    main()
