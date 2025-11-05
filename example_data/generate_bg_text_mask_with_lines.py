"""
配置文件：生成带有大量line干扰的bg_and_text_mask样本
用于训练文字抠图模型
"""
import os
from pathlib import Path

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
    FixedTextColorCfg,
)

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output" / "bg_text_mask_with_lines"
BG_DIR = CURRENT_DIR / "bg"
CHAR_DIR = CURRENT_DIR / "char"
FONT_DIR = CURRENT_DIR / "font"
TEXT_DIR = CURRENT_DIR / "text"

# 字体配置
font_cfg = dict(
    font_dir=FONT_DIR,
    font_size=(25, 40),
)


def get_corpus():
    """获取语料库"""
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            length=(5, 15),
            char_spacing=(-0.2, 1.0),
            **font_cfg
        ),
    )


def bg_text_mask_with_horizontal_lines():
    """生成带水平线干扰的样本"""
    return GeneratorCfg(
        num_image=200,
        save_dir=OUT_DIR / "horizontal_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.8,  # 80%概率应用line效果
                    thickness=(1, 5),  # 线条粗细
                    line_pos_p=(0.2, 0.2, 0, 0, 0, 0, 0, 0, 0.3, 0.3),  # top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, horizontal_middle, vertical_middle
                    color_cfg=None,  # 随机颜色
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def bg_text_mask_with_vertical_lines():
    """生成带垂直线干扰的样本"""
    return GeneratorCfg(
        num_image=200,
        save_dir=OUT_DIR / "vertical_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.8,
                    thickness=(1, 5),
                    line_pos_p=(0, 0, 0.2, 0.2, 0, 0, 0, 0, 0.3, 0.3),  # 主要是左右和垂直中间
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def bg_text_mask_with_corner_lines():
    """生成带角落线干扰的样本"""
    return GeneratorCfg(
        num_image=200,
        save_dir=OUT_DIR / "corner_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.85,
                    thickness=(1, 6),
                    line_pos_p=(0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0),  # 四个角
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def bg_text_mask_with_all_lines():
    """生成带各种线干扰的混合样本"""
    return GeneratorCfg(
        num_image=300,
        save_dir=OUT_DIR / "all_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.9,  # 90%概率
                    thickness=(1, 6),
                    lr_in_offset=(0, 15),  # 左右内偏移
                    lr_out_offset=(0, 10),  # 左右外偏移
                    tb_in_offset=(0, 5),   # 上下内偏移
                    tb_out_offset=(0, 5),  # 上下外偏移
                    line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15),  # 所有位置均匀分布
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def bg_text_mask_with_multiple_lines():
    """生成带多条线干扰的样本（通过多个Line effect叠加）"""
    return GeneratorCfg(
        num_image=250,
        save_dir=OUT_DIR / "multiple_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.7,
                    thickness=(2, 4),
                    line_pos_p=(0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                    color_cfg=None,
                ),
                Line(
                    p=0.5,  # 第二条线50%概率
                    thickness=(1, 3),
                    line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def bg_text_mask_with_thick_lines():
    """生成带粗线干扰的样本"""
    return GeneratorCfg(
        num_image=150,
        save_dir=OUT_DIR / "thick_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.85,
                    thickness=(4, 10),  # 更粗的线条
                    line_pos_p=(0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05),  # 总和=1.0
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def bg_text_mask_no_lines():
    """生成不带线干扰的样本（作为对照）"""
    return GeneratorCfg(
        num_image=100,
        save_dir=OUT_DIR / "no_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_corpus(),
            corpus_effects=None,  # 不应用任何效果
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


# 配置列表：所有要生成的数据集
configs = [
    bg_text_mask_with_horizontal_lines(),  # 200张 水平线
    bg_text_mask_with_vertical_lines(),    # 200张 垂直线
    bg_text_mask_with_corner_lines(),      # 200张 角落线
    bg_text_mask_with_all_lines(),         # 300张 所有位置
    bg_text_mask_with_multiple_lines(),    # 250张 多条线
    bg_text_mask_with_thick_lines(),       # 150张 粗线
    bg_text_mask_no_lines(),               # 100张 无线（对照）
]

# 总计：1400张样本
