"""
多语言混合配置：中文60%、英文30%、中英混合10%
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
)

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output" / "mixed_language"
BG_DIR = CURRENT_DIR / "bg"
CHAR_DIR = CURRENT_DIR / "char"
FONT_DIR = CURRENT_DIR / "font"
TEXT_DIR = CURRENT_DIR / "text"

# 常用中文字体列表
CHINESE_FONTS = [
    "宋体_常规.ttf",
    "微软雅黑_regular.ttf",
    "黑体_regular.ttf",
    "楷体_regular.ttf",
    "仿宋_regular.ttf",
    "思源黑体_cn_regular.ttf",
    "思源宋体_cn_regular.ttf",
]

# 常用英文字体列表
ENGLISH_FONTS = [
    "arial_regular.ttf",
    "arial_bold.ttf",
    "times_new_roman_regular.ttf",
    "times_new_roman_bold.ttf",
    "calibri_regular.ttf",
    "calibri_bold.ttf",
]

# 创建字体列表文件
def create_font_lists():
    """创建中文和英文字体列表文件"""
    # 中文字体列表
    chinese_font_list = FONT_DIR.parent / "font_list" / "chinese_fonts.txt"
    chinese_font_list.parent.mkdir(exist_ok=True)
    with open(chinese_font_list, 'w', encoding='utf-8') as f:
        for font in CHINESE_FONTS:
            f.write(f"{font}\n")
    
    # 英文字体列表
    english_font_list = FONT_DIR.parent / "font_list" / "english_fonts.txt"
    with open(english_font_list, 'w', encoding='utf-8') as f:
        for font in ENGLISH_FONTS:
            f.write(f"{font}\n")
    
    return chinese_font_list, english_font_list

# 创建字体列表
CHINESE_FONT_LIST, ENGLISH_FONT_LIST = create_font_lists()


def get_chinese_corpus():
    """中文语料"""
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            font_dir=FONT_DIR,
            font_list_file=CHINESE_FONT_LIST,
            font_size=(30, 55),  # 增大字体范围以达到更长的像素宽度
            length=(5, 15),      # 最长15个汉字
            char_spacing=(-0.2, 1.0),
        ),
    )


def get_english_corpus():
    """英文语料"""
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "eng_text.txt"],
            filter_by_chars=False,
            font_dir=FONT_DIR,
            font_list_file=ENGLISH_FONT_LIST,
            font_size=(30, 55),  # 增大字体范围
            length=(8, 25),      # 英文字符较窄，增加字符数以达到相似宽度
            char_spacing=(-0.1, 0.3),
        ),
    )


def get_mixed_corpus():
    """中英混合语料"""
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            font_dir=FONT_DIR,
            font_size=(30, 55),  # 增大字体范围
            length=(5, 18),      # 混合文本，稍微增加字符数
            char_spacing=(-0.2, 1.0),
        ),
    )


# ==================== 中文样本配置 (60%) ====================

def chinese_horizontal_lines():
    """中文 - 水平线"""
    return GeneratorCfg(
        num_image=600,  # 120 * 5
        save_dir=OUT_DIR / "chinese_horizontal_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.8,
                    thickness=(1, 5),
                    line_pos_p=(0.2, 0.2, 0, 0, 0, 0, 0, 0, 0.3, 0.3),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=True,
        ),
    )


def chinese_vertical_lines():
    """中文 - 垂直线"""
    return GeneratorCfg(
        num_image=600,  # 120 * 5
        save_dir=OUT_DIR / "chinese_vertical_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.8,
                    thickness=(1, 5),
                    line_pos_p=(0, 0, 0.2, 0.2, 0, 0, 0, 0, 0.3, 0.3),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=True,
        ),
    )


def chinese_all_lines():
    """中文 - 所有线条"""
    return GeneratorCfg(
        num_image=900,  # 180 * 5
        save_dir=OUT_DIR / "chinese_all_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.9,
                    thickness=(1, 6),
                    line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=True,
        ),
    )


def chinese_multiple_lines():
    """中文 - 多条线"""
    return GeneratorCfg(
        num_image=750,  # 150 * 5
        save_dir=OUT_DIR / "chinese_multiple_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=Effects([
                Line(p=0.7, thickness=(2, 4), line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
                Line(p=0.5, thickness=(1, 3), line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=True,
        ),
    )


def chinese_no_lines():
    """中文 - 无线条"""
    return GeneratorCfg(
        num_image=300,  # 60 * 5
        save_dir=OUT_DIR / "chinese_no_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=None,
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=True,
        ),
    )


# ==================== 英文样本配置 (30%) ====================

def english_horizontal_lines():
    """英文 - 水平线"""
    return GeneratorCfg(
        num_image=300,  # 60 * 5
        save_dir=OUT_DIR / "english_horizontal_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_english_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.8,
                    thickness=(1, 5),
                    line_pos_p=(0.2, 0.2, 0, 0, 0, 0, 0, 0, 0.3, 0.3),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def english_vertical_lines():
    """英文 - 垂直线"""
    return GeneratorCfg(
        num_image=300,  # 60 * 5
        save_dir=OUT_DIR / "english_vertical_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_english_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.8,
                    thickness=(1, 5),
                    line_pos_p=(0, 0, 0.2, 0.2, 0, 0, 0, 0, 0.3, 0.3),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def english_all_lines():
    """英文 - 所有线条"""
    return GeneratorCfg(
        num_image=450,  # 90 * 5
        save_dir=OUT_DIR / "english_all_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_english_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.9,
                    thickness=(1, 6),
                    line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def english_multiple_lines():
    """英文 - 多条线"""
    return GeneratorCfg(
        num_image=375,  # 75 * 5
        save_dir=OUT_DIR / "english_multiple_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_english_corpus(),
            corpus_effects=Effects([
                Line(p=0.7, thickness=(2, 4), line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
                Line(p=0.5, thickness=(1, 3), line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def english_no_lines():
    """英文 - 无线条"""
    return GeneratorCfg(
        num_image=150,  # 30 * 5
        save_dir=OUT_DIR / "english_no_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_english_corpus(),
            corpus_effects=None,
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=True,
        ),
    )


# ==================== 中英混合样本配置 (10%) ====================

def mixed_horizontal_lines():
    """混合 - 水平线"""
    return GeneratorCfg(
        num_image=100,  # 20 * 5
        save_dir=OUT_DIR / "mixed_horizontal_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_mixed_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.8,
                    thickness=(1, 5),
                    line_pos_p=(0.2, 0.2, 0, 0, 0, 0, 0, 0, 0.3, 0.3),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def mixed_vertical_lines():
    """混合 - 垂直线"""
    return GeneratorCfg(
        num_image=100,  # 20 * 5
        save_dir=OUT_DIR / "mixed_vertical_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_mixed_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.8,
                    thickness=(1, 5),
                    line_pos_p=(0, 0, 0.2, 0.2, 0, 0, 0, 0, 0.3, 0.3),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def mixed_all_lines():
    """混合 - 所有线条"""
    return GeneratorCfg(
        num_image=150,  # 30 * 5
        save_dir=OUT_DIR / "mixed_all_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_mixed_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.9,
                    thickness=(1, 6),
                    line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def mixed_multiple_lines():
    """混合 - 多条线"""
    return GeneratorCfg(
        num_image=125,  # 25 * 5
        save_dir=OUT_DIR / "mixed_multiple_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_mixed_corpus(),
            corpus_effects=Effects([
                Line(p=0.7, thickness=(2, 4), line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
                Line(p=0.5, thickness=(1, 3), line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def mixed_no_lines():
    """混合 - 无线条"""
    return GeneratorCfg(
        num_image=50,  # 10 * 5
        save_dir=OUT_DIR / "mixed_no_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_mixed_corpus(),
            corpus_effects=None,
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=True,
        ),
    )


# 配置列表
# 中文: 630张 (60%)
# 英文: 315张 (30%)
# 混合: 105张 (10%)
# 总计: 1050张
configs = [
    # 中文 (630张)
    chinese_horizontal_lines(),    # 120
    chinese_vertical_lines(),      # 120
    chinese_all_lines(),           # 180
    chinese_multiple_lines(),      # 150
    chinese_no_lines(),            # 60
    
    # 英文 (315张)
    english_horizontal_lines(),    # 60
    english_vertical_lines(),      # 60
    english_all_lines(),           # 90
    english_multiple_lines(),      # 75
    english_no_lines(),            # 30
    
    # 混合 (105张)
    mixed_horizontal_lines(),      # 20
    mixed_vertical_lines(),        # 20
    mixed_all_lines(),             # 30
    mixed_multiple_lines(),        # 25
    mixed_no_lines(),              # 10
]

# 总计: 1050张样本
# 中文: 630 (60%)
# 英文: 315 (30%)
# 混合: 105 (10%)
