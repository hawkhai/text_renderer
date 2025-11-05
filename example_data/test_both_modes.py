"""
测试两种存储模式
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
OUT_DIR = CURRENT_DIR / "output" / "test_modes"
BG_DIR = CURRENT_DIR / "bg"
CHAR_DIR = CURRENT_DIR / "char"
FONT_DIR = CURRENT_DIR / "font"
FONT_LIST_DIR = CURRENT_DIR / "font_list"
TEXT_DIR = CURRENT_DIR / "text"


def get_chinese_corpus():
    """中文语料"""
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            font_dir=FONT_DIR,
            font_list_file=FONT_LIST_DIR / "chinese_fonts.txt",
            font_size=(30, 55),
            length=(5, 15),
            char_spacing=(-0.2, 1.0),
        ),
    )


def mode_merged():
    """模式1: 三图合一（上下堆叠）"""
    return GeneratorCfg(
        num_image=3,
        save_dir=OUT_DIR / "merged_mode",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=Effects([
                Line(p=0.8, thickness=(2, 5), line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=False,  # 三图合一模式
        ),
    )


def mode_separated():
    """模式2: 分离存储（images/ 和 labels/）"""
    return GeneratorCfg(
        num_image=3,
        save_dir=OUT_DIR / "separated_mode",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=Effects([
                Line(p=0.8, thickness=(2, 5), line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
            save_mask_separately=True,  # 分离存储模式
        ),
    )


configs = [
    mode_merged(),
    mode_separated(),
]
