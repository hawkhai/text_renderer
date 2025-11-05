"""
生成各种类型各10张样本用于Review
包含：水平线、垂直线、角落线、所有线条、多条线、粗线、无线条
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
OUT_DIR = CURRENT_DIR / "output" / "review_all_types"
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


def horizontal_lines():
    """水平线"""
    return GeneratorCfg(
        num_image=10,
        save_dir=OUT_DIR / "horizontal_lines",
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
        ),
    )


def vertical_lines():
    """垂直线"""
    return GeneratorCfg(
        num_image=10,
        save_dir=OUT_DIR / "vertical_lines",
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
        ),
    )


def corner_lines():
    """角落线"""
    return GeneratorCfg(
        num_image=10,
        save_dir=OUT_DIR / "corner_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.85,
                    thickness=(1, 6),
                    line_pos_p=(0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def all_lines():
    """所有线条"""
    return GeneratorCfg(
        num_image=10,
        save_dir=OUT_DIR / "all_lines",
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
        ),
    )


def multiple_lines():
    """多条线"""
    return GeneratorCfg(
        num_image=10,
        save_dir=OUT_DIR / "multiple_lines",
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
        ),
    )


def thick_lines():
    """粗线"""
    return GeneratorCfg(
        num_image=10,
        save_dir=OUT_DIR / "thick_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=Effects([
                Line(
                    p=0.85,
                    thickness=(4, 10),
                    line_pos_p=(0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05),
                    color_cfg=None,
                ),
            ]),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


def no_lines():
    """无线条"""
    return GeneratorCfg(
        num_image=10,
        save_dir=OUT_DIR / "no_lines",
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            perspective_transform=NormPerspectiveTransformCfg(15, 15, 1.2),
            corpus=get_chinese_corpus(),
            corpus_effects=None,
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
            return_bg_and_mask=True,
        ),
    )


configs = [
    horizontal_lines(),
    vertical_lines(),
    corner_lines(),
    all_lines(),
    multiple_lines(),
    thick_lines(),
    no_lines(),
]
