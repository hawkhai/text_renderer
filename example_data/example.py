import inspect
import os
from pathlib import Path
import imgaug.augmenters as iaa

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    FixedTextColorCfg,
)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout


CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output"
DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg"
CHAR_DIR = DATA_DIR / "char"
FONT_DIR = DATA_DIR / "font"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "text"

font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(30, 31),
)

perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)


def get_char_corpus():
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            length=(5, 10),
            char_spacing=(-0.3, 1.3),
            **font_cfg
        ),
    )


def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True
):
    return GeneratorCfg(
        num_image=50,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )


def chn_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Line(0.5, color_cfg=FixedTextColorCfg()),
                OneOf([DropoutRand(), DropoutVertical()]),
            ]
        ),
    )


def enum_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "enum_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "chn.txt",
                **font_cfg
            ),
        ),
    )


def rand_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=RandCorpus(
            RandCorpusCfg(chars_file=CHAR_DIR / "chn.txt", **font_cfg),
        ),
    )


def eng_word_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=WordCorpus(
            WordCorpusCfg(
                text_paths=[TEXT_DIR / "eng_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
    )


def same_line_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=[
            EnumCorpus(
                EnumCorpusCfg(
                    text_paths=[TEXT_DIR / "enum_text.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    **font_cfg
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    length=(5, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
        ],
        corpus_effects=[Effects([Padding(), DropoutRand()]), NoEffects()],
        layout_effects=Effects(Line(p=1)),
    )


def extra_text_line_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=ExtraTextLineLayout(),
        corpus=[
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    length=(9, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    length=(9, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
        ],
        corpus_effects=[Effects([Padding()]), NoEffects()],
        layout_effects=Effects(Line(p=1)),
    )


def imgaug_emboss_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
            ]
        ),
    )


def vertical_text_basic():
    """
    Basic vertical text samples with standard effects
    """
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=CharCorpus(
            CharCorpusCfg(
                text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "chn.txt",
                length=(4, 8),
                char_spacing=(0.05, 0.25),
                horizontal=False,
                **font_cfg
            ),
        ),
        corpus_effects=Effects(
            [
                Line(0.2, color_cfg=FixedTextColorCfg()),
                OneOf([DropoutRand(), DropoutVertical(), NoEffects()]),
            ]
        ),
    )
    cfg.num_image = 2500  # 1/4 of total vertical images
    return cfg


def vertical_text_long():
    """
    Long vertical text samples
    """
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=CharCorpus(
            CharCorpusCfg(
                text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "chn.txt",
                length=(8, 15),  # Longer text
                char_spacing=(0.1, 0.4),
                horizontal=False,
                font_dir=font_cfg["font_dir"],
                font_list_file=font_cfg["font_list_file"],
                font_size=(25, 35),  # Variable font size
            ),
        ),
        corpus_effects=Effects(
            [
                OneOf([
                    Line(0.4, color_cfg=FixedTextColorCfg()),
                    DropoutVertical(),
                    Padding(p=0.3, w_ratio=[0.1, 0.2], h_ratio=[0.1, 0.3], center=True),
                    NoEffects()
                ]),
            ]
        ),
    )
    cfg.num_image = 2500  # 1/4 of total vertical images
    return cfg


def vertical_text_compact():
    """
    Compact vertical text with tight spacing
    """
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=CharCorpus(
            CharCorpusCfg(
                text_paths=[TEXT_DIR / "chn_text.txt"],  # Focus on Chinese
                filter_by_chars=True,
                chars_file=CHAR_DIR / "chn.txt",
                length=(5, 12),
                char_spacing=(-0.1, 0.1),  # Tight spacing
                horizontal=False,
                font_dir=font_cfg["font_dir"],
                font_list_file=font_cfg["font_list_file"],
                font_size=(28, 32),
            ),
        ),
        corpus_effects=Effects(
            [
                OneOf([
                    DropoutRand(dropout_p=(0.1, 0.3)),
                    DropoutVertical(num_line=(1, 3)),
                    Line(0.3, thickness=(1, 3)),
                    NoEffects()
                ]),
            ]
        ),
    )
    cfg.num_image = 2500  # 1/4 of total vertical images
    return cfg


def vertical_text_stylized():
    """
    Stylized vertical text with enhanced effects
    """
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=CharCorpus(
            CharCorpusCfg(
                text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "chn.txt",
                length=(6, 10),
                char_spacing=(0.2, 0.5),  # Loose spacing for effects
                horizontal=False,
                font_dir=font_cfg["font_dir"],
                font_list_file=font_cfg["font_list_file"],
                font_size=(32, 40),  # Larger font for effects
            ),
        ),
        corpus_effects=Effects(
            [
                Padding(p=0.5, w_ratio=[0.15, 0.25], h_ratio=[0.2, 0.4], center=True),
                OneOf([
                    [Line(0.6, color_cfg=FixedTextColorCfg(), thickness=(2, 4)), DropoutRand()],
                    DropoutVertical(num_line=(2, 5), thickness=(2, 3)),
                    ImgAugEffect(aug=iaa.Emboss(alpha=(0.5, 0.8), strength=(0.8, 1.2))),
                    NoEffects()
                ]),
            ]
        ),
        gray=False,  # Color images for some variety
    )
    cfg.num_image = 2500  # 1/4 of total vertical images
    return cfg


# fmt: off
# The configuration file must have a configs variable
configs = [
    chn_data(),
    enum_data(),
    rand_data(),
    eng_word_data(),
    same_line_data(),
    extra_text_line_data(),
    imgaug_emboss_example(),
    # Vertical text samples - 4 different styles, 10000 images total
    vertical_text_basic(),     # 2500 images - basic vertical text
    vertical_text_long(),      # 2500 images - longer vertical text
    vertical_text_compact(),   # 2500 images - compact spacing
    vertical_text_stylized(),  # 2500 images - enhanced effects
]
# fmt: on
