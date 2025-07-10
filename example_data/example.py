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


def get_font_names():
    """Read font names from font_list.txt"""
    font_list_file = FONT_LIST_DIR / "font_list.txt"
    if font_list_file.exists():
        with open(font_list_file, 'r', encoding='utf-8') as f:
            fonts = [line.strip() for line in f if line.strip()]
        return fonts
    return ["simsun.ttf"]  # fallback


def create_font_specific_cfg(font_name, base_name_suffix=""):
    """Create font-specific configuration"""
    font_name_clean = font_name.replace('.ttf', '').replace('.otf', '').replace('.ttc', '')
    return dict(
        font_dir=FONT_DIR,
        font_list_file=None,  # Use specific font
        font_path=FONT_DIR / font_name,  # Direct font path
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


def create_vertical_configs_by_font():
    """
    Create vertical text configurations organized by font and corpus language
    Directory structure: font_name/corpus_language/style/
    """
    font_names = get_font_names()
    configs = []
    
    # Define corpus configurations
    corpus_configs = {
        'chinese': {
            'text_paths': [TEXT_DIR / "chn_text.txt"],
            'chars_file': CHAR_DIR / "chn.txt",
            'name': 'chinese'
        },
        'english': {
            'text_paths': [TEXT_DIR / "eng_text.txt"],
            'chars_file': CHAR_DIR / "eng.txt",
            'name': 'english'
        },
        'mixed': {
            'text_paths': [TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            'chars_file': CHAR_DIR / "chn.txt",
            'name': 'mixed'
        }
    }
    
    # Style configurations
    style_configs = {
        'basic': {
            'length': (4, 8),
            'char_spacing': (0.05, 0.25),
            'font_size': (30, 31),
            'effects': Effects([
                Line(0.2, color_cfg=FixedTextColorCfg()),
                OneOf([DropoutRand(), DropoutVertical(), NoEffects()]),
            ]),
            'gray': True
        },
        'long': {
            'length': (8, 15),
            'char_spacing': (0.1, 0.4),
            'font_size': (25, 35),
            'effects': Effects([
                OneOf([
                    Line(0.4, color_cfg=FixedTextColorCfg()),
                    DropoutVertical(),
                    Padding(p=0.3, w_ratio=[0.1, 0.2], h_ratio=[0.1, 0.3], center=True),
                    NoEffects()
                ]),
            ]),
            'gray': True
        },
        'compact': {
            'length': (5, 12),
            'char_spacing': (-0.1, 0.1),
            'font_size': (28, 32),
            'effects': Effects([
                OneOf([
                    DropoutRand(dropout_p=(0.1, 0.3)),
                    DropoutVertical(num_line=2),
                    Line(0.3, thickness=(1, 3)),
                    NoEffects()
                ]),
            ]),
            'gray': True
        },
        'stylized': {
            'length': (6, 10),
            'char_spacing': (0.2, 0.5),
            'font_size': (32, 40),
            'effects': Effects([
                Padding(p=0.5, w_ratio=[0.15, 0.25], h_ratio=[0.2, 0.4], center=True),
                OneOf([
                    [Line(0.6, color_cfg=FixedTextColorCfg(), thickness=(2, 4)), DropoutRand()],
                    DropoutVertical(num_line=3, thickness=2),
                    ImgAugEffect(aug=iaa.Emboss(alpha=(0.5, 0.8), strength=(0.8, 1.2))),
                    NoEffects()
                ]),
            ]),
            'gray': False
        }
    }
    
    # Calculate images per configuration
    total_configs = len(font_names) * len(corpus_configs) * len(style_configs)
    images_per_config = 1000 // total_configs
    
    for font_name in font_names:
        font_name_clean = font_name.replace('.ttf', '').replace('.otf', '').replace('.ttc', '')
        
        for corpus_key, corpus_info in corpus_configs.items():
            for style_key, style_info in style_configs.items():
                # Create configuration name: font/corpus/style
                cfg_name = f"{font_name_clean}/{corpus_info['name']}/{style_key}"
                
                cfg = GeneratorCfg(
                    num_image=images_per_config,
                    save_dir=OUT_DIR / cfg_name,
                    render_cfg=RenderCfg(
                        bg_dir=BG_DIR,
                        perspective_transform=perspective_transform,
                        gray=style_info['gray'],
                        corpus=CharCorpus(
                            CharCorpusCfg(
                                text_paths=corpus_info['text_paths'],
                                filter_by_chars=True,
                                chars_file=corpus_info['chars_file'],
                                length=style_info['length'],
                                char_spacing=style_info['char_spacing'],
                                horizontal=False,  # Vertical text
                                font_dir=FONT_DIR,
                                font_size=style_info['font_size'],
                            ),
                        ),
                        corpus_effects=style_info['effects'],
                    ),
                )
                configs.append(cfg)
    
    return configs


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
    # Vertical text samples - organized by font, 4 styles each
    *create_vertical_configs_by_font(),  # Dynamic configs for each font
]
# fmt: on
