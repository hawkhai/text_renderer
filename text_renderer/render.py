from typing import Tuple, List

from PIL import Image
from loguru import logger

import cv2
import numpy as np
from PIL.Image import Image as PILImage
from PIL.ImageFont import FreeTypeFont
from tenacity import retry

from text_renderer.bg_manager import BgManager
from text_renderer.config import RenderCfg
from text_renderer.utils.draw_utils import draw_text_on_bg, transparent_img
from text_renderer.utils import utils
from text_renderer.utils.errors import PanicError
from text_renderer.utils.math_utils import PerspectiveTransform
from text_renderer.utils.bbox import BBox
from text_renderer.utils.font_text import FontText
from text_renderer.utils.types import FontColor, is_list


class Render:
    def __init__(self, cfg: RenderCfg):
        self.cfg = cfg
        self.layout = cfg.layout
        if isinstance(cfg.corpus, list) and len(cfg.corpus) == 1:
            self.corpus = cfg.corpus[0]
        else:
            self.corpus = cfg.corpus

        if is_list(self.corpus) and is_list(self.cfg.corpus_effects):
            if len(self.corpus) != len(self.cfg.corpus_effects):
                raise PanicError(
                    f"corpus length({self.corpus}) is not equal to corpus_effects length({self.cfg.corpus_effects})"
                )

        if is_list(self.corpus) and (
            self.cfg.corpus_effects and not is_list(self.cfg.corpus_effects)
        ):
            raise PanicError("corpus is list, corpus_effects is not list")

        if not is_list(self.corpus) and is_list(self.cfg.corpus_effects):
            raise PanicError("corpus_effects is list, corpus is not list")

        self.bg_manager = BgManager(cfg.bg_dir, cfg.pre_load_bg_img)

    @retry
    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, str]:
        try:
            if self._should_apply_layout():
                img, text, cropped_bg, transformed_text_mask, pure_text_mask = self.gen_multi_corpus()
            else:
                img, text, cropped_bg, transformed_text_mask, pure_text_mask = self.gen_single_corpus()

            if self.cfg.render_effects is not None:
                img, _ = self.cfg.render_effects.apply_effects(
                    img, BBox.from_size(img.size)
                )

            if self.cfg.return_bg_and_mask:
                # 使用纯净的文字mask（不含干扰效果），保留灰度信息
                # 将RGBA转为灰度，保留alpha通道作为灰度值
                pure_text_array = np.array(transformed_text_mask)
                if pure_text_array.shape[2] == 4:  # RGBA
                    # 使用alpha通道作为mask强度（保留灰度渐变）
                    gray_mask = pure_text_array[:, :, 3]  # 取alpha通道
                else:
                    gray_mask = cv2.cvtColor(pure_text_array, cv2.COLOR_RGB2GRAY)
                
                # 先对img进行norm处理（调整高度到64），获得目标尺寸
                img_array = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                img_normed = self.norm(img_array)
                target_h, target_w = img_normed.shape[:2]  # 获取norm后的目标尺寸
                
                # mask按相同的尺寸resize，保持单通道
                mask_normed = cv2.resize(gray_mask, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                
                # 增强mask：确保文字区域有很多255像素
                mask_normed = mask_normed.astype(np.float32)
                mask_max = mask_normed.max()
                if mask_max > 0:
                    mask_normed = (mask_normed / mask_max * 255)
                    mask_normed = np.where(mask_normed > 30, mask_normed, 0)
                    mask_normed = np.where(mask_normed > 150, 255, mask_normed)
                    mask_normed = mask_normed.clip(0, 255).astype(np.uint8)
                else:
                    mask_normed = mask_normed.astype(np.uint8)
                
                # 根据save_mask_separately参数决定返回格式
                if self.cfg.save_mask_separately:
                    # 分离存储模式：返回[input_image, mask_image]
                    # input_image: BGR格式，用于保存为jpg
                    # mask_image: 单通道灰度，用于保存为png
                    np_img = [img_normed, mask_normed]
                else:
                    # 三图合一模式：上下堆叠输入图、背景图、mask图
                    bg_array = cv2.cvtColor(np.array(cropped_bg.convert("RGB")), cv2.COLOR_RGB2BGR)
                    bg_normed = cv2.resize(bg_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    
                    # 转mask为RGB用于显示
                    mask_display = cv2.cvtColor(mask_normed, cv2.COLOR_GRAY2BGR)
                    
                    # 三图上下堆叠：输入图、背景图、mask图
                    merge_target = np.zeros((target_h * 3, target_w, 3), dtype=np.uint8)
                    merge_target[0:target_h, :] = img_normed           # 顶部：带干扰的输入
                    merge_target[target_h:target_h*2, :] = bg_normed   # 中部：纯背景
                    merge_target[target_h*2:target_h*3, :] = mask_display  # 底部：纯净mask
                    
                    np_img = merge_target
            else:
                img = img.convert("RGB")
                np_img = np.array(img)
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                np_img = self.norm(np_img)
            return np_img, text
        except Exception as e:
            logger.exception(e)
            raise e

    def gen_single_corpus(self) -> Tuple[PILImage, str, PILImage, PILImage, PILImage]:
        font_text = self.corpus.sample()

        bg = self.bg_manager.get_bg()
        if self.cfg.text_color_cfg is not None:
            text_color = self.cfg.text_color_cfg.get_color(bg)

        # corpus text_color has higher priority than RenderCfg.text_color_cfg
        if self.corpus.cfg.text_color_cfg is not None:
            text_color = self.corpus.cfg.text_color_cfg.get_color(bg)

        text_mask = draw_text_on_bg(
            font_text, text_color, char_spacing=self.corpus.cfg.char_spacing
        )
        
        # 保存纯净的文字mask（用于生成最终的mask输出，不含干扰）
        pure_text_mask = text_mask.copy()

        if self.cfg.corpus_effects is not None:
            text_mask, _ = self.cfg.corpus_effects.apply_effects(
                text_mask, BBox.from_size(text_mask.size)
            )

        if self.cfg.perspective_transform is not None:
            transformer = PerspectiveTransform(self.cfg.perspective_transform)
            # TODO: refactor this, now we must call get_transformed_size to call gen_warp_matrix
            _ = transformer.get_transformed_size(text_mask.size)

            try:
                (
                    transformed_text_mask,
                    transformed_text_pnts,
                ) = transformer.do_warp_perspective(text_mask)
                
                # 对纯净mask也应用相同的perspective transform
                pure_transformed_mask, _ = transformer.do_warp_perspective(pure_text_mask)
            except Exception as e:
                logger.exception(e)
                logger.error(font_text.font_path, "text", font_text.text)
                raise e
        else:
            transformed_text_mask = text_mask
            pure_transformed_mask = pure_text_mask

        img, cropped_bg = self.paste_text_mask_on_bg(bg, transformed_text_mask)

        return img, font_text.text, cropped_bg, transformed_text_mask, pure_transformed_mask

    def gen_multi_corpus(self) -> Tuple[PILImage, str, PILImage, PILImage, PILImage]:
        font_texts: List[FontText] = [it.sample() for it in self.corpus]

        bg = self.bg_manager.get_bg()

        text_color = None
        if self.cfg.text_color_cfg is not None:
            text_color = self.cfg.text_color_cfg.get_color(bg)

        text_masks, text_bboxes, pure_text_masks = [], [], []
        for i in range(len(font_texts)):
            font_text = font_texts[i]

            if text_color is None:
                _text_color = self.corpus[i].cfg.text_color_cfg.get_color(bg)
            else:
                _text_color = text_color
            text_mask = draw_text_on_bg(
                font_text, _text_color, char_spacing=self.corpus[i].cfg.char_spacing
            )
            
            # 保存纯净的文字mask
            pure_text_masks.append(text_mask.copy())

            text_bbox = BBox.from_size(text_mask.size)
            if self.cfg.corpus_effects is not None:
                effects = self.cfg.corpus_effects[i]
                if effects is not None:
                    text_mask, text_bbox = effects.apply_effects(text_mask, text_bbox)
            text_masks.append(text_mask)
            text_bboxes.append(text_bbox)

        text_mask_bboxes, merged_text = self.layout(
            font_texts,
            [it.copy() for it in text_bboxes],
            [BBox.from_size(it.size) for it in text_masks],
        )
        if len(text_mask_bboxes) != len(text_bboxes):
            raise PanicError(
                "points and text_bboxes should have same length after layout output"
            )

        merged_bbox = BBox.from_bboxes(text_mask_bboxes)
        merged_text_mask = transparent_img(merged_bbox.size)
        for text_mask, bbox in zip(text_masks, text_mask_bboxes):
            merged_text_mask.paste(text_mask, bbox.left_top)
        
        # 创建纯净的merged mask（不含effects）
        pure_merged_mask = transparent_img(merged_bbox.size)
        for pure_mask, bbox in zip(pure_text_masks, text_mask_bboxes):
            pure_merged_mask.paste(pure_mask, bbox.left_top)

        if self.cfg.perspective_transform is not None:
            transformer = PerspectiveTransform(self.cfg.perspective_transform)
            # TODO: refactor this, now we must call get_transformed_size to call gen_warp_matrix
            _ = transformer.get_transformed_size(merged_text_mask.size)

            (
                transformed_text_mask,
                transformed_text_pnts,
            ) = transformer.do_warp_perspective(merged_text_mask)
            
            # 对纯净mask也应用perspective transform
            pure_transformed_mask, _ = transformer.do_warp_perspective(pure_merged_mask)
        else:
            transformed_text_mask = merged_text_mask
            pure_transformed_mask = pure_merged_mask

        if self.cfg.layout_effects is not None:
            transformed_text_mask, _ = self.cfg.layout_effects.apply_effects(
                transformed_text_mask, BBox.from_size(transformed_text_mask.size)
            )

        img, cropped_bg = self.paste_text_mask_on_bg(bg, transformed_text_mask)

        return img, merged_text, cropped_bg, transformed_text_mask, pure_transformed_mask

    def paste_text_mask_on_bg(
        self, bg: PILImage, transformed_text_mask: PILImage
    ) -> Tuple[PILImage, PILImage]:
        """

        Args:
            bg:
            transformed_text_mask:

        Returns:

        """
        x_offset, y_offset = utils.random_xy_offset(transformed_text_mask.size, bg.size)
        bg = self.bg_manager.guard_bg_size(bg, transformed_text_mask.size)
        bg = bg.crop(
            (
                x_offset,
                y_offset,
                x_offset + transformed_text_mask.width,
                y_offset + transformed_text_mask.height,
            )
        )
        if self.cfg.return_bg_and_mask:
            _bg = bg.copy()
        else:
            _bg = bg
        bg.paste(transformed_text_mask, (0, 0), mask=transformed_text_mask)
        return bg, _bg

    def get_text_color(self, bg: PILImage, text: str, font: FreeTypeFont) -> FontColor:
        # TODO: better get text color
        # text_mask = self.draw_text_on_transparent_bg(text, font)
        np_img = np.array(bg)
        # mean = np.mean(np_img, axis=2)
        mean = np.mean(np_img)

        alpha = np.random.randint(110, 255)
        r = np.random.randint(0, int(mean * 0.7))
        g = np.random.randint(0, int(mean * 0.7))
        b = np.random.randint(0, int(mean * 0.7))
        fg_text_color = (r, g, b, alpha)

        return fg_text_color

    def _should_apply_layout(self) -> bool:
        return isinstance(self.corpus, list) and len(self.corpus) > 1

    def norm(self, image: np.ndarray) -> np.ndarray:
        if self.cfg.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.cfg.height != -1 and self.cfg.height != image.shape[0]:
            height, width = image.shape[:2]
            width = int(width // (height / self.cfg.height))
            image = cv2.resize(
                image, (width, self.cfg.height), interpolation=cv2.INTER_CUBIC
            )

        return image
