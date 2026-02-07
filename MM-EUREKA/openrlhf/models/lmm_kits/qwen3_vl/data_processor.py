from typing import Dict, List

import torch

try:
    # 可用则使用，若旧版 transformers/qwen_vl_utils 不支持会在 except 路径回退
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:  # pragma: no cover - 兼容老版本环境
    process_vision_info = None  # type: ignore

from ..base.data_processor import BaseDataProcessor


class Qwen3_VLDataProcessor(BaseDataProcessor):
    def __call__(
        self,
        messages,
        max_length,
        padding=True,
        device=None,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    ) -> Dict:
        messages = self._format_messages(messages)
        processor = self.processor
        texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs = None
        video_inputs = None
        video_kwargs = {}

        # 优先使用 qwen_vl_utils（如可用），否则由 AutoProcessor 直接处理
        if process_vision_info is not None:
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)  # type: ignore
            except Exception:
                image_inputs, video_inputs, video_kwargs = None, None, {}

        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=padding,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_tensors=return_tensors,
            **video_kwargs,
        )
        if device:
            return {k: v.to(device) for k, v in batch.items()}
        return {k: v for k, v in batch.items()}

    def make_input_batch(self, inputs: List[Dict]) -> Dict:
        # 与 qwen2.5 一致的拼接策略；Qwen3-VL Processor 输出键名保持相同约定
        batch: Dict = {}
        for inp in inputs:
            batch.update({k: None for k, v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
            elif k in ["pixel_values", "image_grid_thw"]:
                batch[k] = torch.cat([inp[k] for inp in inputs if k in inp], dim=0)
            else:
                raise ValueError(f"Unknown key {k} for Qwen3_VLDataProcessor")
        return batch

    def split_input_batch(self, batch: Dict) -> List[Dict]:
        batch_size = len(batch["input_ids"])
        batch_kwargs: List[Dict] = [{} for _ in range(batch_size)]

        keys = []
        for k, v in batch.items():
            if v is not None:
                keys.append(k)
            else:
                for i in range(batch_size):
                    batch_kwargs[i][k] = None

        if "pixel_values" in keys and ("input_ids" not in keys or "image_grid_thw" not in keys):
            raise ValueError("Cannot split batch with pixel_values without input_ids and image_grid_thw")
        if "image_grid_thw" in keys and ("input_ids" not in keys):
            raise ValueError("Cannot split batch with image_grid_thw without input_ids")

        for k in ["input_ids", "attention_mask"]:
            if k in keys:
                vals = batch[k]
                if isinstance(vals, torch.Tensor):
                    vals = torch.unbind(vals)
                assert batch_size == len(vals)
                for i, v in enumerate(vals):
                    batch_kwargs[i][k] = v

        if "pixel_values" in keys:
            thws = batch["image_grid_thw"]  # (total_img_num, (t,h,w))
            pixel_values = batch["pixel_values"]
            vision_start_id = self.processor.tokenizer("<|vision_start|>")["input_ids"][0]
            vision_end_id = self.processor.tokenizer("<|vision_end|>")["input_ids"][0]
            for i in range(batch_size):
                input_ids_i = batch_kwargs[i]["input_ids"]
                if not isinstance(input_ids_i, torch.Tensor):
                    input_ids_i = torch.tensor(input_ids_i)
                vision_start_num = (input_ids_i == vision_start_id).sum().item()
                vision_end_num = (input_ids_i == vision_end_id).sum().item()

                if vision_start_num != vision_end_num:
                    batch_kwargs[i]["pixel_values"] = None
                    batch_kwargs[i]["image_grid_thw"] = None
                    if vision_start_num > 0:
                        thws_to_skip = thws[:vision_start_num]
                        thws = thws[vision_start_num:]
                        if len(thws_to_skip) > 0:
                            if not isinstance(thws_to_skip, torch.Tensor):
                                thws_to_skip = torch.stack(thws_to_skip)
                            patches_to_skip = thws_to_skip.prod(dim=1).sum().item()
                            pixel_values = pixel_values[patches_to_skip:]
                    continue

                img_num = vision_start_num
                if img_num == 0:
                    batch_kwargs[i]["pixel_values"] = None
                    batch_kwargs[i]["image_grid_thw"] = None
                    continue
                thws_i = thws[:img_num]
                assert len(thws_i) == img_num
                thws = thws[img_num:]
                if not isinstance(thws_i, torch.Tensor):
                    thws_i = torch.stack(thws_i)
                batch_kwargs[i]["image_grid_thw"] = thws_i
                patchs_num = thws_i.prod(dim=1).sum().item()
                pixel_values_i = pixel_values[:patchs_num]
                assert len(pixel_values_i) == patchs_num
                pixel_values = pixel_values[patchs_num:]
                batch_kwargs[i]["pixel_values"] = pixel_values_i
            assert len(thws) == 0
            assert len(pixel_values) == 0
        return batch_kwargs


DataProcessor = Qwen3_VLDataProcessor

__all__ = ["DataProcessor"]


