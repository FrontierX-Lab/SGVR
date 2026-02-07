from typing import Dict, List

import torch
from qwen_vl_utils import process_vision_info

from ..base.data_processor import BaseDataProcessor


class Qwen2_VLDataProcessor(BaseDataProcessor):
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
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

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
        # each element has no batch dimension
        batch = {}
        
        # 预检查：过滤掉有问题的样本（视觉标记不匹配或缺少图像数据）
        valid_inputs = []
        for i, inp in enumerate(inputs):
            if "input_ids" in inp and inp["input_ids"] is not None:
                input_ids = inp["input_ids"]
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids)
                vision_start_id = self.processor.tokenizer("<|vision_start|>")["input_ids"][0]
                vision_end_id = self.processor.tokenizer("<|vision_end|>")["input_ids"][0]
                vision_start_num = (input_ids == vision_start_id).sum().item()
                vision_end_num = (input_ids == vision_end_id).sum().item()
                if vision_start_num != vision_end_num:
                    print(f"警告: 样本{i}视觉标记不匹配 (start={vision_start_num}, end={vision_end_num})，在批处理前跳过")
                    continue
                if vision_start_num > 0:
                    has_pixel_values = "pixel_values" in inp and inp["pixel_values"] is not None
                    has_image_grid = "image_grid_thw" in inp and inp["image_grid_thw"] is not None
                    if not has_pixel_values or not has_image_grid:
                        print(f"警告: 样本{i}有视觉标记但缺少图像数据，在批处理前跳过")
                        continue
            valid_inputs.append(inp)
        if not valid_inputs:
            raise ValueError("所有样本都被过滤掉了，无法创建批处理")

        # collect all keys
        for inp in valid_inputs:
            batch.update({k: None for k, v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in valid_inputs if k in inp], dim=0)
            elif k in ["pixel_values", "image_grid_thw"]:
                # qwen2vl concat all patches of all images in a batch in the first dimension
                valid_inputs_for_k = [inp[k] for inp in valid_inputs if k in inp and inp[k] is not None]
                if valid_inputs_for_k:
                    batch[k] = torch.cat(valid_inputs_for_k, dim=0)
                else:
                    batch[k] = None
            else:
                raise ValueError(f"Unknown key {k} for Qwen2VLDataProcessor")
        return batch

    def split_input_batch(self, batch: Dict) -> List[Dict]:
        batch_size = len(batch["input_ids"])
        batch_kwargs = [{} for _ in range(batch_size)]
        # first process None values
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
                    print(f"错误: 样本{i}在split_input_batch中发现视觉标记不匹配，这不应该发生")
                    raise ValueError(f"样本{i}视觉标记不匹配 (start={vision_start_num}, end={vision_end_num})")
                img_num = vision_start_num
                if img_num == 0:
                    batch_kwargs[i]["pixel_values"] = None
                    batch_kwargs[i]["image_grid_thw"] = None
                    continue
                thws_i = thws[:img_num]
                if len(thws_i) != img_num:
                    print(f"错误: 样本{i}在split_input_batch中发现图像数据不匹配，这不应该发生")
                    raise ValueError(f"样本{i}图像数据不匹配 (期望{img_num}个图像，实际{len(thws_i)}个)")
                thws = thws[img_num:]
                if not isinstance(thws_i, torch.Tensor):
                    thws_i = torch.stack(thws_i)
                batch_kwargs[i]["image_grid_thw"] = thws_i
                patchs_num = thws_i.prod(dim=1).sum().item()
                pixel_values_i = pixel_values[:patchs_num]
                if len(pixel_values_i) != patchs_num:
                    print(f"错误: 样本{i}在split_input_batch中发现pixel_values不匹配，这不应该发生")
                    raise ValueError(f"样本{i}pixel_values不匹配 (期望{patchs_num}个patches，实际{len(pixel_values_i)}个)")
                pixel_values = pixel_values[patchs_num:]
                batch_kwargs[i]["pixel_values"] = pixel_values_i
            if len(thws) > 0:
                print(f"警告: 批处理结束后仍有{len(thws)}个thws未使用")
            if pixel_values is not None and len(pixel_values) > 0:
                print(f"警告: 批处理结束后仍有{len(pixel_values)}个pixel_values未使用")
        return batch_kwargs


DataProcessor = Qwen2_VLDataProcessor

__all__ = ["DataProcessor"]


