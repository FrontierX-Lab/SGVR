from typing import Dict, List

import torch
from qwen_vl_utils import process_vision_info

from ..base.data_processor import BaseDataProcessor


class Qwen2_5_VLDataProcessor(BaseDataProcessor):
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
        # collect all keys
        for inp in inputs:
            batch.update({k: None for k, v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
            elif k in ["pixel_values", "image_grid_thw"]:
                # qwen2vl concat all patches of all images in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in inputs if k in inp], dim=0)
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
                
                # 检测视觉标记不匹配问题，直接跳过而不是修复
                if vision_start_num != vision_end_num:
                    # 获取更多调试信息
                    input_ids_str = input_ids_i.tolist() if isinstance(input_ids_i, torch.Tensor) else input_ids_i
                    vision_start_positions = [j for j, token_id in enumerate(input_ids_str) if token_id == vision_start_id]
                    vision_end_positions = [j for j, token_id in enumerate(input_ids_str) if token_id == vision_end_id]
                    
                    # 尝试解码部分token来获取更多上下文
                    try:
                        # 解码input_ids的前100个token和后100个token
                        if len(input_ids_str) > 200:
                            context_start = self.processor.tokenizer.decode(input_ids_str[:100], skip_special_tokens=False)
                            context_end = self.processor.tokenizer.decode(input_ids_str[-100:], skip_special_tokens=False)
                        else:
                            context_start = self.processor.tokenizer.decode(input_ids_str, skip_special_tokens=False)
                            context_end = ""
                        
                        print(f"警告: 样本{i}视觉标记不匹配 (start={vision_start_num}, end={vision_end_num})，跳过此样本")
                        print(f"  视觉标记位置: start={vision_start_positions}, end={vision_end_positions}")
                        print(f"  输入长度: {len(input_ids_str)} tokens")
                        print(f"  前100 tokens: {context_start[:200]}...")
                        if context_end:
                            print(f"  后100 tokens: ...{context_end[-200:]}")
                    except Exception as e:
                        print(f"警告: 样本{i}视觉标记不匹配 (start={vision_start_num}, end={vision_end_num})，跳过此样本")
                        print(f"  无法解码token上下文: {e}")
                    
                    batch_kwargs[i]["pixel_values"] = None
                    batch_kwargs[i]["image_grid_thw"] = None
                    # 需要消费掉对应的thws和pixel_values，避免后续断言失败
                    if vision_start_num > 0:
                        # 计算需要跳过的thws数量
                        thws_to_skip = thws[:vision_start_num]
                        thws = thws[vision_start_num:]
                        # 计算需要跳过的pixel_values数量
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

    def _fix_vision_tokens_in_tensor(self, input_ids: torch.Tensor, vision_start_id: int, vision_end_id: int) -> torch.Tensor:
        """
        修复tensor中的视觉标记不匹配问题
        
        Args:
            input_ids: 输入token序列
            vision_start_id: 视觉开始标记的token ID
            vision_end_id: 视觉结束标记的token ID
            
        Returns:
            修复后的token序列
        """
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        
        # 统计视觉标记数量
        start_count = (input_ids == vision_start_id).sum().item()
        end_count = (input_ids == vision_end_id).sum().item()
        
        # 如果标记数量匹配，直接返回
        if start_count == end_count:
            return input_ids
        
        # 转换为列表进行修复
        tokens_list = input_ids.tolist()
        
        # 如果start标记多于end标记，添加缺失的end标记
        if start_count > end_count:
            missing_end_count = start_count - end_count
            # 找到最后一个start标记的位置
            last_start_pos = -1
            for i in range(len(tokens_list) - 1, -1, -1):
                if tokens_list[i] == vision_start_id:
                    last_start_pos = i
                    break
            
            if last_start_pos != -1:
                # 在最后一个start标记后插入end标记
                tokens_list = tokens_list[:last_start_pos + 1] + [vision_end_id] * missing_end_count + tokens_list[last_start_pos + 1:]
                print(f"添加了{missing_end_count}个缺失的视觉结束标记")
        
        # 如果end标记多于start标记，移除多余的end标记
        elif end_count > start_count:
            excess_end_count = end_count - start_count
            # 从后往前移除多余的end标记
            removed_count = 0
            for i in range(len(tokens_list) - 1, -1, -1):
                if tokens_list[i] == vision_end_id and removed_count < excess_end_count:
                    tokens_list.pop(i)
                    removed_count += 1
            print(f"移除了{removed_count}个多余的视觉结束标记")
        
        return torch.tensor(tokens_list, dtype=input_ids.dtype, device=input_ids.device)


DataProcessor = Qwen2_5_VLDataProcessor

__all__ = ["DataProcessor"]
