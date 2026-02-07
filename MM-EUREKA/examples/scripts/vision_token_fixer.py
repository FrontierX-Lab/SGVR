"""
视觉标记修复工具
用于修复token序列中视觉标记不匹配的问题，避免AssertionError
"""

import re
from typing import List, Tuple, Optional


def fix_vision_tokens_in_text(text: str) -> str:
    """
    修复文本中的视觉标记不匹配问题
    
    Args:
        text: 包含视觉标记的文本
        
    Returns:
        修复后的文本
    """
    if not text:
        return text
    
    # 统计视觉标记数量
    vision_start_count = text.count('<|vision_start|>')
    vision_end_count = text.count('<|vision_end|>')
    
    # 如果标记数量匹配，直接返回
    if vision_start_count == vision_end_count:
        return text
    
    print(f"检测到视觉标记不匹配: start={vision_start_count}, end={vision_end_count}")
    
    # 如果start标记多于end标记，添加缺失的end标记
    if vision_start_count > vision_end_count:
        missing_end_count = vision_start_count - vision_end_count
        # 在最后一个start标记后添加缺失的end标记
        for _ in range(missing_end_count):
            text += '<|vision_end|>'
        print(f"添加了{missing_end_count}个缺失的<|vision_end|>标记")
    
    # 如果end标记多于start标记，移除多余的end标记
    elif vision_end_count > vision_start_count:
        excess_end_count = vision_end_count - vision_start_count
        # 从后往前移除多余的end标记
        for _ in range(excess_end_count):
            last_end_pos = text.rfind('<|vision_end|>')
            if last_end_pos != -1:
                text = text[:last_end_pos] + text[last_end_pos + len('<|vision_end|>'):]
        print(f"移除了{excess_end_count}个多余的<|vision_end|>标记")
    
    return text

def fix_vision_tokens_in_token_list(tokens: List[int], tokenizer) -> List[int]:
    """
    修复token列表中的视觉标记不匹配问题
    
    Args:
        tokens: token ID列表
        tokenizer: 用于解码和编码的tokenizer
        
    Returns:
        修复后的token列表
    """
    if not tokens:
        return tokens
    
    try:
        # 解码为文本
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        
        # 修复视觉标记
        fixed_text = fix_vision_tokens_in_text(text)
        
        # 重新编码为tokens
        fixed_tokens = tokenizer.encode(fixed_text, add_special_tokens=False)
        
        return fixed_tokens
    except Exception as e:
        print(f"修复token列表时出错: {e}")
        return tokens


def fix_vision_tokens_in_token_list(tokens: List[int], 
                                   vision_start_token: int, 
                                   vision_end_token: int) -> List[int]:
    """
    修复token列表中的视觉标记不匹配问题
    
    Args:
        tokens: token ID列表
        vision_start_token: 视觉开始标记的token ID
        vision_end_token: 视觉结束标记的token ID
        
    Returns:
        修复后的token列表
    """
    if not tokens:
        return tokens
    
    # 统计视觉标记数量
    start_count = tokens.count(vision_start_token)
    end_count = tokens.count(vision_end_token)
    
    # 如果标记数量匹配，直接返回
    if start_count == end_count:
        return tokens
    
    print(f"检测到token中视觉标记不匹配: start={start_count}, end={end_count}")
    
    # 如果start标记多于end标记，添加缺失的end标记
    if start_count > end_count:
        missing_end_count = start_count - end_count
        # 在最后一个start标记后添加缺失的end标记
        last_start_pos = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == vision_start_token:
                last_start_pos = i
                break
        
        if last_start_pos != -1:
            # 在最后一个start标记后插入end标记
            fixed_tokens = tokens[:last_start_pos + 1] + [vision_end_token] * missing_end_count + tokens[last_start_pos + 1:]
            print(f"添加了{missing_end_count}个缺失的视觉结束标记")
            return fixed_tokens
    
    # 如果end标记多于start标记，移除多余的end标记
    elif end_count > start_count:
        excess_end_count = end_count - start_count
        # 从后往前移除多余的end标记
        fixed_tokens = tokens.copy()
        removed_count = 0
        for i in range(len(fixed_tokens) - 1, -1, -1):
            if fixed_tokens[i] == vision_end_token and removed_count < excess_end_count:
                fixed_tokens.pop(i)
                removed_count += 1
        print(f"移除了{removed_count}个多余的视觉结束标记")
        return fixed_tokens
    
    return tokens


def validate_vision_tokens(text: str) -> Tuple[bool, int, int]:
    """
    验证视觉标记是否匹配
    
    Args:
        text: 包含视觉标记的文本
        
    Returns:
        (是否匹配, start标记数量, end标记数量)
    """
    if not text:
        return True, 0, 0
    
    start_count = text.count('<|vision_start|>')
    end_count = text.count('<|vision_end|>')
    
    return start_count == end_count, start_count, end_count


def validate_vision_tokens_in_list(tokens: List[int], 
                                  vision_start_token: int, 
                                  vision_end_token: int) -> Tuple[bool, int, int]:
    """
    验证token列表中的视觉标记是否匹配
    
    Args:
        tokens: token ID列表
        vision_start_token: 视觉开始标记的token ID
        vision_end_token: 视觉结束标记的token ID
        
    Returns:
        (是否匹配, start标记数量, end标记数量)
    """
    if not tokens:
        return True, 0, 0
    
    start_count = tokens.count(vision_start_token)
    end_count = tokens.count(vision_end_token)
    
    return start_count == end_count, start_count, end_count


def safe_process_with_vision_fix(process_func, *args, **kwargs):
    """
    安全地处理包含视觉标记的数据，自动修复标记不匹配问题
    
    Args:
        process_func: 要调用的处理函数
        *args, **kwargs: 传递给处理函数的参数
        
    Returns:
        处理函数的结果
    """
    try:
        return process_func(*args, **kwargs)
    except AssertionError as e:
        if "vision_start_num == vision_end_num" in str(e):
            print(f"捕获到视觉标记不匹配错误: {e}")
            print("建议在数据预处理阶段使用fix_vision_tokens_in_text()修复标记")
            # 返回默认值或重新抛出异常
            raise e
        else:
            raise e
    except Exception as e:
        raise e


# 使用示例
if __name__ == "__main__":
    # 测试文本修复
    test_text = "这是一个测试<|vision_start|>图像内容<|vision_start|>更多内容"
    print(f"原始文本: {test_text}")
    
    is_valid, start_count, end_count = validate_vision_tokens(test_text)
    print(f"标记验证: 匹配={is_valid}, start={start_count}, end={end_count}")
    
    fixed_text = fix_vision_tokens_in_text(test_text)
    print(f"修复后文本: {fixed_text}")
    
    is_valid_after, start_count_after, end_count_after = validate_vision_tokens(fixed_text)
    print(f"修复后验证: 匹配={is_valid_after}, start={start_count_after}, end={end_count_after}")
    
    # 测试token修复
    test_tokens = [1, 2, 151644, 3, 4, 151644, 5, 6]  # 假设151644是vision_start_token
    vision_start = 151644
    vision_end = 151645  # 假设151645是vision_end_token
    
    print(f"\n原始tokens: {test_tokens}")
    is_valid_tokens, start_count_tokens, end_count_tokens = validate_vision_tokens_in_list(
        test_tokens, vision_start, vision_end)
    print(f"Token验证: 匹配={is_valid_tokens}, start={start_count_tokens}, end={end_count_tokens}")
    
    fixed_tokens = fix_vision_tokens_in_token_list(test_tokens, vision_start, vision_end)
    print(f"修复后tokens: {fixed_tokens}")
