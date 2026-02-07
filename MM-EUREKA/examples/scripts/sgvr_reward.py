import os
import sys
import re
import time
import traceback
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json
import openai
import math



from vision_token_fixer import validate_vision_tokens

FORMAT_REWARD_WEIGHT = 0.5      
LINK_ACCURACY_REWARD_WEIGHT = 1.0  
REPETITION_PENALTY_WEIGHT = 0.0 
FINAL_GOAL_REWARD_WEIGHT = 0.0  
# =====================================================

# ==================== API Validation Configuration ====================
ENABLE_SUBGOAL_COUNT_CHECK = False  # Whether to enable subgoal count checking
MAX_RETRY_ATTEMPTS = 0  # Maximum retry attempts (when count mismatch occurs)
# =====================================================

# API configuration - must be set via environment variables
INFER_BASE_URL = os.environ.get("INFER_BASE_URL")
INFER_API_KEY = os.environ.get("INFER_API_KEY")

if not INFER_BASE_URL:
    raise ValueError("INFER_BASE_URL environment variable is required")
if not INFER_API_KEY:
    raise ValueError("INFER_API_KEY environment variable is required")

# Log paths - must be set via environment variables
LOG_PATH = os.environ.get("REWARD_LOG_PATH")
DEBUG_LOG_PATH = os.environ.get("DEBUG_LOG_PATH")

if not LOG_PATH:
    raise ValueError("REWARD_LOG_PATH environment variable is required")
if not DEBUG_LOG_PATH:
    raise ValueError("DEBUG_LOG_PATH environment variable is required")







def safe_tensor_creation(data, dtype=torch.float32):
    """Safely create tensor, handling empty data"""
    if not data:
        return torch.tensor([], dtype=dtype)
    return torch.tensor(data, dtype=dtype)

def generate_sample_fingerprint(prompt: str, label: str) -> str:
    """Generate unique fingerprint for a sample, used to locate problematic samples in dataset"""
    import hashlib
    
    # Use first 100 characters of prompt and label to generate fingerprint
    prompt_snippet = prompt[:100] if prompt else ""
    label_snippet = label[:100] if label else ""
    
    # Generate MD5 hash as unique identifier
    content = f"{prompt_snippet}|{label_snippet}"
    fingerprint = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    return fingerprint

def find_problem_sample_in_dataset(dataset_path: str, target_fingerprint: str) -> dict:
    """
    Find problematic sample with specified fingerprint in dataset
    
    Args:
        dataset_path: Path to dataset file
        target_fingerprint: Target sample fingerprint
        
    Returns:
        Dictionary containing found sample information
    """
    import json
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for i, sample in enumerate(data):
            # Extract prompt and label
            if 'message' in sample:
                message = sample['message']
                if isinstance(message, str):
                    try:
                        message_data = json.loads(message)
                        # Build prompt and label
                        prompt_parts = []
                        for item in message_data:
                            if item.get('role') == 'user':
                                content = item.get('content', '')
                                if isinstance(content, list):
                                    for content_item in content:
                                        if content_item.get('type') == 'text':
                                            prompt_parts.append(content_item.get('text', ''))
                                elif isinstance(content, str):
                                    prompt_parts.append(content)
                        
                        prompt = ' '.join(prompt_parts)
                        label = sample.get('answer', '')
                        
                        # Generate fingerprint and compare
                        fingerprint = generate_sample_fingerprint(prompt, label)
                        if fingerprint == target_fingerprint:
                            return {
                                "found": True,
                                "index": i,
                                "sample": sample,
                                "prompt": prompt,
                                "label": label,
                                "fingerprint": fingerprint
                            }
                    except Exception as e:
                        continue
        
        return {"found": False, "message": "No matching sample found"}
        
    except Exception as e:
        return {"found": False, "error": str(e)}

def write_debug_log(message: str, sample_id: int = None, query: str = None, response: str = None, 
                   prompt: str = None, label: str = None, additional_info: dict = None):
    """Write debug log, recording debugging information and related samples"""
    try:
        with open(DEBUG_LOG_PATH, "a+", encoding='utf-8') as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            f.write(f"[{current_time}] {message}\n")
            
            if sample_id is not None:
                f.write(f"  Sample ID: {sample_id}\n")
            
            # Generate sample fingerprint for tracking
            if prompt is not None and label is not None:
                fingerprint = generate_sample_fingerprint(prompt, label)
                f.write(f"  Sample Fingerprint: {fingerprint}\n")
            
            if query is not None:
                f.write(f"  Query Length: {len(query) if query else 0} characters\n")
                f.write(f"  Query Content: {query[:200]}{'...' if len(query) > 200 else ''}\n")
            
            if prompt is not None:
                f.write(f"  Prompt Length: {len(prompt) if prompt else 0} characters\n")
                f.write(f"  Prompt Content: {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n")
            
            if label is not None:
                f.write(f"  Label Length: {len(label) if label else 0} characters\n")
                f.write(f"  Label Content: {label[:200]}{'...' if len(label) > 200 else ''}\n")
            
            if response is not None:
                f.write(f"  Response Length: {len(response) if response else 0} characters\n")
                f.write(f"  Response Content: {response[:200]}{'...' if len(response) > 200 else ''}\n")
            
            if additional_info:
                for key, value in additional_info.items():
                    f.write(f"  {key}: {value}\n")
            
            f.write("  " + "="*50 + "\n")
    except Exception as e:
        print(f"Failed to write debug log: {e}")

def _clean_and_validate_input(data):
    """Clean and validate input data, remove None values and invalid content"""
    if data is None:
        return ""
    
    if not isinstance(data, str):
        try:
            data = str(data)
        except Exception:
            return ""
    
    # Remove None strings
    data = data.replace("None", "")
    
    # Remove other possible invalid markers
    data = data.replace("null", "")
    data = data.replace("NULL", "")
    

    return data

def _call_eval_model(prompt: str, model: str = "gpt-5-nano", max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """Call specified model for evaluation (plain text), return raw string result"""
    client = openai.OpenAI(api_key=INFER_API_KEY, base_url=INFER_BASE_URL)
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,  # Support for specifying model
                messages=[{"role": "user", "content": prompt}],
                timeout=60_000,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            time.sleep(retry_delay)


def _extract_json_block(text: str):
    """Extract JSON block from text"""
    if not text:
        return None
    try:
        # Remove common markdown code block wrappers
        cleaned = text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        # Try to locate first JSON object
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception:
        return None

def _escape_braces(text):
    """Escape braces"""
    return text.replace("{", "{{").replace("}", "}}")

def safe_format(template: str, **kwargs) -> str:
    """Safe string formatting, handling brace escaping"""
    try:
        # First escape braces in template
        escaped_template = _escape_braces(template)
        # Then replace placeholders
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            escaped_template = escaped_template.replace(placeholder, str(value))
        return escaped_template
    except Exception as e:
        print(f"Formatting error: {e}")
        return template



# - For solving problems (where a specific numerical answer or value is required): The final answer must be correct or mathematically equivalent to the reference answer. If the student's final answer is incorrect, significantly reduce the score even if the proof steps are mostly correct.



def _get_groundtruth_subgoal_count(groundtruth_solution: str) -> int:
    """Get subgoal count from ground truth by code parsing
    Parse from solution field, format is usually a list like ['1', '0', '1'], or a single value or empty
    """
    if not groundtruth_solution:
        return 0
    
    try:
        # Clean solution string, remove possible whitespace
        solution_str = groundtruth_solution.strip()
        
        # Try to parse as Python list (using ast.literal_eval is safer)
        import ast
        try:
            # Try using ast.literal_eval for parsing (safer)
            parsed = ast.literal_eval(solution_str)
            
            if isinstance(parsed, list):
                # If it's a list, return list length
                count = len(parsed)
                print(f"DEBUG: Parsed ground truth subgoal count (list): {count}")
                return count
            elif parsed is not None:
                # If it's a single value (not a list), return 1
                print(f"DEBUG: Parsed ground truth subgoal count (single value): 1")
                return 1
            else:
                # If it's None or empty, return 0
                print(f"DEBUG: Parsed ground truth subgoal count (empty): 0")
                return 0
                
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, try using eval (less safe but more compatible)
            try:
                parsed = eval(solution_str)
                
                if isinstance(parsed, list):
                    count = len(parsed)
                    print(f"DEBUG: Parsed ground truth subgoal count (list, using eval): {count}")
                    return count
                elif parsed is not None:
                    print(f"DEBUG: Parsed ground truth subgoal count (single value, using eval): 1")
                    return 1
                else:
                    print(f"DEBUG: Parsed ground truth subgoal count (empty, using eval): 0")
                    return 0
                    
            except Exception as eval_error:
                print(f"DEBUG: Failed to parse solution, trying regex matching: {eval_error}")
                # If eval also fails, try regex matching for list format
                # Match formats like ['1', '0', '1'] or ["1", "0", "1"]
                list_match = re.search(r'\[(.*?)\]', solution_str)
                if list_match:
                    # Extract list content, count comma-separated elements
                    list_content = list_match.group(1).strip()
                    if not list_content:
                        # Empty list
                        print(f"DEBUG: Parsed ground truth subgoal count (empty list): 0")
                        return 0
                    else:
                        # Count elements (separated by commas, but be careful of commas in strings)
                        # Simple method: count quote pairs
                        quoted_items = re.findall(r'["\']([^"\']*)["\']', list_content)
                        if quoted_items:
                            count = len(quoted_items)
                            print(f"DEBUG: Parsed ground truth subgoal count (regex matched list): {count}")
                            return count
                        else:
                            # If no quotes, split by comma
                            items = [item.strip() for item in list_content.split(',') if item.strip()]
                            count = len(items) if items else 0
                            print(f"DEBUG: Parsed ground truth subgoal count (regex matched, no quotes): {count}")
                            return count if count > 0 else 1  # If no separator found, might be a single value
                
                # If nothing matches, might be a single value, return 1
                print(f"DEBUG: Parsed ground truth subgoal count (default single value): 1")
                return 1
        
    except Exception as e:
        print(f"Failed to parse ground truth subgoal count: {e}")
        import traceback
        print(traceback.format_exc())
        # Return 0 on error
        return 0

def combined_accuracy_reward_func(student_answer: str, groundtruth_solution: str) -> tuple:
    """Combined accuracy reward calculation, returns true/false labels for each subgoal and final goal result"""
    if not student_answer or not groundtruth_solution:
        print(f"DEBUG: combined_accuracy_reward_func - Empty input: student_answer={student_answer}, groundtruth_solution={groundtruth_solution}")
        return [], 0.0
    
    try:
        print(f"DEBUG: combined_accuracy_reward_func - Starting API call")
        
        # Get ground truth subgoal count (if checking is enabled)
        gt_subgoal_count = 0
        if ENABLE_SUBGOAL_COUNT_CHECK:
            gt_subgoal_count = _get_groundtruth_subgoal_count(groundtruth_solution)
            print(f"DEBUG: Ground truth subgoal count: {gt_subgoal_count}")
        
        # Build new evaluation prompt, return true/false labels for each subgoal
        # Choose different template based on whether checking is enabled
        if ENABLE_SUBGOAL_COUNT_CHECK:
            # New template: explicitly state it's a template, don't copy example values
            count_hint = ""
            if gt_subgoal_count > 0:
                count_hint = f"\n\nCRITICAL: The ground truth solution contains EXACTLY {gt_subgoal_count} sub-goals. You MUST return exactly {gt_subgoal_count} items in the \"subgoal_results\" array, with indices from 0 to {gt_subgoal_count - 1}. Do NOT copy the example values below - you must analyze the actual content."
            
            combined_eval_prompt = f"""You are given two solutions to a geometry problem that requires numerical answers. Your task is to evaluate each sub-goal in the student's answer.

Output STRICT JSON only with the schema below:

{{
  "subgoal_results": [
    {{"index": 0, "correct": true}},
    {{"index": 1, "correct": false}},
    {{"index": 2, "correct": true}}
  ],
  "final_answer_correct": boolean
}}

IMPORTANT: The above is a TEMPLATE showing the format only. DO NOT copy these example values!
- "index" must be integers starting from 0, 1, 2, 3, ... (one for each sub-goal you find in the ground truth)
- "correct" must be either true or false (boolean values, not strings)
- You must count ALL sub-goals in the ground truth solution and return exactly that many items
- The number of items in "subgoal_results" MUST match the actual number of sub-goals in the ground truth solution{count_hint}

Rules for Sub-goal Evaluation:
- Extract each numerical answer or conclusion from the ground truth solution
- For each sub-goal, check if the student's corresponding answer is mathematically equivalent
- Consider mathematical equivalence (e.g., 1/2 = 0.5, √2 ≈ 1.414, 2√2 ≈ 2.828)
- Accept both exact forms (like √2) and decimal approximations (like 1.414)
- Accept both fraction forms (like 1/2) and decimal forms (like 0.5)
- Use tolerance of 0.02 for decimal comparisons
- For each sub-goal, return true if correct, false if incorrect
- IMPORTANT: The number of sub-goals in "subgoal_results" must match the total number of sub-goals in the ground truth solution

Rules for Final Goal:
- Extract the final answer from both the student's solution and the ground truth solution
- The final answer is typically the last numerical result or conclusion
- Consider mathematical equivalence (e.g., 1/2 = 0.5, √2 ≈ 1.414, 2√2 ≈ 2.828)
- Accept both exact forms (like √2) and decimal approximations (like 1.414)
- Accept both fraction forms (like 1/2) and decimal forms (like 0.5)
- Use tolerance of 0.02 for decimal comparisons
- Return true if the final answers match, false otherwise

Student Answer:
{student_answer}

Ground Truth Solution:
{groundtruth_solution}

Please analyze:
1. Extract all sub-goals from the ground truth solution
2. For each sub-goal, check if the student's corresponding answer is correct
3. Extract the final answer from both solutions and compare
4. Provide the results in the JSON format above"""
        else:
            # Old template: used when checking is not enabled
            combined_eval_prompt = """You are given two solutions to a geometry problem that requires numerical answers. Your task is to evaluate each sub-goal in the student's answer.

Output STRICT JSON only with the schema below:

{{
  "subgoal_results": [
    {{"index": 0, "correct": true}},
    {{"index": 1, "correct": false}},
    {{"index": 2, "correct": true}}
  ],
  "final_answer_correct": boolean
}}

Rules for Sub-goal Evaluation:
- Extract each numerical answer or conclusion from the ground truth solution
- For each sub-goal, check if the student's corresponding answer is mathematically equivalent
- Consider mathematical equivalence (e.g., 1/2 = 0.5, √2 ≈ 1.414, 2√2 ≈ 2.828)
- Accept both exact forms (like √2) and decimal approximations (like 1.414)
- Accept both fraction forms (like 1/2) and decimal forms (like 0.5)
- Use tolerance of 0.02 for decimal comparisons
- For each sub-goal, return true if correct, false if incorrect
- IMPORTANT: The number of sub-goals in "subgoal_results" must match the total number of sub-goals in the ground truth solution

Rules for Final Goal:
- Extract the final answer from both the student's solution and the ground truth solution
- The final answer is typically the last numerical result or conclusion
- Consider mathematical equivalence (e.g., 1/2 = 0.5, √2 ≈ 1.414, 2√2 ≈ 2.828)
- Accept both exact forms (like √2) and decimal approximations (like 1.414)
- Accept both fraction forms (like 1/2) and decimal forms (like 0.5)
- Use tolerance of 0.02 for decimal comparisons
- Return true if the final answers match, false otherwise

Student Answer:
{student_answer}

Ground Truth Solution:
{groundtruth_solution}

Please analyze:
1. Extract all sub-goals from the ground truth solution
2. For each sub-goal, check if the student's corresponding answer is correct
3. Extract the final answer from both solutions and compare
4. Provide the results in the JSON format above"""

        # Format prompt
        prompt = safe_format(
            combined_eval_prompt,
            student_answer=_escape_braces(student_answer),
            groundtruth_solution=_escape_braces(groundtruth_solution)
        )
        
        # Call API to get score (using configured max retry attempts)
        max_retries = MAX_RETRY_ATTEMPTS
        subgoal_results = []
        final_goal_score = 0.0
        content = ""
        
        for attempt in range(max_retries):
            content = _call_eval_model(prompt, model="gpt-5-nano")
            print(f"DEBUG: combined_accuracy_reward_func API returned content (attempt {attempt + 1}/{max_retries}): {content}")
            
            # Parse score
            eval_json = _extract_json_block(content)
            print(f"DEBUG: combined_accuracy_reward_func parsed JSON: {eval_json}")
            
            subgoal_results = []
            final_goal_score = 0.0
            
            if isinstance(eval_json, dict):
                try:
                    # Parse subgoal results
                    subgoals = eval_json.get('subgoal_results', [])
                    if isinstance(subgoals, list):
                        subgoal_results = subgoals
                    
                    # Parse final goal score
                    final_correct = eval_json.get('final_answer_correct', False)
                    if isinstance(final_correct, bool):
                        final_goal_score = 1.0 if final_correct else 0.0
                        
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
            
            # If JSON parsing fails, try extracting from text
            if not subgoal_results and isinstance(content, str):
                # Try extracting subgoal results
                subgoal_match = re.search(r'"subgoal_results"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                if subgoal_match:
                    try:
                        subgoal_str = "[" + subgoal_match.group(1) + "]"
                        subgoal_results = json.loads(subgoal_str)
                    except:
                        pass
            
            # Check if index count matches ground truth subgoal count (if checking is enabled)
            if ENABLE_SUBGOAL_COUNT_CHECK and gt_subgoal_count > 0 and subgoal_results:
                returned_count = len(subgoal_results)
                if returned_count != gt_subgoal_count:
                    print(f"Warning: Returned index count ({returned_count}) does not match ground truth subgoal count ({gt_subgoal_count})")
                    
                    # If there are retry opportunities, modify prompt and retry
                    if attempt < max_retries - 1:
                        # Add reminder to prompt
                        retry_prompt = prompt + f"""

IMPORTANT CORRECTION NEEDED:
Your previous response had {returned_count} sub-goals in the "subgoal_results" array, but the ground truth solution actually contains {gt_subgoal_count} sub-goals. 
Please carefully re-analyze the ground truth solution and ensure that:
1. You extract ALL {gt_subgoal_count} sub-goals from the ground truth solution
2. The "subgoal_results" array contains exactly {gt_subgoal_count} items with indices from 0 to {gt_subgoal_count - 1}
3. Each sub-goal in the ground truth solution must have a corresponding entry in "subgoal_results"

Please provide a corrected JSON response with the correct number of sub-goals."""
                        prompt = retry_prompt
                        print(f"DEBUG: Re-calling API, added count mismatch reminder")
                        continue
                    else:
                        print(f"Warning: Reached maximum retry attempts ({max_retries}), returning current result")
                else:
                    # Count matches, return directly
                    print(f"DEBUG: Count check passed, returning result")
                    break
            else:
                # If checking is not enabled, or cannot get ground truth count, or no result returned, return directly
                if not ENABLE_SUBGOAL_COUNT_CHECK:
                    # Checking not enabled, return after first call
                    break
                elif gt_subgoal_count == 0:
                    # Cannot get ground truth count, return directly
                    print(f"DEBUG: Cannot get ground truth count, returning current result")
                    break
                elif not subgoal_results:
                    # No result returned, return directly
                    print(f"DEBUG: No subgoal_results obtained, returning current result")
                    break
        
        # If JSON parsing fails, try extracting final goal from text
        if not subgoal_results and isinstance(content, str):
            # Try extracting final goal
            if "true" in content.lower() or "correct" in content.lower():
                final_goal_score = 1.0
            elif "false" in content.lower() or "incorrect" in content.lower():
                final_goal_score = 0.0
        
        return subgoal_results, final_goal_score
            
    except Exception as e:
        print(f"combined_accuracy_reward_func error: {e}")
        return [], 0.0


def extract_proof_with_tags(text):
    """Extract content from <proof> tags in text"""
    if not text:
        return text
        
    # Find <proof> tags
    proof_pattern = re.compile(r'<proof>(.*?)</proof>', re.DOTALL | re.IGNORECASE)
    match = proof_pattern.search(text)
    
    if match:
        return match.group(1).strip()
    else:
        # <proof> tag not found, return original text
        return text

def extract_think_with_tags(text):
    """Extract content from <think> tags in text"""
    if not text:
        return ""
        
    # Find <think> tags
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    match = think_pattern.search(text)
    
    if match:
        return match.group(1).strip()
    else:
        return ""

def extract_answer_with_tags(text):
    """Extract content from <answer> tags in text"""
    if not text:
        return ""
        
    # Find <answer> tags
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
    match = answer_pattern.search(text)
    
    if match:
        return match.group(1).strip()
    else:
        return ""

def has_think_answer_format(text):
    """Check if text contains <think></think><answer></answer> format"""
    if not text:
        return False
    
    # Check if both think and answer tags are present
    has_think = "<think>" in text and "</think>" in text
    has_answer = "<answer>" in text and "</answer>" in text
    
    return has_think and has_answer

def calculate_repetition_penalty(text: str, penalty_weight: float = REPETITION_PENALTY_WEIGHT) -> float:
    """
    Calculate repetition penalty score
    Based on n-gram repetition rate to calculate penalty
    """
    if not text or penalty_weight <= 0:
        return 0.0
    
    def ngram_repetition_penalty(text, n):
        """Calculate n-gram repetition penalty"""
        words = text.strip().split()
        if len(words) < n:
            return 0.0
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        unique_ngrams = set(ngrams)
        repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
        return repetition_ratio
    
    penalties = []
    
    # Calculate 2-gram, 3-gram, 4-gram repetition penalties
    bigram_penalty = ngram_repetition_penalty(text.strip(), 2) * 0.3
    penalties.append(bigram_penalty)
    
    trigram_penalty = ngram_repetition_penalty(text.strip(), 3) * 0.4
    penalties.append(trigram_penalty)
    
    fourgram_penalty = ngram_repetition_penalty(text.strip(), 4) * 0.3
    penalties.append(fourgram_penalty)
    
    # Calculate total penalty score
    total_penalty = sum(penalties) * penalty_weight
    
    # Ensure it doesn't exceed maximum weight
    return min(total_penalty, penalty_weight)


def format_reward_func(completion):
    """Format reward: check if correct format tags are included"""
    if not completion:
        return 0.0
    
    # If FORMAT_REWARD_WEIGHT is not 0, check think+answer format
    if FORMAT_REWARD_WEIGHT > 0.0:
        try:
            if has_think_answer_format(completion):
                return 1.0 * FORMAT_REWARD_WEIGHT
            else:
                return 0.0
        except Exception as e:
            print(f"format_reward_func error: {e}")
            return 0.0
    else:
        return 0.0
    # else:
    #     # Keep original logic: check if proof tags are included
    #     if "<proof>" in completion and "</proof>" in completion:
    #         return 0.5
    #     else:
    #         return 0.0


def accuracy_reward_func(student_solution: str, groundtruth_solution: str) -> float:
    """Reward calculation based on final answer accuracy, using combined evaluation function"""
    if not student_solution or not groundtruth_solution:
        return 0.0
    
    try:
        # Use new combined evaluation function, only return link accuracy part
        subgoal_results, _ = combined_accuracy_reward_func(student_solution, groundtruth_solution)
        # Simple accuracy calculation
        if subgoal_results:
            correct_count = sum(1 for subgoal in subgoal_results if subgoal.get('correct', False))
            return correct_count / len(subgoal_results)
        return 0.0
    except Exception as e:
        print(f"accuracy_reward_func error: {e}")
        return 0.0

def extract_ref_proof_from_label(label: str) -> dict:
    """Extract ref_proof field from label"""
    try:
        # Find <ref_proof> tags
        ref_proof_match = re.search(r"<ref_proof>(.*?)</ref_proof>", label, re.DOTALL)
        if ref_proof_match:
            ref_proof_str = ref_proof_match.group(1).strip()
            # Try parsing as JSON
            try:
                return json.loads(ref_proof_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, try using eval to parse Python dict
                try:
                    # Use eval to parse Python dict string
                    result = eval(ref_proof_str)
                    if isinstance(result, dict):
                        return result
                    else:
                        print(f"ref_proof is not dict type: {type(result)}")
                        return {}
                except Exception as eval_error:
                    print(f"Python dict parsing failed: {eval_error}, content: {ref_proof_str[:100]}...")
                    return {}
        return {}
    except Exception as e:
        print(f"Failed to extract ref_proof: {e}")
        return {}

def extract_problem_from_label(label: str) -> str:
    """Extract problem field from label"""
    try:
        problem_match = re.search(r"<problem>(.*?)</problem>", label, re.DOTALL)
        if problem_match:
            return problem_match.group(1).strip()
        return ""
    except Exception as e:
        print(f"Failed to extract problem: {e}")
        return ""

def extract_solution_from_label(label: str) -> str:
    """Extract solution field from label"""
    try:
        solution_match = re.search(r"<solution>(.*?)</solution>", label, re.DOTALL)
        if solution_match:
            return solution_match.group(1).strip()
        return ""
    except Exception as e:
        print(f"Failed to extract solution: {e}")
        return ""

    


def get_response_from_query(q: str, sample_id: int = None):
    """Extract answer from query"""
    # Input validation
    if q is None:
        print("Warning: Query input is None, returning empty string")
        write_debug_log("Query input is None", sample_id, query=q,
                       additional_info={
                           "query_type": type(q).__name__,
                           "query_is_none": q is None
                       })
        return ""
    
    # Use unified cleaning function
    q = _clean_and_validate_input(q)
    
    if not q or not q.strip():
        print("Warning: Query input is empty, returning empty string")
        write_debug_log("Query input is empty", sample_id, query=q,
                       additional_info={
                           "original_query_length": len(q) if q else 0,
                           "cleaned_query_length": len(q.strip()) if q else 0,
                           "query_after_clean": q[:100] if q else ""
                       })
        return ""
    
    response_prefix = r"<\|im_start\|>assistant\n"
    ends_of_sentence = ["<|im_end|>", "", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return ""
    response = q[pos.end():]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    
    
    return response.strip()

def reward_func(queries, prompts, labels):
    """Reward calculation function, calculates format reward, ref_proof_reward and accuracy_reward"""
    # Input validation
    if not queries or not prompts or not labels:
        print("Warning: Input data is empty, returning empty result")
        write_debug_log("reward_func input data is empty", 
                       additional_info={
                           "queries_length": len(queries) if queries else 0,
                           "prompts_length": len(prompts) if prompts else 0,
                           "labels_length": len(labels) if labels else 0,
                           "queries_type": type(queries).__name__,
                           "prompts_type": type(prompts).__name__,
                           "labels_type": type(labels).__name__
                       })
        return {"rewards": safe_tensor_creation([], torch.float32)}
    
    # Ensure all lists have consistent length
    min_length = min(len(queries), len(prompts), len(labels))
    if min_length == 0:
        print("Warning: Input data length is 0, returning empty result")
        return {"rewards": safe_tensor_creation([], torch.float32)}
    
    # Truncate to minimum length
    queries = queries[:min_length]
    prompts = prompts[:min_length]
    labels = labels[:min_length]
    
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    format_rewards = []
    link_accuracy_rewards = []
    final_goal_rewards = []
    repetition_penalties = []
    
    with open(LOG_PATH, "a+") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for i, (query, prompt, label) in enumerate(zip(queries, prompts, labels)):
            # Generate sample fingerprint for tracking
            sample_fingerprint = generate_sample_fingerprint(prompt, label)
            
            try:
                # Input validation
                if query is None or not query.strip():
                    print(f"Warning: Sample {i} query is empty or None, skipping")
                    write_debug_log(f"Sample {i} query is empty or None", 
                                 sample_id=i, query=query, prompt=prompt, label=label,
                                 additional_info={
                                     "sample_fingerprint": sample_fingerprint,
                                     "query_type": type(query).__name__,
                                     "query_is_none": query is None,
                                     "query_stripped_length": len(query.strip()) if query else 0,
                                     "prompt_length": len(prompt) if prompt else 0,
                                     "label_length": len(label) if label else 0
                                 })
                    rewards.append(0.0)
                    format_rewards.append(0.0)
                    link_accuracy_rewards.append(0.0)
                    final_goal_rewards.append(0.0)
                    repetition_penalties.append(0.0)
                    continue
                
                # Clean input
                prompt = _clean_and_validate_input(prompt)
                label = _clean_and_validate_input(label)
                
                # Additional validation: ensure prompt and label are not empty
                if not prompt.strip() or not label.strip():
                    print(f"Warning: Sample {i} prompt or label is empty, skipping")
                    write_debug_log(f"Sample {i} prompt or label is empty", 
                                 sample_id=i, query=query, prompt=prompt, label=label,
                                 additional_info={
                                     "prompt_stripped_length": len(prompt.strip()) if prompt else 0,
                                     "label_stripped_length": len(label.strip()) if label else 0,
                                     "prompt_is_empty": not prompt.strip() if prompt else True,
                                     "label_is_empty": not label.strip() if label else True
                                 })
                    rewards.append(0.0)
                    format_rewards.append(0.0)
                    link_accuracy_rewards.append(0.0)
                    final_goal_rewards.append(0.0)
                    repetition_penalties.append(0.0)
                    continue
                
                # Extract model answer from query
                response = get_response_from_query(query, sample_id=i)
                
                f.write("===========================================================\n")
                f.write(f"query---------------------------------------------------\n: {query}\n")
                f.write(f"prompt---------------------------------------------------\n: {prompt}\n")
                f.write(f"label---------------------------------------------------\n: {label}\n")
                f.write(f"response---------------------------------------------------\n: {response}\n")
                f.write("===========================================================\n")
                
                if not response:
                    f.write("Warning: Unable to extract response from query\n")
                    rewards.append(0.0)
                    format_rewards.append(0.0)
                    link_accuracy_rewards.append(0.0)
                    final_goal_rewards.append(0.0)
                    repetition_penalties.append(0.0)
                    continue
                
                try:
                    is_valid, start_count, end_count = validate_vision_tokens(response)
                    if not is_valid:
                        print(f"Warning: Sample {i} response has mismatched vision tokens (start={start_count}, end={end_count}), skipping")
                        f.write(f"Warning: Vision tokens mismatch (start={start_count}, end={end_count}), skipping sample\n")
                        write_debug_log(f"Vision tokens mismatch, skipping sample", 
                                     sample_id=i, query=query, response=response, prompt=prompt, label=label,
                                     additional_info={
                                         "sample_fingerprint": sample_fingerprint,
                                         "start_count": start_count, 
                                         "end_count": end_count,
                                         "skip_reason": "vision_tokens_mismatch"
                                     })
                        rewards.append(0.0)
                        format_rewards.append(0.0)
                        link_accuracy_rewards.append(0.0)
                        final_goal_rewards.append(0.0)
                        repetition_penalties.append(0.0)
                        continue
                except Exception as vision_error:
                    print(f"Vision token validation failed: {vision_error}, skipping sample")
                    f.write(f"Vision token validation failed: {vision_error}, skipping sample\n")
                    write_debug_log(f"Vision token validation failed, skipping sample", 
                                 sample_id=i, query=query, response=response, prompt=prompt, label=label,
                                 additional_info={
                                     "sample_fingerprint": sample_fingerprint,
                                     "error": str(vision_error),
                                     "skip_reason": "vision_validation_error"
                                 })
                    rewards.append(0.0)
                    format_rewards.append(0.0)
                    link_accuracy_rewards.append(0.0)
                    final_goal_rewards.append(0.0)
                    repetition_penalties.append(0.0)
                    continue
                
                # Calculate format reward
                format_reward = 0.0
                if FORMAT_REWARD_WEIGHT > 0.0:
                    try:
                        format_reward = format_reward_func(response)
                        # Ensure format_reward is not None
                        if format_reward is None:
                            format_reward = 0.0
                    except Exception as e:
                        print(f"Error calculating format_reward: {e}")
                        format_reward = 0.0
                
                # Calculate link accuracy reward and final goal reward (using new subgoal evaluation)
                link_accuracy_reward = 0.0
                final_goal_reward = 0.0
                
                # Only make API call if either needs to be calculated
                if LINK_ACCURACY_REWARD_WEIGHT > 0.0 or FINAL_GOAL_REWARD_WEIGHT > 0.0:
                    # Extract groundtruth solution
                    groundtruth_solution = extract_solution_from_label(label)
                    f.write(f"DEBUG: Extracted groundtruth_solution: {groundtruth_solution}\n")
                    if groundtruth_solution:
                        if FORMAT_REWARD_WEIGHT > 0.0:
                            # New format: choose different processing based on whether format is satisfied
                            if format_reward > 0:
                                # Format satisfied: use answer content for evaluation
                                answer_content = extract_answer_with_tags(response)
                                f.write(f"DEBUG: Extracted answer_content: {answer_content}\n")
                                if answer_content:
                                    subgoal_results, _ = combined_accuracy_reward_func(answer_content, groundtruth_solution)
                                    f.write(f"DEBUG: Combined API call result - answer_content: subgoal_results={subgoal_results}\n")
                                    # Manually calculate accuracy
                                    if subgoal_results:
                                        correct_count = sum(1 for subgoal in subgoal_results if subgoal.get('correct', False))
                                        link_accuracy_reward = correct_count / len(subgoal_results)
                                        # Use last index's true/false as final_goal_reward
                                        last_subgoal = subgoal_results[-1] if subgoal_results else {}
                                        final_goal_reward = 1.0 if last_subgoal.get('correct', False) else 0.0
                                    f.write(f"DEBUG: Manually calculated link_accuracy={link_accuracy_reward}, final_goal={final_goal_reward}\n")
                            else:
                                # Format not satisfied: use entire response for evaluation
                                f.write(f"DEBUG: Using entire response for combined API call\n")
                                subgoal_results, _ = combined_accuracy_reward_func(response, groundtruth_solution)
                                f.write(f"DEBUG: Combined API call result - response: subgoal_results={subgoal_results}\n")
                                # Manually calculate accuracy
                                if subgoal_results:
                                    correct_count = sum(1 for subgoal in subgoal_results if subgoal.get('correct', False))
                                    link_accuracy_reward = correct_count / len(subgoal_results)
                                    # Use last index's true/false as final_goal_reward
                                    last_subgoal = subgoal_results[-1] if subgoal_results else {}
                                    final_goal_reward = 1.0 if last_subgoal.get('correct', False) else 0.0
                                f.write(f"DEBUG: Manually calculated link_accuracy={link_accuracy_reward}, final_goal={final_goal_reward}\n")
                        else:
                            # Old format: use entire response for evaluation
                            f.write(f"DEBUG: Old format, using entire response for combined API call\n")
                            subgoal_results, _ = combined_accuracy_reward_func(response, groundtruth_solution)
                            f.write(f"DEBUG: Combined API call result - response: subgoal_results={subgoal_results}\n")
                            # Manually calculate accuracy
                            if subgoal_results:
                                correct_count = sum(1 for subgoal in subgoal_results if subgoal.get('correct', False))
                                link_accuracy_reward = correct_count / len(subgoal_results)
                                # Use last index's true/false as final_goal_reward
                                last_subgoal = subgoal_results[-1] if subgoal_results else {}
                                final_goal_reward = 1.0 if last_subgoal.get('correct', False) else 0.0
                            f.write(f"DEBUG: Manually calculated link_accuracy={link_accuracy_reward}, final_goal={final_goal_reward}\n")
                    else:
                        f.write(f"DEBUG: Groundtruth solution not found\n")
                
                
                # Calculate repetition penalty
                repetition_penalty = 0.0
                if REPETITION_PENALTY_WEIGHT > 0.0:
                    repetition_penalty = calculate_repetition_penalty(response)
                
                # Total reward = format reward * weight + link accuracy reward * weight + final goal reward * weight - repetition penalty
                total_reward = (format_reward * FORMAT_REWARD_WEIGHT + 
                              link_accuracy_reward * LINK_ACCURACY_REWARD_WEIGHT +
                              final_goal_reward * FINAL_GOAL_REWARD_WEIGHT -
                              repetition_penalty)
                
                rewards.append(float(total_reward))
                format_rewards.append(float(format_reward))
                link_accuracy_rewards.append(float(link_accuracy_reward))
                final_goal_rewards.append(float(final_goal_reward))
                repetition_penalties.append(float(repetition_penalty))
                
                # Log results
                f.write(f"===============================================================\n")
                f.write(f"Response Length: {len(response)} characters\n")
                f.write(f"Format Reward: {format_reward} (weight: {FORMAT_REWARD_WEIGHT})\n")
                f.write(f"Link Accuracy Reward: {link_accuracy_reward} (weight: {LINK_ACCURACY_REWARD_WEIGHT})\n")
                f.write(f"Final Goal Reward: {final_goal_reward} (weight: {FINAL_GOAL_REWARD_WEIGHT})\n")
                f.write(f"Repetition Penalty: {repetition_penalty} (weight: {REPETITION_PENALTY_WEIGHT})\n")
                f.write(f"Ground Truth Solution: {groundtruth_solution}\n")
                f.write(f"Total Reward: {total_reward}\n")
                f.write(f"===============================================================\n")
                print("Writing to file path:", LOG_PATH)
            except Exception as e:
                f.write(f"Error: {str(e)}\nDetailed error: {traceback.format_exc()}\n")
                rewards.append(0.0)
                format_rewards.append(0.0)
                link_accuracy_rewards.append(0.0)
                final_goal_rewards.append(0.0)
                repetition_penalties.append(0.0)
    
    # Organize return results
    result: Dict[str, Any] = {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
        "link_accuracy_rewards": torch.tensor(link_accuracy_rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(link_accuracy_rewards, dtype=torch.float32),  # Add accuracy_rewards key, same as link_accuracy_rewards
        "final_goal_rewards": torch.tensor(final_goal_rewards, dtype=torch.float32),  # Final goal reward
        "repetition_penalties": torch.tensor(repetition_penalties, dtype=torch.float32),  # Repetition penalty
    }
    
    return result