#!/usr/bin/env python3
"""
GSM8K Knowledge Graph Generator - 完整数据集版本
支持分批处理、断点续传、进度保存
使用 Qwen2.5-7B-Instruct 的正确 Chat 模板
"""

import json
import torch
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Set
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_best_device() -> str:
    """获取最佳可用设备：GPU > MPS > CPU"""
    if torch.cuda.is_available():
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("✓ 使用 MPS (Apple Silicon)")
        return "mps"
    else:
        print("✓ 使用 CPU")
        return "cpu"


class KnowledgeGraphGenerator:
    """知识图谱生成器 - 使用 Qwen Chat 模板"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.device = get_best_device()
        self.model_name = model_name
        
        print(f"\n📦 加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else self.device,
            trust_remote_code=True
        )
        print("✓ 模型加载完成\n")
    
    def generate(self, question_id: int, question: str, ground_truth: str, max_retries: int = 3) -> Optional[Dict]:
        """生成知识图谱，验证答案，错误时重试"""
        kg = None
        last_answer = None
        
        for attempt in range(max_retries):
            try:
                # 构建消息
                messages = self._build_messages(question, ground_truth if attempt > 0 else None, last_answer)
                
                # 使用 Qwen 的 chat 模板
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1500,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 只取生成的新内容
                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # 调试输出（仅第一次尝试）
                if attempt == 0:
                    print(f"    [调试] 输出前150字: {response[:150].replace(chr(10), ' ')}...")
                
                # 解析响应
                kg = self._parse_response(question_id, question, ground_truth, response)
                last_answer = kg["final_answer"]
                
                # 验证答案
                if self._validate_answer(kg["final_answer"], ground_truth):
                    kg["status"] = "correct"
                    kg["attempts"] = attempt + 1
                    return kg
                
                if attempt < max_retries - 1:
                    print(f"    ⚠️ 答案 '{kg['final_answer']}' 错误，正确 '{ground_truth}'，重试 {attempt + 2}/{max_retries}")
                    
            except Exception as e:
                print(f"    ❌ 错误: {str(e)[:80]}")
                if kg is None:
                    kg = {
                        "question_id": question_id,
                        "question": question,
                        "ground_truth_answer": ground_truth,
                        "cot": "",
                        "solution_steps": "",
                        "final_answer": "",
                        "problem_type": "",
                        "required_knowledge": ""
                    }
        
        if kg:
            kg["status"] = "incorrect"
            kg["attempts"] = max_retries
        return kg
    
    def _build_messages(self, question: str, correct_answer: str = None, last_answer: str = None) -> list:
        """构建 Qwen Chat 消息格式"""
        
        system_msg = """你是一位数学教育专家。请分析数学问题并生成结构化的知识图谱。

严格按照以下格式输出（使用【】标记每个部分）：

【链式思维】
逐步推理过程

【解题步骤】
详细计算过程

【最终答案】
只写一个数字

【问题类型】
问题分类

【所需知识】
需要的数学概念"""

        if correct_answer and last_answer:
            user_msg = f"""问题：{question}

你上次的答案 {last_answer} 是错误的。正确答案是 {correct_answer}。
请重新分析，【最终答案】必须是 {correct_answer}"""
        else:
            user_msg = f"""问题：{question}

请解答这道数学题。"""

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    
    def _parse_response(self, qid: int, question: str, ground_truth: str, response: str) -> Dict:
        """解析模型响应为结构化知识图谱"""
        
        def extract_section(text: str, section_name: str) -> str:
            """提取指定部分的内容"""
            pattern = rf"【{section_name}】\s*(.*?)(?=【|$)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
            return ""
        
        # 提取各部分
        cot = extract_section(response, "链式思维")
        steps = extract_section(response, "解题步骤")
        answer_text = extract_section(response, "最终答案")
        problem_type = extract_section(response, "问题类型")
        knowledge = extract_section(response, "所需知识")
        
        # 从答案文本中提取数字
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
        if numbers:
            final_answer = numbers[0]
        else:
            # 尝试从整个响应中查找答案模式
            patterns = [
                r'答案[是为：:\s]+(-?\d+(?:\.\d+)?)',
                r'等于\s*(-?\d+(?:\.\d+)?)',
                r'共[有是]\s*(-?\d+(?:\.\d+)?)',
                r'=\s*(-?\d+(?:\.\d+)?)\s*(?:元|个|页|岁|天|小时|分钟)?$',
            ]
            final_answer = ""
            for pat in patterns:
                match = re.search(pat, response, re.MULTILINE)
                if match:
                    final_answer = match.group(1)
                    break
            
            if not final_answer:
                # 最后尝试：取响应中最后一个数字
                all_nums = re.findall(r'-?\d+(?:\.\d+)?', response)
                if all_nums:
                    final_answer = all_nums[-1]
        
        return {
            "question_id": qid,
            "question": question,
            "ground_truth_answer": ground_truth,
            "cot": cot,
            "solution_steps": steps,
            "final_answer": final_answer,
            "problem_type": problem_type,
            "required_knowledge": knowledge
        }
    
    def _validate_answer(self, model_answer: str, ground_truth: str) -> bool:
        """验证答案是否正确"""
        try:
            model_nums = re.findall(r'-?\d+(?:\.\d+)?', str(model_answer))
            truth_nums = re.findall(r'-?\d+(?:\.\d+)?', str(ground_truth))
            
            if not model_nums or not truth_nums:
                return False
            
            model_num = float(model_nums[0])
            truth_num = float(truth_nums[0])
            
            return abs(model_num - truth_num) < 0.001
        except:
            return str(model_answer).strip() == str(ground_truth).strip()


class BatchProcessor:
    """分批处理器"""
    
    def __init__(self, output_dir: str = "output", batch_size: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.generator = None
        
    def load_processed_ids(self) -> Set[int]:
        """加载已处理的问题ID"""
        processed = set()
        for batch_file in self.output_dir.glob("batch_*.json"):
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for kg in data.get("knowledge_graphs", []):
                    processed.add(kg["question_id"])
        return processed
    
    def save_batch(self, batch_num: int, knowledge_graphs: list, stats: dict):
        """保存批次结果"""
        batch_file = self.output_dir / f"batch_{batch_num:04d}.json"
        result = {
            "batch_num": batch_num,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "knowledge_graphs": knowledge_graphs
        }
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  💾 已保存批次 {batch_num} -> {batch_file.name}")
    
    def merge_all_batches(self, output_file: str = "GSM8K_KG.json"):
        """合并所有批次"""
        print("\n📦 合并所有批次...")
        all_kgs = []
        batch_files = sorted(self.output_dir.glob("batch_*.json"))
        
        for batch_file in batch_files:
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_kgs.extend(data.get("knowledge_graphs", []))
        
        all_kgs.sort(key=lambda x: x["question_id"])
        
        correct = sum(1 for kg in all_kgs if kg["status"] == "correct")
        result = {
            "metadata": {
                "dataset": "GSM8K",
                "model": "Qwen2.5-7B-Instruct",
                "total": len(all_kgs),
                "correct": correct,
                "incorrect": len(all_kgs) - correct,
                "accuracy": f"{correct/len(all_kgs)*100:.2f}%" if all_kgs else "0%",
                "generated_at": datetime.now().isoformat()
            },
            "knowledge_graphs": all_kgs
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 合并完成: {output_path}")
        if all_kgs:
            print(f"📊 总计: {len(all_kgs)} | 正确: {correct} | 准确率: {correct/len(all_kgs)*100:.2f}%")
        return output_path
    
    def process_dataset(self, start_batch: int = 0, max_batches: int = None):
        """处理整个数据集"""
        print("=" * 70)
        print("GSM8K 知识图谱生成器 - Qwen2.5-7B-Instruct")
        print("=" * 70)
        
        self.generator = KnowledgeGraphGenerator()
        
        print("📥 加载 GSM8K 数据集...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        total_problems = len(dataset)
        print(f"✓ 数据集大小: {total_problems} 个问题")
        
        processed_ids = self.load_processed_ids()
        print(f"✓ 已处理: {len(processed_ids)} 个问题")
        
        total_batches = (total_problems + self.batch_size - 1) // self.batch_size
        if max_batches:
            total_batches = min(total_batches, start_batch + max_batches)
        
        print(f"✓ 批次大小: {self.batch_size}")
        print(f"✓ 总批次数: {total_batches}")
        print()
        
        start_time = time.time()
        
        for batch_num in range(start_batch, total_batches):
            batch_start = batch_num * self.batch_size
            batch_end = min(batch_start + self.batch_size, total_problems)
            
            print(f"\n{'='*70}")
            print(f"📦 批次 {batch_num + 1}/{total_batches} (问题 {batch_start}-{batch_end-1})")
            print(f"{'='*70}")
            
            batch_kgs = []
            batch_correct = 0
            batch_skipped = 0
            
            for idx in range(batch_start, batch_end):
                if idx in processed_ids:
                    batch_skipped += 1
                    continue
                
                item = dataset[idx]
                match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', item['answer'])
                ground_truth = match.group(1) if match else item['answer'].split('\n')[-1].strip()
                
                progress = idx - batch_start + 1
                total_in_batch = batch_end - batch_start
                print(f"  [{progress}/{total_in_batch}] ID={idx}: {item['question'][:45]}...")
                
                kg = self.generator.generate(
                    question_id=idx,
                    question=item["question"],
                    ground_truth=ground_truth,
                    max_retries=3
                )
                
                if kg:
                    batch_kgs.append(kg)
                    if kg["status"] == "correct":
                        batch_correct += 1
                        print(f"    ✅ 正确 (尝试 {kg['attempts']} 次)")
                    else:
                        print(f"    ❌ 错误 (模型: {kg['final_answer']}, 正确: {ground_truth})")
            
            if batch_kgs:
                stats = {
                    "processed": len(batch_kgs),
                    "skipped": batch_skipped,
                    "correct": batch_correct,
                    "incorrect": len(batch_kgs) - batch_correct
                }
                self.save_batch(batch_num, batch_kgs, stats)
            
            elapsed = time.time() - start_time
            completed_batches = batch_num - start_batch + 1
            avg_time = elapsed / completed_batches
            remaining = (total_batches - batch_num - 1) * avg_time
            
            print(f"\n  📊 批次: 处理 {len(batch_kgs)} | 正确 {batch_correct} | 跳过 {batch_skipped}")
            print(f"  ⏱️  已用: {elapsed/60:.1f}分钟 | 剩余: {remaining/60:.1f}分钟")
        
        self.merge_all_batches()
        print(f"\n🎉 完成！总用时: {(time.time()-start_time)/3600:.2f}小时")


def main():
    parser = argparse.ArgumentParser(description="GSM8K 知识图谱生成器")
    parser.add_argument("--batch-size", type=int, default=100, help="每批处理的问题数量")
    parser.add_argument("--start-batch", type=int, default=0, help="开始的批次编号")
    parser.add_argument("--max-batches", type=int, default=None, help="最大处理批次数")
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录")
    parser.add_argument("--merge-only", action="store_true", help="仅合并已有批次")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(output_dir=args.output_dir, batch_size=args.batch_size)
    
    if args.merge_only:
        processor.merge_all_batches()
    else:
        processor.process_dataset(start_batch=args.start_batch, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
