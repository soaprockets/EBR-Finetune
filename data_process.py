import json
import os
import csv
import re
import random
from typing import List, Dict
from tqdm import tqdm


def text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的词重叠相似度（Jaccard相似度系数）

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        相似度分数 (0.0-1.0)
    """
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if len(union) > 0 else 0.0


def generate_negative_samples(
    records: List[Dict[str, str]],
    easy_ratio: float = 0.7,
    hard_ratio: float = 0.3,
    use_similarity_based_sampling: bool = True,
    output_file: str = None
) -> List[Dict[str, str]]:
    """
    为数据集生成负样本，按照easy和hard负样本的比例进行采样

    Args:
        records: 包含anchor和positive的记录列表
        easy_ratio: 容易区分负样本的比例 (默认0.7)
        hard_ratio: 难区分负样本的比例 (默认0.3)
        use_similarity_based_sampling: 是否启用基于相似度的采样
        output_file: 输出文件路径，如果提供则实时写入文件 (JSONL格式)

    Returns:
        包含anchor、positive、negative的完整记录列表
    """
    if not records:
        return []

    print(f"开始生成负样本...")
    print(f"  - 数据集大小: {len(records)}")
    print(f"  - Easy比例: {easy_ratio}, Hard比例: {hard_ratio}")
    print(f"  - 相似度采样: {'启用' if use_similarity_based_sampling else '禁用'}")

    # 构建候选池：所有记录的positive作为潜在的负样本
    candidate_pool = list(set(record["positive"] for record in records))
    print(f"  - 候选池大小: {len(candidate_pool)}")

    # 如果指定了输出文件，准备实时写入
    output_f = None
    if output_file:
        print(f"  - 实时输出文件: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_f = open(output_file, "w", encoding="utf-8")

    # 统计计数器
    easy_count = 0
    hard_count = 0

    # 用于跟踪已使用的负样本，提高多样性
    used_negatives = set()

    # 为每个记录生成负样本
    results = []
    with tqdm(total=len(records), desc="负采样进度", unit="样本") as pbar:
        for i, record in enumerate(records):
            anchor = record["anchor"]
            positive = record["positive"]

            # 过滤掉正样本本身
            valid_candidates = [cand for cand in candidate_pool if cand != positive]

            if len(valid_candidates) == 0:
                # 如果没有有效候选，随机选择一个不同的候选
                neg = random.choice(candidate_pool)
                while neg == positive:
                    neg = random.choice(candidate_pool)
                sample = {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": neg
                }
                results.append(sample)

                # 实时写入文件
                if output_f:
                    output_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    output_f.flush()  # 立即刷新到磁盘

                pbar.update(1)
                continue

            if not use_similarity_based_sampling:
                # 简单随机采样
                neg = random.choice(valid_candidates)
                sample = {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": neg
                }
                results.append(sample)

                # 实时写入文件
                if output_f:
                    output_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    output_f.flush()  # 立即刷新到磁盘

                # 随机决定easy还是hard（仅用于统计）
                if random.random() < hard_ratio:
                    hard_count += 1
                else:
                    easy_count += 1
                pbar.set_postfix({
                    'easy': easy_count,
                    'hard': hard_count,
                    'ratio': f"{easy_count/(easy_count+hard_count):.2f}" if (easy_count+hard_count) > 0 else "0.00"
                })
                pbar.update(1)
                continue

            # 基于相似度的智能采样
            try:
                # 计算当前anchor与所有候选负样本的相似度
                candidate_similarities = []
                for cand in valid_candidates:
                    sim = text_similarity(anchor, cand)
                    candidate_similarities.append((cand, sim))

                # 决定采样类型：按照hard_ratio的比例选择hard negative
                sample_type = "hard" if random.random() < hard_ratio else "easy"

                if sample_type == "hard":
                    # hard negative: 从相似度最高的几个样本中随机选择一个（增加多样性）
                    candidate_similarities.sort(key=lambda x: x[1], reverse=True)
                    # 从前3个最相似的样本中随机选择，避免总是选择相同的样本
                    top_k = min(3, len(candidate_similarities))
                    neg = random.choice(candidate_similarities[:top_k])[0]
                    hard_count += 1
                else:
                    # easy negative: 从相似度最低的几个样本中随机选择一个（增加多样性）
                    candidate_similarities.sort(key=lambda x: x[1])
                    # 从前3个最不相似的样本中随机选择，避免总是选择相同的样本
                    top_k = min(3, len(candidate_similarities))
                    neg = random.choice(candidate_similarities[:top_k])[0]
                    easy_count += 1

                sample = {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": neg
                }
                results.append(sample)

                # 实时写入文件
                if output_f:
                    output_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    output_f.flush()  # 立即刷新到磁盘

                pbar.set_postfix({
                    'easy': easy_count,
                    'hard': hard_count,
                    'ratio': f"{easy_count/(easy_count+hard_count):.2f}" if (easy_count+hard_count) > 0 else "0.00"
                })
                pbar.update(1)

            except Exception as e:
                print(f"智能负采样失败，使用随机采样: {e}")
                # fallback到随机采样
                neg = random.choice(valid_candidates)
                sample = {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": neg
                }
                results.append(sample)

                # 实时写入文件
                if output_f:
                    output_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    output_f.flush()  # 立即刷新到磁盘

                if sample_type == "easy":
                    easy_count += 1
                else:
                    hard_count += 1
                pbar.set_postfix({
                    'easy': easy_count,
                    'hard': hard_count,
                    'ratio': f"{easy_count/(easy_count+hard_count):.2f}" if (easy_count+hard_count) > 0 else "0.00"
                })
                pbar.update(1)

    # 关闭输出文件
    if output_f:
        output_f.close()
        print(f"实时输出文件已保存: {output_file}")

    print(f"负采样完成!")
    print(f"  - 总样本数: {len(results)}")
    print(f"  - Easy负样本: {easy_count} ({easy_count/len(results)*100:.1f}%)")
    print(f"  - Hard负样本: {hard_count} ({hard_count/len(results)*100:.1f}%)")
    print(f"  - 目标比例: Easy {easy_ratio*100:.1f}%, Hard {hard_ratio*100:.1f}%")

    return results


def parse_anchor_positive_lines(text: str) -> List[Dict[str, str]]:
    """
    解析形如你截图中的 txt 文本：

        3922740676469679\\x01{"anchor": "...", 
        "positive": "..."}
        3166758620619776\\x01{"anchor": "...", 
        "positive": "..."}

    即：前面是房号 ID，后面跟一个 \\x01 作为分隔符，然后是多行 JSON 字符串，
    JSON 内只关心 anchor / positive 两个字段。

    参数
    ----
    text : str
        原始文本，可以包含多行，每行一条数据。

    返回
    ----
    List[Dict[str, str]]
        形如 [{"anchor": "...", "positive": "..."}, ...]
    """
    results: List[Dict[str, str]] = []

    # 按控制字符 \x01 分隔，每个片段对应一个房号后面的内容
    parts = text.split("\001")
    # 第一个 part 可能是开头的脏数据/空串，可以直接从第二个开始处理
    for part in parts[1:]:
        if not part:
            continue

        # 只提取包含大括号的 JSON 部分：从第一个 '{' 到最后一个 '}' 为止
        start = part.find("{")
        end = part.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue

        json_part = part[start : end + 1].strip()
        if not json_part:
            continue

        try:
            obj = json.loads(json_part)
        except json.JSONDecodeError:
            # 当前片段不是合法 JSON，跳过
            continue

        anchor = obj.get("anchor")
        positive = obj.get("positive")

        # 去掉前后空白和换行
        if isinstance(anchor, str):
            anchor = anchor.strip()
        if isinstance(positive, str):
            positive = positive.strip()

        if anchor is not None and positive is not None:
            results.append({"anchor": anchor, "positive": positive})

    return results



def parse_txt_dir(dir_path: str) -> List[Dict[str, str]]:
    """
    遍历目录下所有 .txt 文件，读取并解析 anchor / positive。

    参数
    ----
    dir_path : str
        存放 txt 的目录路径。

    返回
    ----
    List[Dict[str, str]]
        汇总所有 txt 文件中解析出的 anchor / positive。
    """
    all_results: List[Dict[str, str]] = []

    # 收集所有文件路径
    all_files = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            file_path = os.path.join(root, name)
            all_files.append(file_path)

    if not all_files:
        print(f"警告: 在目录 {dir_path} 中没有找到任何文件")
        return all_results

    print(f"开始处理 {len(all_files)} 个文件...")

    # 使用进度条处理文件
    with tqdm(total=len(all_files), desc="文件处理进度", unit="文件") as pbar:
        for file_path in all_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError as e:
                print(f"跳过文件 {file_path}: {e}")
                pbar.update(1)
                continue

            file_results = parse_anchor_positive_lines(text)
            all_results.extend(file_results)

            # 更新进度条，显示当前文件的处理结果
            pbar.set_postfix({
                '当前文件': os.path.basename(file_path),
                '解析记录': len(file_results),
                '总记录': len(all_results)
            })
            pbar.update(1)

    print(f"文件处理完成! 共解析到 {len(all_results)} 条记录")
    return all_results


def export_to_jsonl_and_csv(
    records: List[Dict[str, str]], jsonl_path: str, csv_path: str
) -> None:
    """
    将解析结果导出为 JSONL 和 CSV 文件。

    JSONL：每行一个 JSON 对象，包含 anchor / positive / negative。
    CSV  ：三列，header 为 anchor, positive, negative。
    """
    print(f"开始导出数据到文件...")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 导出 JSONL
    print(f"导出 JSONL 文件: {jsonl_path}")
    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for item in records:
            f_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 导出 CSV
    print(f"导出 CSV 文件: {csv_path}")
    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["anchor", "positive", "negative"])

        # 使用进度条显示CSV写入进度
        with tqdm(total=len(records), desc="CSV导出进度", unit="记录") as pbar:
            for item in records:
                writer.writerow([
                    item.get("anchor", ""),
                    item.get("positive", ""),
                    item.get("negative", "")
                ])
                pbar.update(1)

    print(f"数据导出完成!")
    print(f"  - JSONL文件: {jsonl_path}")
    print(f"  - CSV文件: {csv_path}")
    print(f"  - 总记录数: {len(records)}")


if __name__ == "__main__":
    # 示例：处理目录下所有 txt 文件，并导出结果
    txt_dir = "/code/hexufeng01/llm_offline/gte_offline/train_text"  # TODO: 替换成你的真实 txt 目录
    data = parse_txt_dir(txt_dir)
    print(f"共解析到 {len(data)} 条记录")

    # 为数据生成负样本（70% easy + 30% hard），启用实时输出
    if data:
        print(f"为数据生成负样本...")
        # 设置实时输出文件路径
        realtime_output = "/code/hexufeng01/llm_offline/gte_offline/demo_data/anchor_positive_negative_realtime.jsonl"
        data_with_negatives = generate_negative_samples(
            records=data,
            easy_ratio=0.7,
            hard_ratio=0.3,
            use_similarity_based_sampling=False,  # 改为随机采样，默认避免重复问题
            output_file=realtime_output  # 启用实时输出
        )
        print(f"负采样完成，生成 {len(data_with_negatives)} 条包含负样本的记录")
        print(f"实时输出文件: {realtime_output}")
    else:
        data_with_negatives = data
        realtime_output = None

    # 导出路径（可根据需要修改）
    jsonl_out = "/code/hexufeng01/llm_offline/gte_offline/demo_data/anchor_positive_negative.jsonl"
    csv_out = "/code/hexufeng01/llm_offline/gte_offline/demo_data/anchor_positive_negative.csv"
    export_to_jsonl_and_csv(data_with_negatives, jsonl_out, csv_out)
    print(f"已导出到 {jsonl_out} 和 {csv_out}")
