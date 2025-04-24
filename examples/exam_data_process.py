# -*- coding: utf-8 -*- # 指定UTF-8编码，确保中文注释正常

import json
import requests
import re
import os                   # 用于处理文件路径
import time                 # 用于可能的延时（如果需要）
from concurrent.futures import ThreadPoolExecutor, as_completed # 导入线程池和as_completed
from tqdm import tqdm       # 导入tqdm用于显示进度条
# import threading            # 最终JSON格式不再需要线程锁，注释掉

# --- 添加 colorama 导入 ---
import colorama
from colorama import Fore, Style, init # Fore 用于前景色, Style 用于样式, init 用于初始化

# --- 初始化 colorama ---
# autoreset=True 表示颜色将在每次 print() 后自动重置
init(autoreset=True)

# --- 配置 ---
# 输入 JSON 文件名 (确保此文件与脚本在同一目录下，或使用绝对路径)
INPUT_JSON_FILENAME = '../data/benchmark/metadata/mydata_qa.json'
# 输出 JSON 文件名 (每次运行会覆盖或创建新文件)
OUTPUT_JSON_FILENAME = 'rag_exam_inputs.json'  # 将扩展名更改为 .json

# API 端点和认证信息 (请替换成你的实际信息)
API_URL = 'https://f.yiya-ai.com/v1/chat-messages'
AUTH_TOKEN = 'app-1CQzBV5wukGNLrVeaxlsyv1c' # 替换成你的 Bearer Token
USER_ID = 'zkdn-1' # 替换成你的 User ID

# 并发设置
CONCURRENCY_LIMIT = 2 # 设置最大并发请求数量

# 请求头
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AUTH_TOKEN}',
    # 如果需要 Cookie，可以取消注释并添加
    # 'Cookie': 'acw_tc=...'
}

# --- 函数定义 ---

def clean_response_text(text):
    """清理模型回答文本"""
    if text is None: return ""
    # 移除 <think>...</think> 标签及其内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # 也可以在这里添加其他清理规则，例如移除特定的前缀/后缀等
    return cleaned_text

def process_query(query_item):
    """处理单个查询项"""
    query = query_item.get('query', '查询内容缺失')
    query_id = query_item.get('query_id', '查询ID缺失') # 保持 query_id 为字符串以匹配示例
    gt_answer = query_item.get('gt_answer', '标准答案缺失')

    # ---- 添加的调试变量 ----
    kb_event_detected = False
    kb_inner_json_parsed = False
    kb_items_found = 0
    # ----------------------

    request_payload = {
        "inputs": {}, "query": query, "response_mode": "streaming",
        "conversation_id": "", "user": USER_ID, "files": []
    }

    response_text = ""
    retrieved_context_list = []
    error_message = None
    success = True

    try:
        # 使用 timeout 参数，例如 180 秒 (3 分钟)
        with requests.post(API_URL, headers=HEADERS, json=request_payload, stream=True, timeout=180) as response:
            response.raise_for_status() # 检查 HTTP 错误状态 (例如 4xx, 5xx)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        json_part = decoded_line[len('data:'):].strip()
                        if not json_part: continue # 跳过空的 data: 行
                        try:
                            event_data = json.loads(json_part)
                            event_type = event_data.get('event')
                            data_payload = event_data.get('data', {})

                            # --- 调试和提取知识库检索结果 ---
                            if event_type == 'node_finished' and data_payload.get('title') == '知识库检索':
                                kb_event_detected = True
                                # --- 将此日志设为红色 ---
                                print(Fore.RED + f"\n[DEBUG KB] query_id={query_id}: '知识库检索' node_finished 事件已检测到。")
                                outputs = data_payload.get('outputs', {})
                                # print('outputs',outputs) # 如果需要，保持此行为白色或使用其他颜色
                                if outputs and 'text' in outputs:
                                    knowledge_json_str = outputs['text']
                                    # --- 将此日志设为红色 ---
                                    print(Fore.RED + f"[DEBUG KB] query_id={query_id}: 获取到 outputs['text'] (内嵌JSON字符串，前500字符): {knowledge_json_str[:500]}...")
                                    try:
                                        knowledge_data = json.loads(knowledge_json_str)
                                        kb_inner_json_parsed = True
                                        # --- 将此日志设为红色 ---
                                        print(Fore.RED + f"[DEBUG KB] query_id={query_id}: 内嵌JSON解析成功。Keys: {list(knowledge_data.keys())}")

                                        is_success = knowledge_data.get('success')
                                        data_section = knowledge_data.get('data')
                                        # 确保 data_section 是字典并且包含 'list'
                                        chunk_list = data_section.get('list') if isinstance(data_section, dict) else None

                                        # --- 将此日志设为红色 ---
                                        print(Fore.RED + f"[DEBUG KB] query_id={query_id}: 检查结构: success={is_success}, data_section exists={data_section is not None}, chunk_list exists={chunk_list is not None}")

                                        if is_success and chunk_list is not None: # 检查 list 是否存在且不为 None
                                             # --- 将此日志设为红色 ---
                                             print(Fore.RED + f"[DEBUG KB] query_id={query_id}: 结构检查通过，开始遍历 list (共 {len(chunk_list)} 项)...")
                                             for i, chunk in enumerate(chunk_list):
                                                # 确保 chunk 是字典
                                                if isinstance(chunk, dict):
                                                    kb_items_found += 1
                                                    # --- 将此日志设为红色 ---
                                                    # print(Fore.RED + f"[DEBUG KB] query_id={query_id}: 处理 chunk {i+1}: {chunk}") # 红色的详细输出可能太多了
                                                    # 优先使用 'sourceName', 否则用 'docId', 再不行就是 '未知'
                                                    doc_id = chunk.get("sourceName", chunk.get("docId", "未知文档来源"))
                                                    # 优先使用 'q', 否则用 'text' (如果存在), 再不行是空字符串
                                                    text = chunk.get("q", chunk.get("text", ""))
                                                    # --- 将此日志设为红色 ---
                                                    print(Fore.RED + f"[DEBUG KB] query_id={query_id}:  -> 提取到 doc_id='{doc_id}', text (前50字符)='{text[:50]}...'")
                                                    retrieved_context_list.append({"doc_id": str(doc_id), "text": text}) # 确保 doc_id 是字符串
                                                # else:
                                                    # print(f"[DEBUG KB] query_id={query_id}:  -> 跳过 list 中的项，不是字典: {chunk}")
                                             # --- 将此日志设为红色 ---
                                             print(Fore.RED + f"[DEBUG KB] query_id={query_id}: list 遍历完成。")
                                        else:
                                             # --- 将此日志设为红色 ---
                                             print(Fore.RED + f"[DEBUG KB] query_id={query_id}: 结构检查未通过或 list 为空/None。")

                                    except json.JSONDecodeError as e_inner:
                                        # 错误通常也适合用红色
                                        print(Fore.RED + f"  [错误] query_id={query_id}: 解析知识库内嵌JSON失败: {e_inner}")
                                        # print(f"  [错误 DUMP] 错误的 JSON 字符串: {knowledge_json_str}") # 可以取消注释以查看错误的JSON
                                    except Exception as e_inner_general:
                                        print(Fore.RED + f"  [错误] query_id={query_id}: 处理知识库内嵌数据时发生意外错误: {e_inner_general}")
                                else:
                                    # 警告可能是黄色 (Fore.YELLOW)
                                    print(Fore.YELLOW + f"[DEBUG KB] query_id={query_id}: '知识库检索' 事件的 outputs 中缺少 'text' 键。Outputs Keys: {list(outputs.keys())}")
                            # --- 结束知识库调试 ---

                            # 提取模型最终回答 (逻辑不变)
                            if event_type == 'node_finished' and data_payload.get('title') == '模型回答':
                                outputs = data_payload.get('outputs', {})
                                if outputs and 'text' in outputs:
                                     response_text = clean_response_text(outputs['text'])
                                     # print(f"[DEBUG RESP] query_id={query_id}: 从 '模型回答' 获取到响应: {response_text[:100]}...")
                            # 后备：如果模型回答节点没有文本，尝试从工作流结束事件获取
                            elif event_type == 'workflow_finished':
                                # print(f"[DEBUG FLOW] query_id={query_id}: Workflow finished event.")
                                outputs = data_payload.get('outputs', {})
                                # 只有在 response_text 仍然为空时才使用 workflow_finished 的 answer
                                if outputs and 'answer' in outputs and not response_text:
                                    response_text = clean_response_text(outputs['answer'])
                                    # print(f"[DEBUG FLOW] query_id={query_id}: 从 'workflow_finished' 获取到响应: {response_text[:100]}...")
                            # 示例：将警告设为黄色
                            elif event_type == 'message': # 虚构的例子
                                if data_payload.get('level') == 'warning':
                                     print(Fore.YELLOW + f"[警告] query_id={query_id}: {data_payload.get('text')}")


                        except json.JSONDecodeError as e:
                            # 将解析错误设为黄色或红色
                            print(Fore.YELLOW + f"  [警告] query_id={query_id}: 解析单行SSE JSON失败: {e}")
                            # print(f"  [警告 DUMP] 错误的行: {decoded_line}") # 可以取消注释以查看错误的行
                        except Exception as e_general:
                             print(Fore.YELLOW + f"  [警告] query_id={query_id}: 处理SSE事件时发生意外错误: {e_general}")

    except requests.exceptions.Timeout:
        error_message = f"API 请求超时 (超过180秒)"
        # 将错误设为红色
        print(Fore.RED + f"  [错误] query_id={query_id}: {error_message}")
        success = False
    except requests.exceptions.RequestException as e:
        # 捕获所有 requests 相关的错误 (连接、HTTP错误等)
        error_message = f"API 请求失败: {e}"
        # 将错误设为红色
        print(Fore.RED + f"  [错误] query_id={query_id}: {error_message}")
        success = False
    except Exception as e:
        # 捕获其他任何意外错误
        error_message = f"处理过程中发生意外错误: {e}"
        # 将错误设为红色
        print(Fore.RED + f"  [错误] query_id={query_id}: {error_message}")
        success = False

    # --- 在函数末尾添加总结性调试信息（红色） ---
    print(Fore.RED + f"[DEBUG SUM] query_id={query_id}: 处理完成。KB事件检测到={kb_event_detected}, 内嵌JSON解析成功={kb_inner_json_parsed}, 找到KB项={kb_items_found}, 最终上下文列表长度={len(retrieved_context_list)}, 处理成功={success}")
    # ---------------------------------------------------------

    # 构建结果字典，与期望的 JSON 结构中的单个元素匹配
    result_item = {
        "query_id": str(query_id), # 确保 query_id 是字符串
        "query": query,
        "gt_answer": gt_answer,
        "response": response_text if success else f"错误: {error_message}", # 如果失败则包含错误信息
        "retrieved_context": retrieved_context_list,
        "_processing_success": success # 内部标志，用于跟踪处理是否成功
    }
    return result_item

# --- 主逻辑 ---

if __name__ == "__main__":
    # 加载原始 JSON 文件
    queries_to_process = []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(script_dir, INPUT_JSON_FILENAME)
        print(f"正在从 {input_file_path} 加载查询...") # 这个可以保持默认颜色
        with open(input_file_path, 'r', encoding='utf-8') as f:
            my_doc_json = json.load(f)
            # 假设输入 JSON 结构是 {"results": [...]}
            queries_to_process = my_doc_json.get("results", [])
            if not queries_to_process:
                 # 将警告设为黄色
                 print(Fore.YELLOW + f"[警告] 在文件 {INPUT_JSON_FILENAME} 中未找到 'results' 键或列表为空。")
            else:
                 print(f"成功加载 {len(queries_to_process)} 个查询。") # 这个可以保持默认颜色
    except FileNotFoundError:
        print(Fore.RED + f"[错误] 输入JSON文件未找到: {input_file_path}") # 示例：将文件错误设为红色
        exit(1)
    except json.JSONDecodeError as e:
        print(Fore.RED + f"[错误] 解析JSON文件 {input_file_path} 失败: {e}")
        exit(1)
    except Exception as e:
        print(Fore.RED + f"[错误] 读取文件 {input_file_path} 时发生意外错误: {e}")
        exit(1)

    all_processed_results_in_memory = [] # 用于存储所有处理结果的列表
    output_file_path = os.path.join(script_dir, OUTPUT_JSON_FILENAME)
    print(f"结果将在处理完成后写入到: {output_file_path} (单个 JSON 文件)") # 这个可以保持默认颜色

    if queries_to_process:
        print(f"开始处理查询，最大并发数: {CONCURRENCY_LIMIT}...") # 这个可以保持默认颜色
        futures = []
        with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
            # 提交所有任务
            for item in queries_to_process:
                # 基本验证，确保 item 是字典并且有必要的键
                if isinstance(item, dict) and 'query_id' in item and 'query' in item:
                    future = executor.submit(process_query, item)
                    futures.append(future)
                else:
                    # 将警告设为黄色
                    print(Fore.YELLOW + f"[警告] 跳过输入文件中的无效项目 (非字典或缺少键): {item}")

            # 使用 tqdm 处理已完成的任务
            progress_bar = tqdm(as_completed(futures), total=len(futures), desc="处理查询中", unit="个查询")
            for future in progress_bar:
                try:
                    processed_result = future.result() # 获取单个处理结果
                    if processed_result:
                        # 移除内部处理标志（如果其他地方还需要此标志，可以保留）
                        processing_success_flag = processed_result.pop("_processing_success", None)
                        # 如果你只想将成功的结果添加到最终 JSON：
                        # if processing_success_flag:
                        #    all_processed_results_in_memory.append(processed_result)
                        # else:
                        #    print(Fore.YELLOW + f"因处理错误跳过 query_id {processed_result.get('query_id')} 的结果。")
                        # 当前：无论成功与否都添加所有结果：
                        all_processed_results_in_memory.append(processed_result)

                except Exception as e:
                    # 处理从 future.result() 抛出的异常 (即 process_query 中的异常)
                    # 通常，process_query 应该捕获自己的异常并返回错误信息，
                    # 但为了安全起见，这里也加一层捕获。
                    # 将获取结果时的严重错误设为红色
                    print(Fore.RED + f"\n[严重错误] 获取任务结果时发生错误: {e} - 该查询结果可能丢失或不完整。")

        print("所有查询处理完成。") # 这个可以保持默认颜色

        # --- 在所有任务完成后，将内存中的结果列表写入最终的 JSON 文件 ---
        final_output_data = {"results": all_processed_results_in_memory}

        print(f"正在将 {len(all_processed_results_in_memory)} 个结果写入到 {output_file_path}...") # 这个可以保持默认颜色
        try:
            with open(output_file_path, "w", encoding="utf-8") as outfile:
                # 使用 json.dump() 将整个 Python 字典写入文件
                # indent=2 使输出的 JSON 文件具有良好的可读性（缩进）
                # ensure_ascii=False 确保中文字符按原样写入，而不是 Unicode 转义序列
                json.dump(final_output_data, outfile, ensure_ascii=False, indent=2)
            print(f"成功写入结果到 {output_file_path}") # 这个可以保持默认颜色
        except IOError as e:
             print(Fore.RED + f"\n[错误] 无法打开或写入输出文件 {output_file_path}: {e}")
        except TypeError as e:
             print(Fore.RED + f"\n[错误] 序列化为 JSON 时发生类型错误 (可能数据结构有问题): {e}")
        except Exception as general_error:
             print(Fore.RED + f"\n[严重错误] 写入最终 JSON 文件时发生未预期的错误: {general_error}")
        # --- 写入完成 ---

    else:
        print("没有加载到有效的查询，无需处理。") # 这个可以保持默认颜色
        # 即使没有查询，也可能需要写入一个空的 results 列表
        print(f"将写入一个空的 'results' 列表到 {output_file_path}") # 这个可以保持默认颜色
        try:
             with open(output_file_path, "w", encoding="utf-8") as outfile:
                 json.dump({"results": []}, outfile, ensure_ascii=False, indent=2)
        except IOError as e:
             print(Fore.RED + f"\n[错误] 无法写入空的输出文件 {output_file_path}: {e}")

    print("\n处理结束。") # 这个可以保持默认颜色