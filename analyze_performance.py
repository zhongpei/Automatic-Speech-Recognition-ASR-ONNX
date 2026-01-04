#!/usr/bin/env python3
"""
性能分析脚本 - 分析ASR推理性能日志

用法:
    python analyze_performance.py performance_*.jsonl
    python analyze_performance.py performance_*.jsonl --compare
    python analyze_performance.py performance_*.jsonl --output report.html
"""

import json
import os
import argparse
import glob
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any
import statistics


def load_jsonl_files(file_patterns: List[str]) -> List[Dict[str, Any]]:
    """加载JSONL文件并返回所有记录"""
    records = []
    for pattern in file_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            record['_source_file'] = file_path
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"警告: 跳过 {file_path}:{line_num} - JSON解析错误: {e}")
            except FileNotFoundError:
                print(f"警告: 文件未找到: {file_path}")
            except Exception as e:
                print(f"错误: 读取文件 {file_path} 时出错: {e}")
    return records


def calculate_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算统计信息"""
    if not records:
        return {}
    
    rtf_values = [r.get('rtf', 0) for r in records]
    processing_times = [r.get('processing_time', 0) for r in records]
    decode_speeds = [r.get('decode_speed', 0) for r in records]
    audio_durations = [r.get('audio_duration', 0) for r in records]
    tokens_decoded = [r.get('tokens_decoded', 0) for r in records]
    
    stats = {
        'count': len(records),
        'rtf': {
            'mean': statistics.mean(rtf_values) if rtf_values else 0,
            'median': statistics.median(rtf_values) if rtf_values else 0,
            'min': min(rtf_values) if rtf_values else 0,
            'max': max(rtf_values) if rtf_values else 0,
            'stdev': statistics.stdev(rtf_values) if len(rtf_values) > 1 else 0
        },
        'processing_time': {
            'mean': statistics.mean(processing_times) if processing_times else 0,
            'median': statistics.median(processing_times) if processing_times else 0,
            'min': min(processing_times) if processing_times else 0,
            'max': max(processing_times) if processing_times else 0,
            'total': sum(processing_times)
        },
        'decode_speed': {
            'mean': statistics.mean(decode_speeds) if decode_speeds else 0,
            'median': statistics.median(decode_speeds) if decode_speeds else 0,
            'min': min(decode_speeds) if decode_speeds else 0,
            'max': max(decode_speeds) if decode_speeds else 0
        },
        'audio_duration': {
            'total': sum(audio_durations),
            'mean': statistics.mean(audio_durations) if audio_durations else 0
        },
        'tokens_decoded': {
            'total': sum(tokens_decoded),
            'mean': statistics.mean(tokens_decoded) if tokens_decoded else 0
        }
    }
    
    return stats


def group_by_model_dir(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按模型目录分组"""
    grouped = defaultdict(list)
    for record in records:
        model_dir = record.get('model_dir', 'unknown')
        grouped[model_dir].append(record)
    return dict(grouped)


def group_by_audio_file(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按音频文件分组"""
    grouped = defaultdict(list)
    for record in records:
        audio_file = record.get('audio_file', 'unknown')
        grouped[audio_file].append(record)
    return dict(grouped)


def compare_models(grouped_records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """比较不同模型的性能"""
    comparison = {}
    model_dirs = sorted(grouped_records.keys())
    
    if len(model_dirs) < 2:
        return comparison
    
    # 计算每个模型的统计信息
    model_stats = {}
    for model_dir in model_dirs:
        model_stats[model_dir] = calculate_statistics(grouped_records[model_dir])
    
    # 选择基准模型（第一个）
    baseline = model_dirs[0]
    baseline_stats = model_stats[baseline]
    
    comparison['baseline'] = baseline
    comparison['models'] = {}
    
    for model_dir in model_dirs[1:]:
        stats = model_stats[model_dir]
        baseline_rtf = baseline_stats['rtf']['mean']
        current_rtf = stats['rtf']['mean']
        
        rtf_improvement = ((baseline_rtf - current_rtf) / baseline_rtf * 100) if baseline_rtf > 0 else 0
        
        baseline_speed = baseline_stats['decode_speed']['mean']
        current_speed = stats['decode_speed']['mean']
        speed_improvement = ((current_speed - baseline_speed) / baseline_speed * 100) if baseline_speed > 0 else 0
        
        comparison['models'][model_dir] = {
            'rtf_mean': current_rtf,
            'rtf_improvement_pct': rtf_improvement,
            'decode_speed_mean': current_speed,
            'speed_improvement_pct': speed_improvement,
            'processing_time_mean': stats['processing_time']['mean'],
            'stats': stats
        }
    
    return comparison


def print_summary(records: List[Dict[str, Any]]):
    """打印总体摘要"""
    stats = calculate_statistics(records)
    
    print("\n" + "=" * 80)
    print("性能分析摘要")
    print("=" * 80)
    print(f"总记录数: {stats['count']}")
    print(f"\nRTF (Real-Time Factor):")
    print(f"  平均值: {stats['rtf']['mean']:.3f}")
    print(f"  中位数: {stats['rtf']['median']:.3f}")
    print(f"  范围: {stats['rtf']['min']:.3f} - {stats['rtf']['max']:.3f}")
    print(f"  标准差: {stats['rtf']['stdev']:.3f}")
    print(f"\n处理时间 (秒):")
    print(f"  平均值: {stats['processing_time']['mean']:.3f}")
    print(f"  总时间: {stats['processing_time']['total']:.3f}")
    print(f"  范围: {stats['processing_time']['min']:.3f} - {stats['processing_time']['max']:.3f}")
    print(f"\n解码速度 (tokens/s):")
    print(f"  平均值: {stats['decode_speed']['mean']:.3f}")
    print(f"  范围: {stats['decode_speed']['min']:.3f} - {stats['decode_speed']['max']:.3f}")
    print(f"\n音频总时长: {stats['audio_duration']['total']:.3f} 秒")
    print(f"总解码tokens: {stats['tokens_decoded']['total']}")
    print("=" * 80)


def print_model_comparison(grouped_records: Dict[str, List[Dict[str, Any]]]):
    """打印模型对比"""
    comparison = compare_models(grouped_records)
    
    if not comparison:
        return
    
    print("\n" + "=" * 80)
    print("模型性能对比")
    print("=" * 80)
    print(f"基准模型: {comparison['baseline']}")
    print()
    
    baseline_stats = calculate_statistics(grouped_records[comparison['baseline']])
    print(f"{'模型目录':<30} {'RTF均值':<12} {'RTF变化%':<12} {'解码速度':<12} {'速度变化%':<12}")
    print("-" * 80)
    print(f"{comparison['baseline']:<30} {baseline_stats['rtf']['mean']:<12.3f} {'基准':<12} {baseline_stats['decode_speed']['mean']:<12.3f} {'基准':<12}")
    
    for model_dir, comp_data in comparison['models'].items():
        rtf_change = comp_data['rtf_improvement_pct']
        speed_change = comp_data['speed_improvement_pct']
        rtf_sign = "+" if rtf_change < 0 else "-"
        speed_sign = "+" if speed_change > 0 else "-"
        
        print(f"{model_dir:<30} {comp_data['rtf_mean']:<12.3f} {rtf_sign}{abs(rtf_change):.1f}%{'':<8} {comp_data['decode_speed_mean']:<12.3f} {speed_sign}{abs(speed_change):.1f}%{'':<8}")
    
    print("=" * 80)


def print_detailed_table(records: List[Dict[str, Any]]):
    """打印详细表格"""
    print("\n" + "=" * 80)
    print("详细性能数据")
    print("=" * 80)
    print(f"{'音频文件':<40} {'模型目录':<20} {'RTF':<10} {'处理时间':<12} {'解码速度':<12} {'Tokens':<10}")
    print("-" * 80)
    
    for record in sorted(records, key=lambda x: (x.get('audio_file', ''), x.get('model_dir', ''))):
        audio_file = os.path.basename(record.get('audio_file', 'unknown'))
        model_dir = record.get('model_dir', 'unknown')
        rtf = record.get('rtf', 0)
        processing_time = record.get('processing_time', 0)
        decode_speed = record.get('decode_speed', 0)
        tokens = record.get('tokens_decoded', 0)
        
        print(f"{audio_file:<40} {model_dir:<20} {rtf:<10.3f} {processing_time:<12.3f} {decode_speed:<12.3f} {tokens:<10}")
    
    print("=" * 80)


def export_csv(records: List[Dict[str, Any]], output_file: str):
    """导出CSV文件"""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['timestamp', 'model_dir', 'audio_file', 'task_prompt', 
                     'audio_duration', 'processing_time', 'rtf', 'tokens_decoded', 
                     'decode_speed', 'recognized_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in records:
            row = {
                'timestamp': record.get('timestamp', ''),
                'model_dir': record.get('model_dir', ''),
                'audio_file': record.get('audio_file', ''),
                'task_prompt': record.get('task_prompt', ''),
                'audio_duration': record.get('audio_duration', 0),
                'processing_time': record.get('processing_time', 0),
                'rtf': record.get('rtf', 0),
                'tokens_decoded': record.get('tokens_decoded', 0),
                'decode_speed': record.get('decode_speed', 0),
                'recognized_text': record.get('recognized_text', '')
            }
            writer.writerow(row)
    
    print(f"\nCSV报告已保存到: {output_file}")


def export_html(records: List[Dict[str, Any]], output_file: str):
    """导出HTML报告"""
    stats = calculate_statistics(records)
    grouped_by_model = group_by_model_dir(records)
    comparison = compare_models(grouped_by_model)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ASR性能分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .stats {{ margin: 20px 0; }}
        .improvement {{ color: green; }}
        .degradation {{ color: red; }}
    </style>
</head>
<body>
    <h1>ASR性能分析报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>总体统计</h2>
    <div class="stats">
        <p><strong>总记录数:</strong> {stats['count']}</p>
        <p><strong>RTF平均值:</strong> {stats['rtf']['mean']:.3f}</p>
        <p><strong>处理时间平均值:</strong> {stats['processing_time']['mean']:.3f} 秒</p>
        <p><strong>解码速度平均值:</strong> {stats['decode_speed']['mean']:.3f} tokens/s</p>
        <p><strong>音频总时长:</strong> {stats['audio_duration']['total']:.3f} 秒</p>
    </div>
"""
    
    if comparison:
        html += """
    <h2>模型性能对比</h2>
    <table>
        <tr>
            <th>模型目录</th>
            <th>RTF均值</th>
            <th>RTF变化</th>
            <th>解码速度 (tokens/s)</th>
            <th>速度变化</th>
        </tr>
"""
        baseline_stats = calculate_statistics(grouped_by_model[comparison['baseline']])
        html += f"""
        <tr>
            <td>{comparison['baseline']} (基准)</td>
            <td>{baseline_stats['rtf']['mean']:.3f}</td>
            <td>-</td>
            <td>{baseline_stats['decode_speed']['mean']:.3f}</td>
            <td>-</td>
        </tr>
"""
        for model_dir, comp_data in comparison['models'].items():
            rtf_change = comp_data['rtf_improvement_pct']
            speed_change = comp_data['speed_improvement_pct']
            rtf_class = 'improvement' if rtf_change < 0 else 'degradation'
            speed_class = 'improvement' if speed_change > 0 else 'degradation'
            
            html += f"""
        <tr>
            <td>{model_dir}</td>
            <td>{comp_data['rtf_mean']:.3f}</td>
            <td class="{rtf_class}">{rtf_change:+.1f}%</td>
            <td>{comp_data['decode_speed_mean']:.3f}</td>
            <td class="{speed_class}">{speed_change:+.1f}%</td>
        </tr>
"""
        html += """
    </table>
"""
    
    html += """
    <h2>详细数据</h2>
    <table>
        <tr>
            <th>时间戳</th>
            <th>模型目录</th>
            <th>音频文件</th>
            <th>RTF</th>
            <th>处理时间 (秒)</th>
            <th>解码速度 (tokens/s)</th>
            <th>Tokens</th>
        </tr>
"""
    
    for record in sorted(records, key=lambda x: (x.get('timestamp', ''), x.get('audio_file', ''))):
        html += f"""
        <tr>
            <td>{record.get('timestamp', '')[:19]}</td>
            <td>{record.get('model_dir', '')}</td>
            <td>{os.path.basename(record.get('audio_file', ''))}</td>
            <td>{record.get('rtf', 0):.3f}</td>
            <td>{record.get('processing_time', 0):.3f}</td>
            <td>{record.get('decode_speed', 0):.3f}</td>
            <td>{record.get('tokens_decoded', 0)}</td>
        </tr>
"""
    
    html += """
    </table>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nHTML报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='分析ASR推理性能日志',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个文件
  python analyze_performance.py performance_20260103_150052.jsonl
  
  # 分析多个文件
  python analyze_performance.py performance_*.jsonl
  
  # 比较不同模型
  python analyze_performance.py performance_*.jsonl --compare
  
  # 导出CSV
  python analyze_performance.py performance_*.jsonl --output report.csv
  
  # 导出HTML
  python analyze_performance.py performance_*.jsonl --output report.html
        """
    )
    parser.add_argument('files', nargs='+', help='JSONL性能日志文件（支持通配符）')
    parser.add_argument('--compare', action='store_true', help='比较不同模型的性能')
    parser.add_argument('--output', type=str, help='导出报告文件（CSV或HTML格式）')
    parser.add_argument('--detailed', action='store_true', help='显示详细表格')
    
    args = parser.parse_args()
    
    # 加载记录
    print(f"正在加载日志文件...")
    records = load_jsonl_files(args.files)
    
    if not records:
        print("错误: 没有找到有效的记录")
        return
    
    print(f"已加载 {len(records)} 条记录")
    
    # 打印摘要
    print_summary(records)
    
    # 如果启用比较模式
    if args.compare:
        grouped_by_model = group_by_model_dir(records)
        if len(grouped_by_model) > 1:
            print_model_comparison(grouped_by_model)
        else:
            print("\n注意: 只找到一个模型目录，无法进行比较")
    
    # 显示详细表格
    if args.detailed:
        print_detailed_table(records)
    
    # 导出报告
    if args.output:
        output_file = args.output
        if output_file.endswith('.csv'):
            export_csv(records, output_file)
        elif output_file.endswith('.html'):
            export_html(records, output_file)
        else:
            print(f"警告: 不支持的文件格式，将导出为CSV: {output_file}.csv")
            export_csv(records, output_file + '.csv')


if __name__ == '__main__':
    main()

