"""
Analyze QuickDraw ndjson raw data files to generate comprehensive statistics.
Calculates averages across all categories and per-category for:
- Minimum/Maximum strokes
- Minimum/Maximum points overall  
- Minimum/Maximum points per stroke

Only analyzes successfully recognized drawings.
Saves results to data_statistics.json
"""

import os
import json
import ndjson
import numpy as np
import math
from typing import Dict, List, Tuple
from collections import defaultdict

# Categories to analyze (same as in rawDataToImage.py)
CATEGORIES = [
    'apple', 'banana', 'blackberry', 'grapes', 'pear', 'strawberry', 'watermelon'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'rawData')


def analyze_drawing(drawing_data: List[List[List[int]]]) -> Dict:
    """
    Analyze a single drawing to extract stroke and point statistics.
    
    Args:
        drawing_data: QuickDraw format drawing data
        
    Returns:
        Dictionary with drawing statistics
    """
    if not drawing_data:
        return None
    
    stroke_count = len(drawing_data)
    total_points = 0
    stroke_points_list = []
    
    # Analyze each stroke
    for stroke in drawing_data:
        if len(stroke) >= 2 and len(stroke[0]) > 0:
            stroke_points = len(stroke[0])
            total_points += stroke_points
            stroke_points_list.append(stroke_points)
    
    # Skip if no valid strokes
    if not stroke_points_list:
        return None
    
    return {
        'stroke_count': stroke_count,
        'total_points': total_points,
        'stroke_points_list': stroke_points_list,
        'min_points_per_stroke': min(stroke_points_list),
        'max_points_per_stroke': max(stroke_points_list),
        'avg_points_per_stroke': sum(stroke_points_list) / len(stroke_points_list)
    }


def analyze_category(category: str) -> Dict:
    """
    Analyze all drawings in a category to generate statistics.
    
    Args:
        category: Category name (e.g., 'apple')
        
    Returns:
        Dictionary with category statistics
    """
    input_file = os.path.join(RAW_DATA_DIR, f'full_simplified_{category}.ndjson')
    
    if not os.path.exists(input_file):
        print(f"❌ {input_file} not found!")
        return None
    
    print(f"📊 Analyzing '{category}'...")
    
    # Statistics collectors for per-image metrics
    stroke_counts = []  # Strokes per image
    total_points_list = []  # Points per image
    avg_points_per_stroke_list = []  # Average points per stroke per image
    
    # Statistics collectors for individual stroke analysis
    all_stroke_points = []  # All individual stroke point counts
    
    total_drawings = 0
    recognized_count = 0
    valid_drawings = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = ndjson.reader(f)
            
            for drawing in reader:
                total_drawings += 1
                
                # Only process recognized drawings
                if not drawing.get('recognized', False):
                    continue
                
                recognized_count += 1
                
                # Analyze drawing
                drawing_stats = analyze_drawing(drawing.get('drawing', []))
                if drawing_stats is None:
                    continue
                
                valid_drawings += 1
                
                # Collect per-image statistics
                stroke_counts.append(drawing_stats['stroke_count'])
                total_points_list.append(drawing_stats['total_points'])
                avg_points_per_stroke_list.append(drawing_stats['avg_points_per_stroke'])
                
                # Collect individual stroke points for overall stroke analysis
                all_stroke_points.extend(drawing_stats['stroke_points_list'])
    
    except Exception as e:
        print(f"❌ Error reading {input_file}: {e}")
        return None
    
    if not stroke_counts:
        print(f"   ⚠️  No valid drawings found")
        return None
    
    # Calculate comprehensive statistics
    stats = {
        'category': category,
        'total_drawings': total_drawings,
        'recognized_drawings': recognized_count,
        'valid_drawings': valid_drawings,
        'recognition_rate': recognized_count / total_drawings if total_drawings > 0 else 0,
        'validity_rate': valid_drawings / recognized_count if recognized_count > 0 else 0,
        
        # Strokes per image statistics
        'strokes_per_image': {
            'min': int(min(stroke_counts)),
            'max': int(max(stroke_counts)),
            'avg': sum(stroke_counts) / len(stroke_counts),
            'std': float(np.std(stroke_counts)),
            'median': float(np.median(stroke_counts))
        },
        
        # Total points per image statistics
        'points_per_image': {
            'min': int(min(total_points_list)),
            'max': int(max(total_points_list)),
            'avg': sum(total_points_list) / len(total_points_list),
            'std': float(np.std(total_points_list)),
            'median': float(np.median(total_points_list))
        },
        
        # Average points per stroke per image statistics
        'avg_points_per_stroke_per_image': {
            'min': min(avg_points_per_stroke_list),
            'max': max(avg_points_per_stroke_list),
            'avg': sum(avg_points_per_stroke_list) / len(avg_points_per_stroke_list),
            'std': float(np.std(avg_points_per_stroke_list)),
            'median': float(np.median(avg_points_per_stroke_list))
        },
        
        # Individual stroke points statistics (across all strokes in all images)
        'points_per_stroke_overall': {
            'min': int(min(all_stroke_points)) if all_stroke_points else 0,
            'max': int(max(all_stroke_points)) if all_stroke_points else 0,
            'avg': sum(all_stroke_points) / len(all_stroke_points) if all_stroke_points else 0,
            'std': float(np.std(all_stroke_points)) if all_stroke_points else 0,
            'median': float(np.median(all_stroke_points)) if all_stroke_points else 0,
            'total_strokes': len(all_stroke_points)
        }
    }
    
    print(f"   ✅ Analyzed {valid_drawings} valid drawings")
    print(f"      Strokes per image: {stats['strokes_per_image']['min']}-{stats['strokes_per_image']['max']} (avg: {stats['strokes_per_image']['avg']:.1f}, std: {stats['strokes_per_image']['std']:.1f})")
    print(f"      Points per image: {stats['points_per_image']['min']}-{stats['points_per_image']['max']} (avg: {stats['points_per_image']['avg']:.1f}, std: {stats['points_per_image']['std']:.1f})")
    print(f"      Points per stroke: {stats['points_per_stroke_overall']['min']}-{stats['points_per_stroke_overall']['max']} (avg: {stats['points_per_stroke_overall']['avg']:.1f}, std: {stats['points_per_stroke_overall']['std']:.1f})")
    
    return stats


def calculate_cross_category_averages(category_stats: List[Dict]) -> Dict:
    """
    Calculate averages across all categories.
    
    Args:
        category_stats: List of category statistics dictionaries
        
    Returns:
        Dictionary with cross-category averages
    """
    if not category_stats:
        return {}
    
    # Collect all values across categories for strokes per image
    stroke_mins = [stats['strokes_per_image']['min'] for stats in category_stats]
    stroke_maxs = [stats['strokes_per_image']['max'] for stats in category_stats]
    stroke_avgs = [stats['strokes_per_image']['avg'] for stats in category_stats]
    stroke_stds = [stats['strokes_per_image']['std'] for stats in category_stats]
    
    # Collect all values across categories for points per image
    points_mins = [stats['points_per_image']['min'] for stats in category_stats]
    points_maxs = [stats['points_per_image']['max'] for stats in category_stats]
    points_avgs = [stats['points_per_image']['avg'] for stats in category_stats]
    points_stds = [stats['points_per_image']['std'] for stats in category_stats]
    
    # Collect all values across categories for average points per stroke per image
    avg_points_per_stroke_mins = [stats['avg_points_per_stroke_per_image']['min'] for stats in category_stats]
    avg_points_per_stroke_maxs = [stats['avg_points_per_stroke_per_image']['max'] for stats in category_stats]
    avg_points_per_stroke_avgs = [stats['avg_points_per_stroke_per_image']['avg'] for stats in category_stats]
    avg_points_per_stroke_stds = [stats['avg_points_per_stroke_per_image']['std'] for stats in category_stats]
    
    # Collect all values across categories for individual stroke points
    stroke_points_mins = [stats['points_per_stroke_overall']['min'] for stats in category_stats]
    stroke_points_maxs = [stats['points_per_stroke_overall']['max'] for stats in category_stats]
    stroke_points_avgs = [stats['points_per_stroke_overall']['avg'] for stats in category_stats]
    stroke_points_stds = [stats['points_per_stroke_overall']['std'] for stats in category_stats]
    
    # Calculate total drawings and recognition rates
    total_drawings = sum(stats['total_drawings'] for stats in category_stats)
    total_recognized = sum(stats['recognized_drawings'] for stats in category_stats)
    total_valid = sum(stats['valid_drawings'] for stats in category_stats)
    total_strokes = sum(stats['points_per_stroke_overall']['total_strokes'] for stats in category_stats)
    
    return {
        'summary': {
            'categories_analyzed': len(category_stats),
            'total_drawings': total_drawings,
            'total_recognized': total_recognized,
            'total_valid': total_valid,
            'total_strokes_analyzed': total_strokes,
            'overall_recognition_rate': total_recognized / total_drawings if total_drawings > 0 else 0,
            'overall_validity_rate': total_valid / total_recognized if total_recognized > 0 else 0
        },
        
        'averages_across_categories': {
            'strokes_per_image': {
                'avg_min': math.floor(sum(stroke_mins) / len(stroke_mins)),  # Round down
                'avg_max': math.ceil(sum(stroke_maxs) / len(stroke_maxs)),   # Round up
                'avg_avg': sum(stroke_avgs) / len(stroke_avgs),
                'avg_std': sum(stroke_stds) / len(stroke_stds),
                'global_min': min(stroke_mins),
                'global_max': max(stroke_maxs)
            },
            
            'points_per_image': {
                'avg_min': math.floor(sum(points_mins) / len(points_mins)),  # Round down
                'avg_max': math.ceil(sum(points_maxs) / len(points_maxs)),   # Round up
                'avg_avg': sum(points_avgs) / len(points_avgs),
                'avg_std': sum(points_stds) / len(points_stds),
                'global_min': min(points_mins),
                'global_max': max(points_maxs)
            },
            
            'avg_points_per_stroke_per_image': {
                'avg_min': sum(avg_points_per_stroke_mins) / len(avg_points_per_stroke_mins),
                'avg_max': sum(avg_points_per_stroke_maxs) / len(avg_points_per_stroke_maxs),
                'avg_avg': sum(avg_points_per_stroke_avgs) / len(avg_points_per_stroke_avgs),
                'avg_std': sum(avg_points_per_stroke_stds) / len(avg_points_per_stroke_stds),
                'global_min': min(avg_points_per_stroke_mins),
                'global_max': max(avg_points_per_stroke_maxs)
            },
            
            'points_per_stroke_overall': {
                'avg_min': math.floor(sum(stroke_points_mins) / len(stroke_points_mins)),  # Round down
                'avg_max': math.ceil(sum(stroke_points_maxs) / len(stroke_points_maxs)),   # Round up
                'avg_avg': sum(stroke_points_avgs) / len(stroke_points_avgs),
                'avg_std': sum(stroke_points_stds) / len(stroke_points_stds),
                'global_min': min(stroke_points_mins),
                'global_max': max(stroke_points_maxs)
            }
        }
    }


def generate_recommendations(cross_category_stats: Dict) -> Dict:
    """
    Generate quality filtering recommendations based on the statistics.
    
    Args:
        cross_category_stats: Cross-category statistics
        
    Returns:
        Dictionary with filtering recommendations
    """
    if not cross_category_stats or 'averages_across_categories' not in cross_category_stats:
        return {}
    
    avgs = cross_category_stats['averages_across_categories']
    
    # Calculate recommended thresholds based on statistics
    recommendations = {
        'quality_filter_presets': {
            'lenient': {
                'min_strokes': max(1, avgs['strokes_per_image']['avg_min'] - 1),
                'max_strokes': avgs['strokes_per_image']['avg_max'] + 20,
                'min_points': max(5, avgs['points_per_image']['avg_min'] - 10),
                'max_points': avgs['points_per_image']['avg_max'] + 300,
                'min_points_per_stroke': max(2, avgs['points_per_stroke_overall']['avg_min'] - 1),
                'max_points_per_stroke': avgs['points_per_stroke_overall']['avg_max'] + 50,
                'description': 'Light filtering, keeps 70-80% of drawings'
            },
            
            'balanced': {
                'min_strokes': avgs['strokes_per_image']['avg_min'],
                'max_strokes': avgs['strokes_per_image']['avg_max'] + 10,
                'min_points': avgs['points_per_image']['avg_min'],
                'max_points': avgs['points_per_image']['avg_max'] + 100,
                'min_points_per_stroke': avgs['points_per_stroke_overall']['avg_min'],
                'max_points_per_stroke': avgs['points_per_stroke_overall']['avg_max'] + 20,
                'description': 'Balanced filtering based on data averages'
            },
            
            'strict': {
                'min_strokes': avgs['strokes_per_image']['avg_min'] + 1,
                'max_strokes': avgs['strokes_per_image']['avg_max'] - 5,
                'min_points': avgs['points_per_image']['avg_min'] + 15,
                'max_points': avgs['points_per_image']['avg_max'] - 50,
                'min_points_per_stroke': avgs['points_per_stroke_overall']['avg_min'] + 1,
                'max_points_per_stroke': avgs['points_per_stroke_overall']['avg_max'] - 20,
                'description': 'Strict filtering, keeps highest quality drawings'
            }
        },
        
        'statistical_insights': {
            'strokes_per_image': {
                'range': f"{avgs['strokes_per_image']['global_min']} to {avgs['strokes_per_image']['global_max']}",
                'average': f"{avgs['strokes_per_image']['avg_avg']:.1f}",
                'std_dev': f"{avgs['strokes_per_image']['avg_std']:.1f}"
            },
            'points_per_image': {
                'range': f"{avgs['points_per_image']['global_min']} to {avgs['points_per_image']['global_max']}",
                'average': f"{avgs['points_per_image']['avg_avg']:.1f}",
                'std_dev': f"{avgs['points_per_image']['avg_std']:.1f}"
            },
            'points_per_stroke': {
                'range': f"{avgs['points_per_stroke_overall']['global_min']} to {avgs['points_per_stroke_overall']['global_max']}",
                'average': f"{avgs['points_per_stroke_overall']['avg_avg']:.1f}",
                'std_dev': f"{avgs['points_per_stroke_overall']['avg_std']:.1f}"
            }
        },
        
        'notes': [
            f"Total strokes analyzed: {cross_category_stats['summary']['total_strokes_analyzed']:,}",
            f"Recognition rate: {cross_category_stats['summary']['overall_recognition_rate']:.1%}",
            f"Validity rate: {cross_category_stats['summary']['overall_validity_rate']:.1%}",
            f"Standard deviations indicate data variability - higher values suggest more diverse drawing styles"
        ]
    }
    
    return recommendations


def main():
    """Main analysis function."""
    print("=" * 60)
    print("📊 QuickDraw Data Statistics Analyzer")
    print("=" * 60)
    
    # Check if raw data directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Raw data directory not found: {RAW_DATA_DIR}")
        print("   Please ensure the rawData directory exists with ndjson files.")
        return
    
    # Analyze each category
    category_stats = []
    
    for category in CATEGORIES:
        stats = analyze_category(category)
        if stats:
            category_stats.append(stats)
    
    if not category_stats:
        print("❌ No valid category data found!")
        return
    
    # Calculate cross-category averages
    print(f"\n📈 Calculating cross-category averages...")
    cross_category_stats = calculate_cross_category_averages(category_stats)
    
    # Generate recommendations
    recommendations = generate_recommendations(cross_category_stats)
    
    # Compile final results
    results = {
        'analysis_info': {
            'script_version': '1.0',
            'categories_analyzed': CATEGORIES,
            'total_categories': len(category_stats),
            'analysis_date': __import__('datetime').datetime.now().isoformat()
        },
        'per_category_stats': category_stats,
        'cross_category_stats': cross_category_stats,
        'recommendations': recommendations
    }
    
    # Save results to file
    output_file = os.path.join(SCRIPT_DIR, 'data_statistics.json')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Statistics saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error saving statistics: {e}")
        return
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"📊 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    summary = cross_category_stats['summary']
    avgs = cross_category_stats['averages_across_categories']
    
    print(f"Categories analyzed: {summary['categories_analyzed']}")
    print(f"Total drawings: {summary['total_drawings']:,}")
    print(f"Recognized drawings: {summary['total_recognized']:,} ({summary['overall_recognition_rate']:.1%})")
    print(f"Valid drawings: {summary['total_valid']:,} ({summary['overall_validity_rate']:.1%})")
    print(f"Total strokes analyzed: {summary['total_strokes_analyzed']:,}")
    
    print(f"\n🎯 COMPREHENSIVE STATISTICS:")
    print(f"  Strokes per image: {avgs['strokes_per_image']['avg_min']}-{avgs['strokes_per_image']['avg_max']} (avg: {avgs['strokes_per_image']['avg_avg']:.1f}, std: {avgs['strokes_per_image']['avg_std']:.1f})")
    print(f"  Points per image: {avgs['points_per_image']['avg_min']}-{avgs['points_per_image']['avg_max']} (avg: {avgs['points_per_image']['avg_avg']:.1f}, std: {avgs['points_per_image']['avg_std']:.1f})")
    print(f"  Points per stroke: {avgs['points_per_stroke_overall']['avg_min']}-{avgs['points_per_stroke_overall']['avg_max']} (avg: {avgs['points_per_stroke_overall']['avg_avg']:.1f}, std: {avgs['points_per_stroke_overall']['avg_std']:.1f})")
    
    print(f"\n📋 Full statistics saved to: data_statistics.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()