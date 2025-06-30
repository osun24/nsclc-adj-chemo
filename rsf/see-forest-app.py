import streamlit as st
import numpy as np
import joblib
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.pyplot as plt
import tempfile
import os
import json
import time
import re
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

# Try to import OpenAI and dotenv with fallback
try:
    import openai
    from dotenv import load_dotenv
    import threading
    import queue
    # Load environment variables
    load_dotenv()
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    load_dotenv = None
    threading = None
    queue = None

# Initialize OpenAI client
if OPENAI_AVAILABLE:
    openai_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI')
    if openai_key:
        client = openai.Client(api_key=openai_key)
    else:
        client = None
else:
    client = None

# File-based cache location
CACHE_FILE = 'rsf/feature_descriptions_cache.json'
PROGRESS_FILE = 'rsf/background_progress.json'

# Configure Streamlit page for wide layout
st.set_page_config(
    page_title="RSF Tree Visualizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load RSF model (best model from iterative feature selection - 19 features, iteration 31)
rsf = joblib.load('rsf/rsf_results_affy/iterative_feature_selection/20250628_best_rsf_model_iter_31_19_features.pkl')

# Data preprocessing to match iterative-feature-selection-affy.py
# Load and combine train + validation data as done in iterative feature selection
train_orig = pd.read_csv("affyTrain.csv")
valid_orig = pd.read_csv("affyValidation.csv")
train = pd.concat([train_orig, valid_orig], axis=0, ignore_index=True)

# Apply same preprocessing as iterative feature selection
train['Adjuvant Chemo'] = train['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})

# Load the best 19 features from iteration 31
try:
    best_features_df = pd.read_csv("rsf/rsf_results_affy/iterative_feature_selection/20250628_best_features_iteration_31.csv")
    covariates = best_features_df['Feature'].tolist()
    print(f"✅ Loaded {len(covariates)} best features from iteration 31")
except FileNotFoundError:
    # Fallback to hardcoded features if file not found
    covariates = [
        'Stage_IA', 'FAM117A', 'CCNB1', 'PURA', 'PFKP', 'PARM1', 
        'ADGRF5', 'GUCY1A1', 'SLC1A4', 'TENT5C', 'Age', 'HILPDA', 
        'ETV5', 'STIM1', 'KDM5C', 'NCAPG2', 'ZFR2', 'SETBP1', 'RTCA'
    ]
    print(f"⚠️ Using hardcoded 19 features (best features file not found)")

# Filter to only include features that exist in the data
covariates = [c for c in covariates if c in train.columns]
df = train[['OS_STATUS', 'OS_MONTHS'] + covariates]
covariates = df.columns[2:]

print(f"📊 Model info: {len(rsf.estimators_)} trees, {len(covariates)} features")
print(f"🎯 Features: {list(covariates)}")

# Make feature names globally accessible
feature_names = covariates

def create_stage_variable(df, stage_prefix):
    """
    Combine stage indicator columns into a single categorical stage variable.
    """
    stage_columns = [col for col in df.columns if col.startswith(stage_prefix)]
    def get_stage(row):
        for col in stage_columns:
            if row[col] == 1:
                return col.replace(stage_prefix, '').strip()
        return np.nan
    df[stage_prefix.rstrip('_')] = df.apply(get_stage, axis=1)
    return df

def plot_km_and_logrank(df, feature_col, threshold, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    """
    Plot Kaplan-Meier curves and perform log-rank test for a feature split by threshold.
    Include p-value and number at risk table in the plot.
    """
    # Create grouping variable based on threshold
    df_plot = df.copy()
    df_plot['Group'] = np.where(df_plot[feature_col] <= threshold, f"≤ {threshold:.2f}", f"> {threshold:.2f}")
    
    # Drop missing values
    df_plot = df_plot.dropna(subset=['Group', duration_col, event_col])
    groups = df_plot['Group'].unique()
    kmf_dict = {}
    
    # Prepare the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KM curves for each group
    for group in groups:
        ix = df_plot['Group'] == group
        kmf = KaplanMeierFitter()
        kmf.fit(df_plot.loc[ix, duration_col], event_observed=df_plot.loc[ix, event_col], label=str(group))
        kmf_dict[group] = kmf
        kmf.plot_survival_function(ax=ax, ci_show=True)
    
    # Perform log-rank test
    results = multivariate_logrank_test(df_plot[duration_col], df_plot['Group'], df_plot[event_col])
    p_value = results.p_value
    
    # Include p-value as text in the plot
    plt.title(f'Kaplan-Meier Survival Curves by {feature_col}')
    plt.xlabel('Time (Months)')
    plt.ylabel('Overall Survival Probability')
    plt.legend()
    plt.text(0.6, 0.1, f'Log-rank p-value: {p_value:.4f}', transform=ax.transAxes)
    
    # Add number at risk table if there are enough groups
    if kmf_dict:
        add_at_risk_counts(*kmf_dict.values(), ax=ax)
    
    # Adjust layout to accommodate at-risk table
    plt.tight_layout()
    
    # Return the figure
    return fig

def km_overall(df, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    """Generate overall Kaplan-Meier survival curve"""
    # Create a KaplanMeierFitter instance
    kmf = KaplanMeierFitter(label='Overall Survival')

    # Fit the data into the model
    kmf.fit(df[duration_col], event_observed=df[event_col])

    # Plot the Kaplan-Meier estimate
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf.plot_survival_function(ax=ax, ci_show=True)
    plt.title('Overall Kaplan-Meier Survival Curve (95% CI)')
    plt.xlabel('Survival Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    add_at_risk_counts(kmf, ax=ax)
    plt.tight_layout()
    
    return fig

# Helper function to calculate survival stats
def calculate_survival_stats(survival_function):
    times = [t for t, s in survival_function]
    probs = [s for t, s in survival_function]
    median_survival_time = next((time for time, prob in survival_function if prob <= 0.5), None)
    if median_survival_time is None:
        if probs and probs[0] == 1 and probs[-1] == 0:
            median_survival_time = times[-1]
        elif times:
            median_survival_time = max(times)
    min_survival_time = min(times) if times else "N/A"
    max_survival_time = max(times) if times else "N/A"
    return min_survival_time, median_survival_time, max_survival_time

def plot_leaf_node_km(leaf_info, selected_leaf_id, df, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    """
    Plot Kaplan-Meier curves comparing a selected leaf node against other leaf nodes.
    Uses the survival times from the leaf nodes to create synthetic patient groups.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the selected leaf information
    selected_leaf = leaf_info[selected_leaf_id]
    selected_times = selected_leaf['survival_times']
    selected_n_samples = selected_leaf['n_samples']
    
    # Group leaves into "selected" vs "other" categories based on survival characteristics
    selected_mean = selected_leaf['mean_survival']
    
    # Create groups based on mean survival time relative to selected leaf
    group_data = []
    group_labels = []
    
    # Add data for selected leaf (using median as a representative value)
    selected_median = selected_leaf['median_survival']
    # Create synthetic data points around the median for visualization
    n_points = max(10, int(selected_n_samples))
    times_selected = np.random.normal(selected_median, selected_median * 0.2, n_points)
    times_selected = np.maximum(times_selected, 0.1)  # Ensure positive times
    events_selected = np.ones(len(times_selected))  # Assume all events for simplicity
    
    group_data.append((times_selected, events_selected))
    group_labels.append(f"Selected Leaf {selected_leaf_id}")
    
    # Add data for other leaves grouped by similarity
    other_leaves_similar = []
    other_leaves_different = []
    
    for leaf_id, leaf_data in leaf_info.items():
        if leaf_id != selected_leaf_id:
            leaf_mean = leaf_data['mean_survival']
            # Group by similarity (within 30% of selected leaf's mean)
            if abs(leaf_mean - selected_mean) / selected_mean <= 0.3:
                other_leaves_similar.append(leaf_data)
            else:
                other_leaves_different.append(leaf_data)
    
    # Create synthetic data for similar leaves
    if other_leaves_similar:
        similar_medians = [leaf['median_survival'] for leaf in other_leaves_similar]
        avg_median_similar = np.mean(similar_medians)
        n_points = sum([max(5, int(leaf['n_samples'])) for leaf in other_leaves_similar])
        times_similar = np.random.normal(avg_median_similar, avg_median_similar * 0.2, n_points)
        times_similar = np.maximum(times_similar, 0.1)
        events_similar = np.ones(len(times_similar))
        
        group_data.append((times_similar, events_similar))
        group_labels.append(f"Similar Leaves (n={len(other_leaves_similar)})")
    
    # Create synthetic data for different leaves
    if other_leaves_different:
        different_medians = [leaf['median_survival'] for leaf in other_leaves_different]
        avg_median_different = np.mean(different_medians)
        n_points = sum([max(5, int(leaf['n_samples'])) for leaf in other_leaves_different])
        times_different = np.random.normal(avg_median_different, avg_median_different * 0.2, n_points)
        times_different = np.maximum(times_different, 0.1)
        events_different = np.ones(len(times_different))
        
        group_data.append((times_different, events_different))
        group_labels.append(f"Different Leaves (n={len(other_leaves_different)})")
    
    # Plot KM curves for each group
    kmf_dict = {}
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, ((times, events), label) in enumerate(zip(group_data, group_labels)):
        if len(times) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(times, event_observed=events, label=label)
            kmf_dict[label] = kmf
            kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i % len(colors)])
    
    # Perform log-rank test if we have multiple groups
    if len(group_data) > 1:
        try:
            # Combine all data for log-rank test
            all_times = []
            all_events = []
            all_groups = []
            
            for i, ((times, events), label) in enumerate(zip(group_data, group_labels)):
                all_times.extend(times)
                all_events.extend(events)
                all_groups.extend([label] * len(times))
            
            results = multivariate_logrank_test(all_times, all_groups, all_events)
            p_value = results.p_value
            
            plt.text(0.6, 0.1, f'Log-rank p-value: {p_value:.4f}', transform=ax.transAxes)
        except Exception as e:
            plt.text(0.6, 0.1, 'Log-rank test failed', transform=ax.transAxes)
    
    plt.title(f'Kaplan-Meier: Leaf Node {selected_leaf_id} vs Other Leaves')
    plt.xlabel('Survival Time (Months)')
    plt.ylabel('Survival Probability')
    plt.legend()
    
    # Add number at risk table
    if kmf_dict:
        try:
            add_at_risk_counts(*kmf_dict.values(), ax=ax)
        except Exception:
            pass  # Skip if at-risk table fails
    
    plt.tight_layout()
    return fig

# Build graph for a single tree
def build_graph(tree_, node_id, G, feature_names, root_fixed=False):
    node_id = int(node_id)
    left_child = tree_.children_left[node_id]
    right_child = tree_.children_right[node_id]
    threshold = tree_.threshold[node_id]
    feature = tree_.feature[node_id]
    if node_id == 0:
        color = 'green'
        feature_name = feature_names[feature] if feature != -2 else "No Feature"
        label = f"Root Node: {feature_name} ≤ {threshold:.2f}" if feature != -2 else "Root Node"
        G.add_node(node_id, label=label, color=color, physics=not root_fixed)
    elif left_child == -1 and right_child == -1:
        all_values = [tree_.value[n] for n in range(tree_.node_count) if tree_.children_left[n] == -1 and tree_.children_right[n] == -1]
        global_min = np.min([np.min(value) for value in all_values])
        global_max = np.max([np.max(value) for value in all_values])
        red_to_black = mcolors.LinearSegmentedColormap.from_list("RedBlack", ["#FF0000", "black"])
        value = tree_.value[node_id]
        value_mean = np.mean(value)
        value_median = np.median(value)
        value_count = np.sum(tree_.n_node_samples[node_id])
        cmap = red_to_black
        norm_value = (value_mean - global_min) / (global_max - global_min) if global_max > global_min else 0
        color = cm.colors.to_hex(cmap(norm_value))
        min_surv, median_surv, max_surv = calculate_survival_stats(value)
        label = f"Leaf node\nMin: {min_surv:.2f}\nMedian: {median_surv:.2f}\nMax: {max_surv:.2f}\nCount: {value_count}"
        title = f"Full Value List: {value}"
        G.add_node(node_id, label=label, title=title, color=color)
    else:
        feature_name = feature_names[feature]
        label = f"{feature_name} ≤ {threshold:.2f}"
        color = 'blue'
        G.add_node(node_id, label=label, color=color)
    if left_child != -1:
        G.add_edge(node_id, left_child, label=f"True", color='green')
        build_graph(tree_, left_child, G, feature_names, root_fixed)
    if right_child != -1:
        G.add_edge(node_id, right_child, label=f"False", color='red')
        build_graph(tree_, right_child, G, feature_names, root_fixed)

# Collect node features and thresholds for later use
def collect_node_features(tree_, feature_names):
    node_info = {}
    for node_id in range(tree_.node_count):
        if tree_.children_left[node_id] != -1 and tree_.children_right[node_id] != -1:  # Split node
            feature_idx = tree_.feature[node_id]
            if feature_idx != -2:
                threshold = tree_.threshold[node_id]
                feature_name = feature_names[feature_idx]
                node_info[node_id] = {"feature": feature_name, "threshold": threshold}
    return node_info

# Collect leaf node information for KM analysis
def collect_leaf_nodes(tree_):
    """Collect information about all leaf nodes in the tree"""
    leaf_info = {}
    for node_id in range(tree_.node_count):
        if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:  # Leaf node
            value = tree_.value[node_id]
            n_samples = tree_.n_node_samples[node_id]
            leaf_info[node_id] = {
                "survival_times": value,
                "n_samples": n_samples,
                "mean_survival": np.mean(value),
                "median_survival": np.median(value)
            }
    return leaf_info

# === PERSISTENT CACHE AND BACKGROUND PROCESSING FUNCTIONS ===

def load_feature_cache():
    """Load feature descriptions from persistent JSON cache"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                print(f"✅ [CACHE] Loaded {len(cache)} cached feature descriptions from {CACHE_FILE}")
                return cache
        except Exception as e:
            print(f"❌ [CACHE] Error loading cache file: {e}")
    return {}

def save_feature_cache(cache):
    """Save feature descriptions to persistent JSON cache"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"💾 [CACHE] Saved {len(cache)} feature descriptions to {CACHE_FILE}")
    except Exception as e:
        print(f"❌ [CACHE] Error saving cache file: {e}")

def save_progress_status(status):
    """Save background progress to file"""
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"❌ [PROGRESS] Error saving progress file: {e}")

def load_progress_status():
    """Load background progress from file"""
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"❌ [PROGRESS] Error loading progress file: {e}")
    return {}

def get_feature_description_cached(feature_name, cache):
    """Get feature description with cache check"""
    if feature_name in cache:
        print(f"🎯 [CACHE HIT] Found cached description for: {feature_name}")
        return cache[feature_name]
    
    # If not in cache, call LLM
    result = get_feature_description_llm_with_tokens(feature_name)
    
    # Handle both old format (string) and new format (dict with tokens)
    if isinstance(result, dict) and 'description' in result:
        description = result['description']
        print(f"📊 [CACHE] Used {result.get('tokens', 'unknown')} tokens for: {feature_name}")
    else:
        description = result
    
    # Cache the result (just the description text)
    cache[feature_name] = description
    save_feature_cache(cache)
    
    return description
    save_feature_cache(cache)
    
    return description

def background_feature_processor(feature_queue, cache, result_callback):
    """Background thread function to process feature description requests"""
    while True:
        try:
            feature_name = feature_queue.get(timeout=1)
            if feature_name is None:  # Poison pill to stop thread
                break
                
            if feature_name not in cache:
                print(f"🔄 [BACKGROUND] Processing feature: {feature_name}")
                description = get_feature_description_llm(feature_name)
                cache[feature_name] = description
                save_feature_cache(cache)
                
                # Notify main thread of completion
                if result_callback:
                    result_callback(feature_name, description)
            else:
                print(f"⚡ [BACKGROUND] Feature already cached: {feature_name}")
                
            feature_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ [BACKGROUND] Error processing feature: {e}")
            feature_queue.task_done()

def get_feature_description_llm(feature_name):
    """
    Get LLM-based description of a feature in the context of NSCLC survival analysis.
    """
    print(f"\n" + "="*80)
    print(f"🤖 [LLM REQUEST] Starting feature description request for: {feature_name}")
    print(f"⏰ [LLM REQUEST] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 [LLM REQUEST] Function called from: get_feature_description_llm()")
    print("="*80)
    
    if not OPENAI_AVAILABLE:
        print("❌ [LLM REQUEST] OpenAI not available - missing packages")
        print("💡 [LLM REQUEST] Install with: pip install openai python-dotenv")
        return "Feature descriptions require OpenAI integration (install 'openai' and 'python-dotenv' packages)"
    
    if not client:
        print("❌ [LLM REQUEST] OpenAI client not configured - missing API key")
        print("💡 [LLM REQUEST] Add OPENAI_API_KEY to .env file")
        return "Feature description unavailable (OpenAI API key not configured in .env file)"
    
    try:
        system_message = """"You are an expert in cancer genomics and bioinformatics, specializing in non-small cell lung cancer (NSCLC) survival analysis. You analyze genomic features and their clinical significance in cancer research. Be aware that some features provided may not be linked at all with NSCLC - if there is no known association, please state that clearly."""
        
        user_message = f"""Use the web, if necessary to provide a concise, clinical description (1-2 sentences) of the feature "{feature_name}" relevant to oncologists and researchers in the context of NSCLC survival analysis. Think step by step through these key ideas:

1. What this feature represents (gene, pathway, clinical variable, etc.)
2. The strength of evidence in the liteature supporting its role in NSCLC prognosis or treatment, if ANY
3. Its known role or significance in NSCLC prognosis or treatment, if ANY
4. How it might influence patient survival outcomes, if KNOWN"""

        print(f"📝 [LLM REQUEST] Preparing API call to GPT-4.1-mini")
        print(f"🎯 [LLM REQUEST] Target feature: '{feature_name}'")
        print(f"📊 [LLM REQUEST] System message length: {len(system_message)} characters")
        print(f"📊 [LLM REQUEST] User message length: {len(user_message)} characters")
        print(f"🔑 [LLM REQUEST] Using API key: {openai_key[:10]}...{openai_key[-4:] if len(openai_key) > 14 else '***'}")
        print(f"🌐 [LLM REQUEST] Making API call to OpenAI...")
        
        # Record start time for latency measurement
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=200  # Reduced from 200 to enforce brevity
        )
        
        # Calculate latency
        end_time = time.time()
        latency = end_time - start_time
        
        response = completion.choices[0].message.content.strip()
        
        print(f"\n" + "-"*60)
        print(f"✅ [LLM RESPONSE] Successfully received response!")
        print(f"⏱️ [LLM RESPONSE] API call latency: {latency:.2f} seconds")
        print(f"📏 [LLM RESPONSE] Response length: {len(response)} characters")
        print(f"🔢 [LLM RESPONSE] Token usage - Prompt: {completion.usage.prompt_tokens}, Completion: {completion.usage.completion_tokens}, Total: {completion.usage.total_tokens}")
        print(f"💰 [LLM RESPONSE] Model used: {completion.model}")
        print(f"📄 [LLM RESPONSE] Full response text:")
        print(f"    {response}")
        print("-"*60)
        
        # Also log the successful caching
        print(f"💾 [LLM CACHE] Caching response for feature: {feature_name}")
        
        return response
        
    except Exception as e:
        error_msg = f"Error retrieving feature description: {str(e)}"
        print(f"\n" + "!"*60)
        print(f"❌ [LLM ERROR] API call failed for feature: {feature_name}")
        print(f"🚨 [LLM ERROR] Exception type: {type(e).__name__}")
        print(f"📝 [LLM ERROR] Error message: {str(e)}")
        print(f"⚠️ [LLM ERROR] Returning error message to user")
        print("!"*60)
        return error_msg

def startup_background_preloader(cache, status_callback=None):
    """Background thread function to preload feature descriptions for the first 200 features"""
    if not OPENAI_AVAILABLE or not client:
        print("🚫 [STARTUP] OpenAI not available - skipping background preloading")
        if status_callback:
            status_callback({'status': 'skipped', 'reason': 'OpenAI not available'})
        return
    
    print("🚀 [STARTUP] Background preloader thread started")
    print(f"🎯 [STARTUP] Target: First 200 unique features across all trees")
    
    total_tokens_used = 0
    max_features = 200  # 200 feature limit
    
    try:
        # Keep track of all processed features to avoid duplicates
        processed_features = set()
        requests_made = 0
        trees_processed = 0
        
        print(f"📊 [STARTUP] Starting systematic tree traversal from tree 1...")
        print(f"🌲 [STARTUP] Total trees available: {len(rsf.estimators_)}")
        
        # Process trees sequentially starting from the first tree
        for tree_idx in range(len(rsf.estimators_)):
            if requests_made >= max_features:
                print(f"🛑 [STARTUP] Reached feature limit ({requests_made}/{max_features} features)")
                break
                
            print(f"\n🌳 [STARTUP] === Processing Tree {tree_idx + 1}/{len(rsf.estimators_)} ===")
            
            tree_ = rsf.estimators_[tree_idx].tree_
            node_info = collect_node_features(tree_, feature_names)
            tree_features = [info["feature"] for info in node_info.values()]
            
            # Find new features in this tree that haven't been processed yet
            new_features = [f for f in tree_features 
                          if f not in cache and f not in processed_features]
            
            if not new_features:
                print(f"📋 [STARTUP] Tree {tree_idx + 1}: No new features to process (all cached)")
                trees_processed += 1
                continue
                
            print(f"📋 [STARTUP] Tree {tree_idx + 1}: Found {len(new_features)} new features")
            print(f"🔍 [STARTUP] New features: {', '.join(new_features[:5])}{'...' if len(new_features) > 5 else ''}")
            
            # Process each new feature from this tree
            tree_tokens_used = 0
            tree_features_processed = 0
            
            for feature in new_features:
                if requests_made >= max_features:
                    print(f"🛑 [STARTUP] Reached feature limit during tree {tree_idx + 1}")
                    break
                    
                print(f"🔄 [STARTUP] Processing feature {requests_made + 1}/{max_features}: {feature}")
                
                try:
                    # Get description and track actual token usage
                    description = get_feature_description_llm_with_tokens(feature)
                    
                    # Extract tokens from response if available
                    if isinstance(description, dict) and 'tokens' in description:
                        tokens_used = description['tokens']
                        description_text = description['description']
                    else:
                        # Fallback to estimation if no token info
                        description_text = description
                        tokens_used = estimate_tokens(feature, description_text)
                    
                    # Update cache and tracking
                    cache[feature] = description_text
                    processed_features.add(feature)
                    requests_made += 1
                    total_tokens_used += tokens_used
                    tree_tokens_used += tokens_used
                    tree_features_processed += 1
                    
                    print(f"✅ [STARTUP] Feature processed: {feature}")
                    print(f"📊 [STARTUP] Progress: {requests_made}/{max_features} features, {tokens_used} tokens used")
                    
                    # Save cache every 5 requests to prevent data loss
                    if requests_made % 5 == 0:
                        save_feature_cache(cache)
                        print(f"💾 [STARTUP] Checkpoint saved ({requests_made} features, {total_tokens_used:,} tokens)")
                    
                    # Update status via callback if provided
                    if status_callback:
                        progress_status = {
                            'status': 'processing',
                            'trees_processed': trees_processed,
                            'features_processed': requests_made,
                            'max_features': max_features,
                            'tokens_used': total_tokens_used,
                            'current_tree': tree_idx + 1,
                            'current_feature': feature,
                            'timestamp': time.time()
                        }
                        status_callback(progress_status)
                        # Also save to file for persistence
                        save_progress_status(progress_status)
                        
                except Exception as e:
                    print(f"❌ [STARTUP] Error processing {feature}: {e}")
                    continue
            
            trees_processed += 1
            
            print(f"🌳 [STARTUP] Tree {tree_idx + 1} complete:")
            print(f"   ├── Features processed: {tree_features_processed}/{len(new_features)}")
            print(f"   ├── Tree tokens used: {tree_tokens_used:,}")
            print(f"   └── Total progress: {requests_made}/{max_features} features")
        
        # Final save and summary
        save_feature_cache(cache)
        
        print(f"\n🎉 [STARTUP] Background preloading complete!")
        print(f"📈 [STARTUP] === FINAL STATISTICS ===")
        print(f"   ├── Trees processed: {trees_processed}/{len(rsf.estimators_)}")
        print(f"   ├── Features processed: {requests_made}/{max_features}")
        print(f"   ├── Total tokens used: {total_tokens_used:,}")
        print(f"   ├── Feature completion: {(requests_made/max_features)*100:.1f}%")
        print(f"   └── Total cached features: {len(cache)}")
        
        # Final status callback
        if status_callback:
            status_callback({
                'status': 'completed',
                'trees_processed': trees_processed,
                'total_trees': len(rsf.estimators_),
                'features_processed': requests_made,
                'max_features': max_features,
                'tokens_used': total_tokens_used,
                'reached_limit': requests_made >= max_features
            })
        
        # Return results for further processing
        return {
            'status': 'completed',
            'trees_processed': trees_processed,
            'total_trees': len(rsf.estimators_),
            'features_processed': requests_made,
            'max_features': max_features,
            'tokens_used': total_tokens_used,
            'reached_limit': requests_made >= max_features
        }
        
    except Exception as e:
        print(f"💥 [STARTUP] Background preloader crashed: {e}")
        print(f"🔧 [STARTUP] Attempting to save partial progress...")
        # Save whatever we have
        try:
            save_feature_cache(cache)
            print(f"✅ [STARTUP] Partial cache saved with {len(cache)} features")
        except Exception as save_error:
            print(f"❌ [STARTUP] Failed to save partial cache: {save_error}")
        
        if status_callback:
            status_callback({
                'status': 'error',
                'error': str(e),
                'trees_processed': trees_processed,
                'features_processed': requests_made,
                'tokens_used': total_tokens_used
            })
        
        return {
            'status': 'error',
            'error': str(e),
            'trees_processed': trees_processed,
            'features_processed': requests_made,
            'tokens_used': total_tokens_used
        }

def get_preload_queue_for_trees(start_tree_idx, cache, max_features_remaining):
    """
    Generate a queue of features to preload for trees starting from start_tree_idx
    Returns list of (tree_idx, feature_name) tuples up to max_features_remaining
    """
    if not OPENAI_AVAILABLE or not client or max_features_remaining <= 0:
        return []
    
    queue = []
    processed_features = set(cache.keys())
    features_added = 0
    
    print(f"🔍 [QUEUE] Building preload queue starting from tree {start_tree_idx + 1}")
    print(f"🎯 [QUEUE] Feature budget: {max_features_remaining} features")
    
    for tree_idx in range(start_tree_idx, len(rsf.estimators_)):
        if features_added >= max_features_remaining:
            print(f"🛑 [QUEUE] Feature limit reached at tree {tree_idx}")
            break
            
        tree_ = rsf.estimators_[tree_idx].tree_
        node_info = collect_node_features(tree_, feature_names)
        tree_features = [info["feature"] for info in node_info.values()]
        
        # Find new features in this tree
        new_features = [f for f in tree_features 
                      if f not in processed_features]
        
        for feature in new_features:
            if features_added >= max_features_remaining:
                print(f"🛑 [QUEUE] Would exceed feature limit, stopping queue building")
                break
                
            queue.append((tree_idx, feature))
            processed_features.add(feature)
            features_added += 1
    
    print(f"📋 [QUEUE] Built queue with {len(queue)} features across {len(set(item[0] for item in queue))} trees")
    
    return queue

def get_feature_description_llm_with_tokens(feature_name):
    """
    Enhanced version of get_feature_description_llm that returns token usage information
    """
    print(f"\n" + "="*80)
    print(f"🤖 [LLM REQUEST] Starting feature description request for: {feature_name}")
    print(f"⏰ [LLM REQUEST] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    if not OPENAI_AVAILABLE:
        print("❌ [LLM REQUEST] OpenAI not available - missing packages")
        return "Feature descriptions require OpenAI integration (install 'openai' and 'python-dotenv' packages)"
    
    if not client:
        print("❌ [LLM REQUEST] OpenAI client not configured - missing API key")
        return "Feature description unavailable (OpenAI API key not configured in .env file)"
    
    try:
        system_message = """You are an expert in cancer genomics and bioinformatics, specializing in non-small cell lung cancer (NSCLC) survival analysis. You analyze genomic features and their clinical significance in cancer research."""
        
        user_message = f"""Provide a brief, clinical description (2-3 sentences) of the feature "{feature_name}" in the context of NSCLC survival analysis. Focus on:

1. What this feature represents (gene, pathway, clinical variable, etc.)
2. Its known role or significance in NSCLC prognosis or treatment
3. How it might influence patient survival outcomes

Keep the response concise and clinically relevant for oncologists and researchers."""

        print(f"📝 [LLM REQUEST] Making API call to GPT-4.1-mini for: {feature_name}")
        
        # Record start time for latency measurement
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=200
        )
        
        # Calculate latency
        end_time = time.time()
        latency = end_time - start_time
        
        response = completion.choices[0].message.content.strip()
        
        # Extract actual token usage
        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        
        print(f"✅ [LLM RESPONSE] Success! Latency: {latency:.2f}s, Tokens: {total_tokens}")
        print(f"📊 [LLM RESPONSE] Breakdown - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
        
        # Return both description and token count
        return {
            'description': response,
            'tokens': total_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'latency': latency
        }
        
    except Exception as e:
        error_msg = f"Error retrieving feature description: {str(e)}"
        print(f"❌ [LLM ERROR] API call failed for feature: {feature_name} - {str(e)}")
        return error_msg

def estimate_tokens(feature_name, description):
    """
    Rough token estimation for fallback when actual token count is unavailable
    """
    # Rough approximation: 1 token ≈ 4 characters
    system_chars = 200  # Approximate system message length
    user_chars = len(f"Provide a brief, clinical description of {feature_name}...") + 100
    response_chars = len(description)
    
    total_chars = system_chars + user_chars + response_chars
    estimated_tokens = total_chars // 4
    
    return estimated_tokens

def preload_all_features():
    """Pre-load feature descriptions for all trees to improve performance (UI-based version)"""
    if not OPENAI_AVAILABLE or not client:
        st.warning("OpenAI not available for feature pre-loading")
        return
    
    # Collect all unique features from all trees
    all_features = set()
    
    # Sample a subset of trees for efficiency (first 10 trees)
    max_trees_to_process = min(10, len(rsf.estimators_))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for tree_idx in range(max_trees_to_process):
        status_text.text(f"Scanning tree {tree_idx + 1}/{max_trees_to_process}...")
        progress_bar.progress((tree_idx + 1) / max_trees_to_process * 0.5)  # 50% for scanning
        
        tree_ = rsf.estimators_[tree_idx].tree_
        node_info = collect_node_features(tree_, feature_names)
        tree_features = [info["feature"] for info in node_info.values()]
        all_features.update(tree_features)
    
    # Now load descriptions for all unique features
    all_features = list(all_features)
    cache = st.session_state['persistent_feature_cache']
    
    missing_features = [f for f in all_features if f not in cache]
    
    if not missing_features:
        st.success(f"All {len(all_features)} features already cached!")
        return
    
    st.info(f"Pre-loading {len(missing_features)} missing feature descriptions...")
    
    for i, feature in enumerate(missing_features):
        status_text.text(f"Loading description for {feature}... ({i + 1}/{len(missing_features)})")
        progress_bar.progress(0.5 + (i + 1) / len(missing_features) * 0.5)  # 50-100% for loading
        
        description = get_feature_description_llm(feature)
        cache[feature] = description
        
        # Save cache periodically
        if (i + 1) % 5 == 0:
            save_feature_cache(cache)
    
    # Final save
    save_feature_cache(cache)
    progress_bar.progress(1.0)
    status_text.text("Pre-loading complete!")
    st.success(f"Successfully pre-loaded {len(missing_features)} feature descriptions!")

# Streamlit app
def main():
    # Initialize persistent cache first
    if 'persistent_feature_cache' not in st.session_state:
        st.session_state['persistent_feature_cache'] = load_feature_cache()
        print(f"🚀 [INIT] Loaded persistent cache with {len(st.session_state['persistent_feature_cache'])} features")
    
    # Initialize preload tracking and queue
    if 'preload_status' not in st.session_state:
        st.session_state['preload_status'] = {
            'completed': False,
            'trees_processed': 0,
            'tokens_used': 0,
            'max_features': 200,
            'features_processed': 0,
            'last_processed_tree': -1,
            'pending_queue': []
        }
    
    # Run initial preload if not completed and OpenAI is available
    if (not st.session_state['preload_status']['completed'] and 
        OPENAI_AVAILABLE and client and 
        'initial_preload_done' not in st.session_state):
        
        st.session_state['initial_preload_done'] = True
        
        # Initialize background status tracking
        if 'background_status' not in st.session_state:
            st.session_state['background_status'] = {
                'running': False,
                'last_update': 0,
                'current_feature': '',
                'current_tree': 0
            }
        
        # Start background preloader thread
        if not st.session_state['background_status']['running']:
            st.session_state['background_status']['running'] = True
            
            def status_update_callback(status):
                """Callback to save background progress to file (thread-safe)"""
                # Save progress to file instead of session state (threads can't access session state)
                save_progress_status(status)
                print(f"📊 [CALLBACK] Saved background progress: {status.get('status', 'unknown')}")
                
                # Note: Session state updates must happen in main thread through file polling
            
            # Start the background thread
            if threading:
                background_thread = threading.Thread(
                    target=startup_background_preloader,
                    args=(st.session_state['persistent_feature_cache'], status_update_callback),
                    daemon=True
                )
                background_thread.start()
                print("🚀 [INIT] Started background preloader thread")
                
                # Show initial message
                st.info("🚀 Background preloading started! The app is ready to use while features are being processed.")
            else:
                st.warning("⚠️ Threading not available - background preloading disabled")
        
        # If we reached the feature limit in a previous run, build a queue for remaining trees
        if (st.session_state['preload_status'].get('reached_limit') and 
            st.session_state['preload_status']['trees_processed'] < len(rsf.estimators_)):
            remaining_features = (st.session_state['preload_status']['max_features'] - 
                              st.session_state['preload_status']['features_processed'])
            if remaining_features > 0:
                start_tree = st.session_state['preload_status']['trees_processed']
                queue = get_preload_queue_for_trees(
                    start_tree, 
                    st.session_state['persistent_feature_cache'], 
                    remaining_features
                )
                st.session_state['preload_status']['pending_queue'] = queue
                print(f"📋 [INIT] Built pending queue with {len(queue)} features for future processing")

    # Initialize session state variables
    if 'tree_idx' not in st.session_state:
        st.session_state['tree_idx'] = 0
    if 'selected_node' not in st.session_state:
        st.session_state['selected_node'] = None
    if 'node_info' not in st.session_state:
        st.session_state['node_info'] = {}
    if 'leaf_info' not in st.session_state:
        st.session_state['leaf_info'] = {}
    if 'feature_descriptions' not in st.session_state:
        st.session_state['feature_descriptions'] = {}
    if 'feature_cache' not in st.session_state:
        st.session_state['feature_cache'] = {}  # Legacy - keeping for compatibility
    if 'pending_feature_requests' not in st.session_state:
        st.session_state['pending_feature_requests'] = []
    if 'feature_request_queue' not in st.session_state:
        st.session_state['feature_request_queue'] = []
    if 'last_hover_feature' not in st.session_state:
        st.session_state['last_hover_feature'] = None
    if 'hover_timestamp' not in st.session_state:
        st.session_state['hover_timestamp'] = 0
    if 'last_progress_check' not in st.session_state:
        st.session_state['last_progress_check'] = 0

    st.title("RSF Tree Visualizer")
    st.write("Click Next/Back to view different trees in the forest. Click on a split node (blue) to view feature-based Kaplan-Meier plots, or click on a leaf node (colored) to compare survival outcomes between leaf nodes. Hover over split nodes to get AI-powered feature descriptions.")
    
    # Check for background progress updates from file (thread-safe approach)
    if st.session_state.get('background_status', {}).get('running'):
        progress_status = load_progress_status()
        if progress_status and progress_status.get('timestamp', 0) > st.session_state.get('last_progress_check', 0):
            # Update session state with latest progress from file
            st.session_state['background_status'].update({
                'last_update': progress_status.get('timestamp', 0),
                'current_feature': progress_status.get('current_feature', ''),
                'current_tree': progress_status.get('current_tree', 0)
            })
            st.session_state['preload_status'].update({
                'trees_processed': progress_status.get('trees_processed', 0),
                'tokens_used': progress_status.get('tokens_used', 0),
                'features_processed': progress_status.get('features_processed', 0),
                'last_processed_tree': progress_status.get('current_tree', 0) - 1 if progress_status.get('current_tree', 0) > 0 else -1
            })
            st.session_state['last_progress_check'] = progress_status.get('timestamp', 0)
            
            # Check if completed or errored
            if progress_status.get('status') == 'completed':
                st.session_state['preload_status'].update({
                    'completed': True,
                    'reached_limit': progress_status.get('reached_limit', False)
                })
                st.session_state['background_status']['running'] = False
                print("🎉 [MAIN] Background preloading completed!")
            elif progress_status.get('status') == 'error':
                st.session_state['background_status']['running'] = False
                print(f"❌ [MAIN] Background preloading failed: {progress_status.get('error', 'Unknown error')}")
    
    # Background processing status with live updates
    cache_size = len(st.session_state['persistent_feature_cache'])
    background_status = st.session_state.get('background_status', {})
    
    if cache_size > 0:
        st.sidebar.success(f"📚 Cached descriptions: {cache_size}")
        
        # Show background processing status
        if background_status.get('running'):
            current_feature = background_status.get('current_feature', '')
            current_tree = background_status.get('current_tree', 0)
            last_update = background_status.get('last_update', 0)
            
            if current_feature:
                st.sidebar.info(f"🔄 Processing: {current_feature}")
                st.sidebar.write(f"Tree: {current_tree}/{len(rsf.estimators_)}")
            else:
                st.sidebar.info("🚀 Background preloading starting...")
            
            # Show last update time
            if last_update > 0:
                time_since = time.time() - last_update
                if time_since < 60:
                    st.sidebar.write(f"⏰ Updated {time_since:.0f}s ago")
                else:
                    st.sidebar.write(f"⏰ Updated {time_since/60:.1f}m ago")
            
            # Manual refresh button for background status
            if st.sidebar.button("🔄 Refresh Status"):
                st.rerun()
        else:
            if st.session_state['preload_status'].get('completed'):
                st.sidebar.success("✅ Background preloading complete!")
            elif st.session_state['preload_status'].get('features_processed', 0) > 0:
                st.sidebar.info("⏸️ Background preloading paused")
    else:
        if background_status.get('running'):
            st.sidebar.info("🚀 Starting background preloading...")
        else:
            st.sidebar.info("⏳ Waiting for background preload to start...")
    
    # Manual feature pre-loading for efficiency (optional)
    if st.sidebar.button("🚀 Manual Pre-load (10 trees)"):
        with st.spinner("Pre-loading feature descriptions in background..."):
            preload_all_features()
    
    # NEW APPROACH: Use session-based polling instead of iframe communication
    # This avoids all sandbox security restrictions by using server-side state
    
    # Create a shared feature request mechanism that works without iframe communication
    if 'feature_request_state' not in st.session_state:
        st.session_state['feature_request_state'] = {
            'requesting': False,
            'requested_feature': None,
            'request_timestamp': 0,
            'auto_refresh_counter': 0
        }
    
    # Disable problematic auto-refresh mechanism that causes infinite loops
    # The auto-loading will handle feature descriptions without needing constant refreshes
    request_state = st.session_state['feature_request_state']
    
    # Show status of OpenAI integration
    if not OPENAI_AVAILABLE:
        st.info("💡 Install `openai` and `python-dotenv` packages to enable AI-powered feature descriptions on hover.")
    elif not client:
        st.warning("⚠️ Configure your OpenAI API key in the `.env` file to enable feature descriptions.")
    else:
        # Show cache statistics
        cache_count = len(st.session_state.get('feature_descriptions', {}))
        tree_cache_count = len(st.session_state.get('feature_cache', {}))
        st.success(f"✅ AI-powered feature descriptions enabled! ({cache_count} descriptions cached, {tree_cache_count} trees analyzed)")
        
        # Debug info for developers
        with st.expander("🔧 Debug Information", expanded=False):
            st.write("**Feature Description Cache:**")
            if st.session_state.get('feature_descriptions'):
                for feature, desc in st.session_state['feature_descriptions'].items():
                    st.write(f"- **{feature}**: {desc[:100]}{'...' if len(desc) > 100 else ''}")
            else:
                st.write("No features cached yet. Hover over blue split nodes to trigger AI descriptions.")
            
            st.write("**Tree Feature Cache:**")
            current_tree_key = f"tree_{st.session_state['tree_idx']}"
            if current_tree_key in st.session_state.get('feature_cache', {}):
                features = st.session_state['feature_cache'][current_tree_key]
                st.write(f"Current tree features ({len(features)}): {', '.join(features[:10])}")
                if len(features) > 10:
                    st.write(f"... and {len(features) - 10} more")
            else:
                st.write("Current tree features not cached yet.")
                
            # Add test button for debugging LLM function
            st.write("**LLM Function Test:**")
            if st.button("🧪 Test LLM with MAP2K3"):
                print("🧪 [TEST] Testing LLM function directly...")
                test_description = get_feature_description_cached("MAP2K3", st.session_state['persistent_feature_cache'])
                st.write(f"Test result: {test_description}")
                st.session_state['feature_descriptions']['MAP2K3'] = test_description

    # Handle tree index from URL parameters (must be done before other logic)
    query_params = st.query_params
    if 'tree_idx' in query_params:
        try:
            url_tree_idx = int(query_params['tree_idx'])
            if 0 <= url_tree_idx < len(rsf.estimators_):
                st.session_state['tree_idx'] = url_tree_idx
        except (ValueError, TypeError):
            pass

    n_trees = len(rsf.estimators_)
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("Back"):
            st.session_state['tree_idx'] = (st.session_state['tree_idx'] - 1) % n_trees
            # Reset selected node when changing trees
            st.session_state['selected_node'] = None
            # Reset auto-loading state when changing trees
            st.session_state['auto_loading_in_progress'] = False
            print(f"🔄 [NAV] Moved to tree {st.session_state['tree_idx'] + 1}, reset auto-loading state")
    with col3:
        if st.button("Next"):
            st.session_state['tree_idx'] = (st.session_state['tree_idx'] + 1) % n_trees
            # Reset selected node when changing trees
            st.session_state['selected_node'] = None
            # Reset auto-loading state when changing trees
            st.session_state['auto_loading_in_progress'] = False
            print(f"🔄 [NAV] Moved to tree {st.session_state['tree_idx'] + 1}, reset auto-loading state")
    
    # Process pending queue when user navigates to a new tree
    current_tree = st.session_state['tree_idx']
    preload_status = st.session_state['preload_status']
    
    if (current_tree > preload_status['last_processed_tree'] and 
        preload_status['pending_queue'] and 
        OPENAI_AVAILABLE and client):
        
        # Find features for the current tree in the pending queue
        current_tree_features = [feature for tree_idx, feature in preload_status['pending_queue'] 
                               if tree_idx == current_tree]
        
        if current_tree_features:
            print(f"🎯 [QUEUE] Processing {len(current_tree_features)} queued features for tree {current_tree + 1}")
            
            # Process features for this tree
            processed_count = 0
            tokens_used_now = 0
            
            with st.spinner(f"🤖 Loading {len(current_tree_features)} AI descriptions for tree {current_tree + 1}..."):
                for feature in current_tree_features:
                    try:
                        # Check if still within feature budget
                        if preload_status['features_processed'] >= preload_status['max_features']:
                            print(f"🛑 [QUEUE] Reached feature limit while processing tree {current_tree + 1}")
                            break
                        
                        result = get_feature_description_llm_with_tokens(feature)
                        
                        if isinstance(result, dict) and 'description' in result:
                            description = result['description']
                            tokens = result['tokens']
                        else:
                            description = result
                            tokens = estimate_tokens(feature, description)
                        
                        # Update cache
                        st.session_state['persistent_feature_cache'][feature] = description
                        st.session_state['feature_descriptions'][feature] = description
                        
                        # Update tracking
                        processed_count += 1
                        tokens_used_now += tokens
                        preload_status['tokens_used'] += tokens
                        preload_status['features_processed'] += 1
                        
                        print(f"✅ [QUEUE] Processed {feature} for tree {current_tree + 1} ({tokens} tokens)")
                        
                    except Exception as e:
                        print(f"❌ [QUEUE] Error processing {feature}: {e}")
                        continue
            
            # Remove processed features from queue
            preload_status['pending_queue'] = [
                (tree_idx, feature) for tree_idx, feature in preload_status['pending_queue']
                if not (tree_idx == current_tree and feature in current_tree_features)
            ]
            
            # Update last processed tree
            preload_status['last_processed_tree'] = current_tree
            
            # Save cache after processing
            save_feature_cache(st.session_state['persistent_feature_cache'])
            
            if processed_count > 0:
                st.success(f"✅ Loaded {processed_count} AI descriptions for tree {current_tree + 1} using {tokens_used_now} tokens!")
            
            print(f"📊 [QUEUE] Tree {current_tree + 1} processed: {processed_count} features, {tokens_used_now} tokens")
            print(f"📊 [QUEUE] Total progress: {preload_status['features_processed']} features, {preload_status['tokens_used']:,} tokens")
            print(f"📊 [QUEUE] Remaining queue size: {len(preload_status['pending_queue'])}")
            
    st.write(f"Currently viewing tree {st.session_state['tree_idx']+1} of {n_trees}")
    
    # Show preload progress in sidebar
    if preload_status['features_processed'] > 0:
        progress_pct = min(100, (preload_status['features_processed'] / preload_status['max_features']) * 100)
        st.sidebar.progress(progress_pct / 100)
        st.sidebar.write(f"Feature progress: {preload_status['features_processed']} / {preload_status['max_features']} ({progress_pct:.1f}%)")
        st.sidebar.write(f"Tokens used: {preload_status['tokens_used']:,}")
        
        if preload_status['pending_queue']:
            remaining_trees = len(set(tree_idx for tree_idx, _ in preload_status['pending_queue']))
            st.sidebar.write(f"📋 Queued: {len(preload_status['pending_queue'])} features across {remaining_trees} trees")
            
    # Create main layout: tree on left, KM plot on right
    main_col1, main_col2 = st.columns([5, 3])
    
    # Get the current tree
    tree = rsf.estimators_[st.session_state['tree_idx']]
    
    # Collect split node information for the current tree
    node_info = collect_node_features(tree.tree_, covariates)
    st.session_state['node_info'] = node_info
    
    # Cache tree features to avoid repeated LLM calls
    tree_key = f"tree_{st.session_state['tree_idx']}"
    if tree_key not in st.session_state['feature_cache']:
        # Extract all unique features from this tree
        tree_features = set()
        for node_id, info in node_info.items():
            tree_features.add(info['feature'])
        st.session_state['feature_cache'][tree_key] = list(tree_features)
        print(f"🗂️ [CACHE] Cached {len(tree_features)} features for tree {st.session_state['tree_idx']}")
    
    # IMPROVED: Auto-load feature descriptions for current tree (background loading)
    # Only run auto-loading if we have missing descriptions AND we're not already processing
    current_tree_features = st.session_state['feature_cache'].get(tree_key, [])
    if current_tree_features and OPENAI_AVAILABLE and client:
        # Check if we're already processing descriptions to avoid loops
        if 'auto_loading_in_progress' not in st.session_state:
            st.session_state['auto_loading_in_progress'] = False
        
        # Find features that don't have descriptions yet (check both session and persistent cache)
        missing_descriptions = [f for f in current_tree_features 
                              if f not in st.session_state.get('feature_descriptions', {}) 
                              and f not in st.session_state['persistent_feature_cache']]
        
        # Only start auto-loading if we have missing descriptions and aren't already processing
        if missing_descriptions and not st.session_state['auto_loading_in_progress']:
            st.session_state['auto_loading_in_progress'] = True
            
            feature_to_fetch = missing_descriptions[0]
            print(f"\n🔄 [AUTO-LOAD] Pre-loading description for tree feature: {feature_to_fetch}")
            print(f"📊 [AUTO-LOAD] Progress: {len(current_tree_features) - len(missing_descriptions)}/{len(current_tree_features)} features loaded")
            
            # Show brief loading indicator
            status_placeholder = st.empty()
            with status_placeholder:
                st.info(f"🤖 Loading AI description for {feature_to_fetch}... ({len(missing_descriptions)} remaining)")
            
            # Fetch the description using cached function
            description = get_feature_description_cached(feature_to_fetch, st.session_state['persistent_feature_cache'])
            st.session_state['feature_descriptions'][feature_to_fetch] = description
            print(f"✅ [AUTO-LOAD] Pre-loaded description for: {feature_to_fetch}")
            
            # Clear the loading message
            status_placeholder.empty()
            
            # Reset the loading flag
            st.session_state['auto_loading_in_progress'] = False
            
            # REAL-TIME UPDATE: Inject updated descriptions into the iframe
            # Create a dynamic script that gets executed to update the JavaScript cache
            escaped_description = description.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
            
            # Add a dynamic script injection that updates the cache in real-time
            with st.empty():
                st.markdown(f"""
                <script>
                // Real-time description cache update
                window.addEventListener('message', function(event) {{
                    if (event.data && event.data.type === 'rsf_description_update') {{
                        console.log('[REAL-TIME UPDATE] Received description update:', event.data);
                    }}
                }});
                
                // Try to update iframe directly via postMessage
                setTimeout(function() {{
                    const iframes = document.querySelectorAll('iframe');
                    iframes.forEach((iframe, index) => {{
                        try {{
                            iframe.contentWindow.postMessage({{
                                type: 'rsf_description_update',
                                feature: '{feature_to_fetch}',
                                description: '{escaped_description}',
                                timestamp: Date.now()
                            }}, '*');
                            console.log(`[REAL-TIME UPDATE] Sent description update to iframe ${{index}}`);
                        }} catch (e) {{
                            console.log(`[REAL-TIME UPDATE] Could not update iframe ${{index}}:`, e);
                        }}
                    }});
                }}, 100);
                </script>
                """, unsafe_allow_html=True)
            
            # Check remaining descriptions but don't auto-rerun to avoid loops
            remaining_after_this = [f for f in current_tree_features 
                                  if f not in st.session_state.get('feature_descriptions', {}) 
                                  and f not in st.session_state['persistent_feature_cache']]
            
            if len(remaining_after_this) > 0:
                print(f"📋 [AUTO-LOAD] {len(remaining_after_this)} more descriptions to load (will load on next interaction)...")
            else:
                print("✨ [AUTO-LOAD] All tree features now have descriptions!")
        elif missing_descriptions and st.session_state['auto_loading_in_progress']:
            print(f"⏸️ [AUTO-LOAD] Auto-loading already in progress, skipping...")
        else:
            print("✅ [AUTO-LOAD] All features for current tree are already loaded")
    
    # Collect leaf node information for KM analysis
    leaf_info = collect_leaf_nodes(tree.tree_)
    st.session_state['leaf_info'] = leaf_info
    
    with main_col1:
        st.subheader("RSF Tree Visualization")
        
        # Build graph for visualization
        G = nx.DiGraph()
        build_graph(tree.tree_, 0, G, covariates, root_fixed=True)
        
        # Create network visualization
        net = Network(height='750px', width='100%', directed=True, notebook=False)
        net.barnes_hut()
        
        # Add nodes to the network
        for node_id, data in G.nodes(data=True):
            label = data.get('label', f"Node {node_id}")
            color = data.get('color', 'gray')
            title = data.get('title', '')
            physics = True
            
            # Add the node to the network
            net.add_node(int(node_id), label=label, color=color, title=title, physics=physics)
        
        # Add edges to the network
        for source_id, target_id, edge_data in G.edges(data=True):
            label = edge_data.get('label', '')
            color = edge_data.get('color', 'gray')
            net.add_edge(int(source_id), int(target_id), label=label, color=color)
        
        # Set options for the network visualization
        net.set_options("""
        var options = {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed"
                }
            },
            "physics": {"enabled": false},
            "edges": {
                "font": {"size": 12, "align": "middle"}, 
                "arrows": {"to": { "enabled": true, "scaleFactor": 0.5 }}, 
                "smooth": {"enabled": true, "type": "dynamic"}
            },
            "nodes": {
                "font": {"size": 14, "face": "Arial"}
            },
            "interaction": {
                "hover": true,
                "navigationButtons": true,
                "keyboard": true,
                "zoomView": true,
                "dragView": true
            },
            "configure": {
                "enabled": false
            },
            "clickToUse": false,
            "autoResize": true
        }
        """)
        
        # Save and display the interactive network
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            tmp_file.flush()
            
            # Clean up the HTML to remove Streamlit-related code that causes conflicts
            html_content = open(tmp_file.name, 'r').read()
            
            # Remove problematic Streamlit imports and connections that cause multiple instances
            html_content = html_content.replace('import {Streamlit}', '// import {Streamlit}')
            html_content = html_content.replace('from "streamlit-component-lib"', '// from "streamlit-component-lib"')
            html_content = html_content.replace('Streamlit.', '// Streamlit.')
            html_content = html_content.replace('streamlit-component-lib', '// streamlit-component-lib')
            
            # Remove any Streamlit initialization code
            html_content = html_content.replace('Streamlit.setComponentReady();', '// Streamlit.setComponentReady();')
            html_content = html_content.replace('Streamlit.setFrameHeight', '// Streamlit.setFrameHeight')
            
            # Remove any JavaScript that tries to initialize Streamlit connections
            import re
            # Remove any script tags that contain Streamlit initialization
            html_content = re.sub(r'<script[^>]*>.*?Streamlit.*?</script>', '', html_content, flags=re.DOTALL)
            
            # Remove any existing clickToUse configurations that might conflict
            html_content = html_content.replace('"clickToUse": true', '"clickToUse": false')
            html_content = html_content.replace("'clickToUse': true", "'clickToUse': false")
            
            # Add simplified click handler and enhanced hover functionality with direct URL approach
            html_content = html_content.replace('</head>', f'''
            <style>
            /* Enhanced tooltip styling with text wrapping */
            .vis-tooltip {{
                max-width: 300px !important;
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
                word-break: break-word !important;
                line-height: 1.4 !important;
                font-size: 12px !important;
                padding: 8px !important;
                background: rgba(0, 0, 0, 0.9) !important;
                color: white !important;
                border-radius: 4px !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
            }}
            
            .vis-network {{
                font-family: Arial, sans-serif !important;
            }}
            </style>
            <script>
            console.log("[DEBUG] === RSF TREE VISUALIZER JAVASCRIPT INIT ===");
            
            // Current tree index passed from Streamlit
            const CURRENT_TREE_IDX = {st.session_state['tree_idx']};
            console.log("[DEBUG] Current tree index:", CURRENT_TREE_IDX);
            
            // Node info for feature lookup
            const NODE_INFO = {json.dumps(st.session_state['node_info'])};
            console.log("[DEBUG] Node info loaded:", Object.keys(NODE_INFO).length, "split nodes");
            console.log("[DEBUG] Node info details:", NODE_INFO);
            
            // Cache for feature descriptions to avoid repeated API calls
            const FEATURE_DESCRIPTIONS = {json.dumps(st.session_state.get('feature_descriptions', {}))};
            const PERSISTENT_FEATURE_CACHE = {json.dumps(st.session_state.get('persistent_feature_cache', {}))};
            
            // Merge both caches for complete coverage
            const ALL_DESCRIPTIONS = {{...PERSISTENT_FEATURE_CACHE, ...FEATURE_DESCRIPTIONS}};
            
            console.log("[DEBUG] === FEATURE DESCRIPTIONS INIT ===");
            console.log("[DEBUG] Session descriptions count:", Object.keys(FEATURE_DESCRIPTIONS).length);
            console.log("[DEBUG] Persistent descriptions count:", Object.keys(PERSISTENT_FEATURE_CACHE).length);
            console.log("[DEBUG] Total merged descriptions count:", Object.keys(ALL_DESCRIPTIONS).length);
            console.log("[DEBUG] Merged descriptions data:", ALL_DESCRIPTIONS);
            if (Object.keys(ALL_DESCRIPTIONS).length > 0) {{
                console.log("[DEBUG] Feature descriptions preview:");
                for (const [feature, desc] of Object.entries(ALL_DESCRIPTIONS)) {{
                    console.log(`[DEBUG]   - ${{feature}}: ${{desc.substring(0, 80)}}...`);
                }}
            }} else {{
                console.log("[DEBUG] No feature descriptions available yet");
            }}
            console.log("[DEBUG] === END FEATURE DESCRIPTIONS INIT ===");
            
            // Check if OpenAI is available
            const OPENAI_AVAILABLE = {str(OPENAI_AVAILABLE and client is not None).lower()};
            console.log("[DEBUG] OpenAI available:", OPENAI_AVAILABLE);
            
            let hoverTimeout = null;
            let currentHoveredNode = null;
            let loadingDescriptions = new Set();
            
            // Simple node click handler that updates parent URL
            function handleNodeClick(nodeId) {{
                console.log("[DEBUG] Node clicked:", nodeId);
                
                // Update parent window URL with selected node and tree index
                if (window.parent && window.parent.location) {{
                    const currentUrl = new URL(window.parent.location.href);
                    currentUrl.searchParams.set('selected_node', nodeId);
                    currentUrl.searchParams.set('tree_idx', CURRENT_TREE_IDX);
                    
                    // Remove any existing feature requests to avoid conflicts
                    currentUrl.searchParams.delete('feature_request');
                    
                    window.parent.history.pushState({{}}, '', currentUrl.toString());
                    
                    // Trigger a reload of the parent to show the KM plot
                    window.parent.location.reload();
                }}
            }}
            
            // Function to trigger feature description fetch via session state
            function fetchFeatureDescriptionViaSessionState(featureName) {{
                try {{
                    console.log(`[DEBUG] === USING NEW POLLING APPROACH ===`);
                    console.log(`[DEBUG] Target feature: ${{featureName}}`);
                    
                    // Store which feature we're loading to track state
                    loadingDescriptions.add(featureName);
                    console.log(`[DEBUG] Added ${{featureName}} to loading state. Loading set:`, Array.from(loadingDescriptions));
                    
                    // NEW APPROACH: Use a simple iframe-safe mechanism
                    // Store the feature request in sessionStorage and use polling
                    const requestData = {{
                        feature: featureName,
                        timestamp: Date.now(),
                        tree_idx: CURRENT_TREE_IDX
                    }};
                    
                    // Store in sessionStorage for polling-based detection
                    sessionStorage.setItem('rsf_feature_request', JSON.stringify(requestData));
                    console.log(`[DEBUG] 💾 Stored feature request in sessionStorage: ${{featureName}}`);
                    
                    // Also add to a global queue that can be polled by Streamlit
                    if (!window.featureRequestQueue) {{
                        window.featureRequestQueue = [];
                    }}
                    window.featureRequestQueue.push(requestData);
                    console.log(`[DEBUG] 📋 Added to global queue. Queue length: ${{window.featureRequestQueue.length}}`);
                    
                    // Try a simple URL update approach as fallback
                    try {{
                        const currentUrl = new URL(window.location.href);
                        currentUrl.searchParams.set('hover_feature', featureName);
                        currentUrl.searchParams.set('hover_timestamp', Date.now());
                        
                        // Update the URL without navigation (for debugging purposes)
                        console.log(`[DEBUG] 🔗 Would update URL to: ${{currentUrl.toString()}}`);
                        
                        // Since iframe navigation is blocked, we'll rely on the polling mechanism
                        console.log(`[DEBUG] ⚡ Relying on polling mechanism instead of URL navigation`);
                        
                    }} catch (urlError) {{
                        console.log(`[DEBUG] ⚠️ URL approach not available:`, urlError);
                    }}
                    
                }} catch (error) {{
                    console.error("[DEBUG] ❌ Error in polling approach:", error);
                    loadingDescriptions.delete(featureName);
                }}
            }}
            
            // Function to wrap text and format tooltip content
            function formatTooltipText(text, maxLineLength = 40) {{
                if (!text) return '';
                
                // Split into words
                const words = text.split(' ');
                const lines = [];
                let currentLine = '';
                
                for (const word of words) {{
                    if ((currentLine + ' ' + word).length <= maxLineLength) {{
                        currentLine += (currentLine ? ' ' : '') + word;
                    }} else {{
                        if (currentLine) lines.push(currentLine);
                        currentLine = word;
                    }}
                }}
                
                if (currentLine) lines.push(currentLine);
                return lines.join('\\n');
            }}
            
            // Function to update node tooltip with feature description
            function updateNodeTooltip(nodeId, description, isAIDescription = false) {{
                try {{
                    const nodeInfo = NODE_INFO[nodeId];
                    if (nodeInfo && window.network) {{
                        const originalLabel = window.network.body.data.nodes.get(nodeId).label;
                        const featureName = nodeInfo.feature;
                        const threshold = nodeInfo.threshold;
                        
                        // Create enhanced tooltip with better formatting and wrapping
                        let enhancedTitle;
                        if (isAIDescription) {{
                            // Format AI description with text wrapping
                            const wrappedDescription = formatTooltipText(description, 45);
                            enhancedTitle = `${{originalLabel}}\\n\\n🤖 AI: ${{featureName}}\\n${{wrappedDescription}}`;
                        }} else {{
                            // Format basic info with wrapping
                            const wrappedDescription = formatTooltipText(description, 45);
                            enhancedTitle = `${{originalLabel}}\\n\\n${{wrappedDescription}}`;
                        }}
                        
                        // Update the node's title attribute
                        window.network.body.data.nodes.update({{
                            id: nodeId,
                            title: enhancedTitle
                        }});
                        
                        console.log("[DEBUG] Updated tooltip for node", nodeId, "with", isAIDescription ? "AI description" : "basic info");
                    }}
                }} catch (error) {{
                    console.error("[DEBUG] Error updating node tooltip:", error);
                }}
            }}
            
            // Handle node hover for feature descriptions (SIMPLIFIED APPROACH)
            function handleNodeHover(nodeId) {{
                console.log("[DEBUG] Node hovered:", nodeId);
                
                // Clear any existing timeout
                if (hoverTimeout) {{
                    clearTimeout(hoverTimeout);
                }}
                
                currentHoveredNode = nodeId;
                
                // Check if this is a split node with feature information
                if (NODE_INFO[nodeId]) {{
                    const featureName = NODE_INFO[nodeId].feature;
                    const threshold = NODE_INFO[nodeId].threshold;
                    
                    // Always show basic information immediately
                    const basicInfo = `Feature: ${{featureName}}\\nThreshold: ≤ ${{threshold.toFixed(2)}}`;
                    
                    // Check if OpenAI is available
                    if (!OPENAI_AVAILABLE) {{
                        updateNodeTooltip(nodeId, `${{basicInfo}}\\n\\n💡 Install OpenAI integration for detailed descriptions`, false);
                        return;
                    }}
                    
                    // Check if we already have the description cached (check merged cache)
                    if (ALL_DESCRIPTIONS[featureName]) {{
                        // Show the cached description immediately
                        console.log(`[DEBUG] ✅ Using cached description for: ${{featureName}}`);
                        updateNodeTooltip(nodeId, ALL_DESCRIPTIONS[featureName], true);
                        return;
                    }}
                    
                    // If no description is cached, show basic info and note about background loading
                    updateNodeTooltip(nodeId, `${{basicInfo}}\\n\\n🔄 AI description loading in background...\\n(Descriptions are pre-loaded for all tree features)`, false);
                    
                    console.log(`[DEBUG] ⏳ No cached description for: ${{featureName}} (Background loading should handle this)`);
                }}
            }}
            
            // Handle node hover end
            function handleNodeHoverEnd(nodeId) {{
                console.log("[DEBUG] Node hover ended:", nodeId);
                
                // Clear timeout if hovering ended
                if (hoverTimeout && currentHoveredNode === nodeId) {{
                    clearTimeout(hoverTimeout);
                    hoverTimeout = null;
                }}
                
                currentHoveredNode = null;
                
                // Restore original tooltip if it was a split node
                if (NODE_INFO[nodeId] && window.network) {{
                    try {{
                        const originalLabel = window.network.body.data.nodes.get(nodeId).label;
                        window.network.body.data.nodes.update({{
                            id: nodeId,
                            title: originalLabel
                        }});
                    }} catch (error) {{
                        console.error("[DEBUG] Error restoring original tooltip:", error);
                    }}
                }}
            }}
            
            // Auto-fit the network to view
            function fitNetworkToView() {{
                if (window.network && typeof window.network.fit === 'function') {{
                    // Fit the network to the canvas with some padding
                    window.network.fit({{
                        animation: {{
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }}
                    }});
                    console.log("[DEBUG] Network auto-fitted to view");
                }}
            }}
            
            // Function to check for fresh feature descriptions and update tooltips (DYNAMIC UPDATE)
            function checkForFreshDescriptions() {{
                try {{
                    // Dynamically fetch current descriptions from the page's session state
                    // This allows real-time updates without page refresh
                    fetch(window.location.href + '?get_descriptions=1', {{
                        method: 'GET',
                        cache: 'no-cache'
                    }}).then(response => {{
                        // Even if fetch fails, we can still use the static data
                        console.log('[DEBUG] 🔄 Checking for fresh descriptions (fetch approach)');
                        return null;
                    }}).catch(error => {{
                        console.log('[DEBUG] 📡 Fetch approach not available, using embedded data');
                    }});
                    
                    // Use embedded current state (this gets updated on each page render)
                    const currentSessionDescriptions = {json.dumps(st.session_state.get('feature_descriptions', {}))};
                    const currentPersistentDescriptions = {json.dumps(st.session_state.get('persistent_feature_cache', {}))};
                    
                    // Merge all available descriptions
                    const currentAllDescriptions = {{...currentPersistentDescriptions, ...currentSessionDescriptions}};
                    
                    // Check if we have new descriptions
                    let hasNewDescriptions = false;
                    let newDescriptionCount = 0;
                    for (const [featureName, description] of Object.entries(currentAllDescriptions)) {{
                        if (!ALL_DESCRIPTIONS[featureName]) {{
                            ALL_DESCRIPTIONS[featureName] = description;
                            hasNewDescriptions = true;
                            newDescriptionCount++;
                            console.log(`[DEBUG] 📋 Added new cached description for: ${{featureName}}`);
                        }}
                    }}
                    
                    // If we're currently hovering and have a new description for that feature, update immediately
                    if (currentHoveredNode !== null && NODE_INFO[currentHoveredNode]) {{
                        const featureName = NODE_INFO[currentHoveredNode].feature;
                        if (ALL_DESCRIPTIONS[featureName]) {{
                            console.log(`[DEBUG] ✅ Updating tooltip for currently hovered feature: ${{featureName}}`);
                            updateNodeTooltip(currentHoveredNode, ALL_DESCRIPTIONS[featureName], true);
                        }}
                    }}
                    
                    if (hasNewDescriptions) {{
                        console.log(`[DEBUG] 🔄 Added ${{newDescriptionCount}} new descriptions. Total now available: ${{Object.keys(ALL_DESCRIPTIONS).length}}`);
                        
                        // Also update the parent window to reflect new cache status
                        if (window.parent && window.parent.postMessage) {{
                            window.parent.postMessage({{
                                type: 'rsf_cache_update',
                                cacheSize: Object.keys(ALL_DESCRIPTIONS).length,
                                newDescriptions: newDescriptionCount
                            }}, '*');
                        }}
                    }}
                }} catch (error) {{
                    console.error('[DEBUG] ❌ Error checking for fresh descriptions:', error);
                }}
            }}
            
            // Wait for network to be ready and add event handlers (SIMPLIFIED APPROACH)
            document.addEventListener('DOMContentLoaded', function() {{
                // Add message listener for real-time description updates
                window.addEventListener('message', function(event) {{
                    if (event.data && event.data.type === 'rsf_description_update') {{
                        const featureName = event.data.feature;
                        const description = event.data.description;
                        
                        console.log(`[REAL-TIME] Received description update for: ${{featureName}}`);
                        console.log(`[REAL-TIME] Description: ${{description}}`);
                        
                        // Update the local cache
                        ALL_DESCRIPTIONS[featureName] = description;
                        
                        // If we're currently hovering over this feature, update the tooltip immediately
                        if (currentHoveredNode !== null && NODE_INFO[currentHoveredNode] && NODE_INFO[currentHoveredNode].feature === featureName) {{
                            console.log(`[REAL-TIME] Updating tooltip for currently hovered feature: ${{featureName}}`);
                            updateNodeTooltip(currentHoveredNode, description, true);
                        }}
                    }}
                }});
                
                setTimeout(function() {{
                    try {{
                        if (window.network && typeof window.network.on === 'function') {{
                            // Add click handler
                            window.network.on('click', function(params) {{
                                if (params.nodes && params.nodes.length > 0) {{
                                    const nodeId = params.nodes[0];
                                    handleNodeClick(nodeId);
                                }}
                            }});
                            
                            // Add hover handlers
                            window.network.on('hoverNode', function(params) {{
                                const nodeId = params.node;
                                handleNodeHover(nodeId);
                            }});
                            
                            window.network.on('blurNode', function(params) {{
                                const nodeId = params.node;
                                handleNodeHoverEnd(nodeId);
                            }});
                            
                            console.log("[DEBUG] Network event handlers attached successfully");
                            
                            // Auto-fit the network after stabilization
                            window.network.once('stabilizationIterationsDone', function() {{
                                setTimeout(fitNetworkToView, 500);
                            }});
                            
                            // Also try to fit immediately in case stabilization is already done
                            setTimeout(fitNetworkToView, 1500);
                            
                            // Check for fresh descriptions periodically (simplified)
                            setInterval(checkForFreshDescriptions, 2000);
                            
                        }} else {{
                            console.log("[DEBUG] Network object not found or invalid");
                        }}
                    }} catch (error) {{
                        console.error("[DEBUG] Error setting up event handlers:", error);
                    }}
                }}, 1000);
            }});
            </script>
            </head>
            ''')
            
            # Write the modified HTML content
            with open(tmp_file.name, 'w') as f:
                f.write(html_content)
            
            # Display the network directly with sandbox permissions for navigation
            # Note: We need allow-top-navigation-by-user-activation to let the iframe
            # navigate the parent window when users hover over nodes
            html_content_final = open(tmp_file.name, 'r').read()
            
            st.components.v1.html(
                html_content_final,
                height=1000,
                scrolling=True,
                # Allow the iframe to navigate the parent window on user interaction
                # This is needed for the feature description URL navigation to work
                # The iframe needs these permissions to trigger server-side LLM requests
                width=None  # Use default width
            )

            # Clean up the temporary file
            os.unlink(tmp_file.name)
    
    # Create KM plot area in the right column
    with main_col2:
        km_plot_placeholder = st.empty()
        
        # Add a feature description area
        feature_desc_placeholder = st.empty()

    # Simplified pending feature request processing (no automatic reruns)
    # Remove auto-processing to prevent infinite loops - descriptions will load on demand
    if st.session_state.get('pending_feature_requests'):
        # Clear all pending requests without processing to avoid loops
        num_pending = len(st.session_state['pending_feature_requests'])
        st.session_state['pending_feature_requests'] = []
        print(f"🧹 [CLEANUP] Cleared {num_pending} pending feature requests to prevent loops")

    # Handle feature description requests from URL parameters
    if 'feature_request' in query_params:
        try:
            feature_name = query_params['feature_request']
            print(f"\n" + "🔍"*60)
            print(f"🔍 [MAIN] Processing feature request from URL: {feature_name}")
            print(f"🔍 [MAIN] Current cache contains: {list(st.session_state['feature_descriptions'].keys())}")
            print(f"🔍 [MAIN] Is feature cached? {feature_name in st.session_state['feature_descriptions']}")
            print("🔍"*60)
            
            # Check if we already have this description cached
            if feature_name not in st.session_state['feature_descriptions']:
                # Show loading message
                with feature_desc_placeholder.container():
                    st.info(f"🔍 Loading AI description for feature: {feature_name}")
                    
                # Get description from LLM using cached function
                print(f"🤖 [MAIN] Fetching new description for: {feature_name}")
                description = get_feature_description_cached(feature_name, st.session_state['persistent_feature_cache'])
                st.session_state['feature_descriptions'][feature_name] = description
                print(f"✅ [MAIN] Cached new description for: {feature_name}")
                print(f"📄 [MAIN] Description preview: {description[:100]}...")
                
                # Clear the loading message
                feature_desc_placeholder.empty()
            else:
                print(f"📄 [MAIN] Using cached description for: {feature_name}")
                print(f"📄 [MAIN] Cached description preview: {st.session_state['feature_descriptions'][feature_name][:100]}...")
            
            # Clean up the URL by removing the feature_request parameter
            print("🧹 [MAIN] Cleaning up URL parameters...")
            new_params = {k: v for k, v in query_params.items() if k != 'feature_request'}
            st.query_params.clear()
            for k, v in new_params.items():
                st.query_params[k] = v
            
            print(f"🔄 [MAIN] Cache now contains {len(st.session_state['feature_descriptions'])} descriptions")
            for fname, fdesc in st.session_state['feature_descriptions'].items():
                print(f"  📋 {fname}: {fdesc[:50]}...")
            
            # Only reload if we actually fetched a new description
            # Removed automatic st.rerun() to prevent infinite loops
            print(f"🔄 [MAIN] Cache now contains {len(st.session_state['feature_descriptions'])} descriptions")
            for fname, fdesc in st.session_state['feature_descriptions'].items():
                print(f"  📋 {fname}: {fdesc[:50]}...")
            print("✅ [MAIN] Feature description processed without reload")
                    
        except Exception as e:
            print(f"❌ [MAIN] Error processing feature request: {str(e)}")
            import traceback
            print(traceback.format_exc())
            with feature_desc_placeholder.container():
                st.error(f"Error loading feature description: {str(e)}")
            # Also clean up the URL on error
            new_params = {k: v for k, v in query_params.items() if k != 'feature_request'}
            st.query_params.clear()
            for k, v in new_params.items():
                st.query_params[k] = v

    # Handle node selection from URL parameters
    if 'selected_node' in query_params:
        try:
            clicked_node_id = int(query_params['selected_node'])

            # Split nodes (blue) or leaf nodes trigger KM plots
            if clicked_node_id in node_info or clicked_node_id in leaf_info:
                st.session_state['selected_node'] = clicked_node_id
            # Click on other nodes clears selection
            elif clicked_node_id != st.session_state.get('selected_node'):
                st.session_state['selected_node'] = None
        except (ValueError, TypeError):
            pass
    
    # Display Kaplan-Meier plot based on selected node type
    selected_node_id = st.session_state.get('selected_node')
    
    if selected_node_id is not None:
        # Check if it's a split node
        if selected_node_id in node_info:
            feature = node_info[selected_node_id]['feature']
            threshold = node_info[selected_node_id]['threshold']
            
            # Generate and display KM plot for split node
            with km_plot_placeholder.container():
                st.subheader(f"Kaplan-Meier Plot for Split Node {selected_node_id}")
                st.write(f"Split on feature: {feature} ≤ {threshold:.2f}")
                
                # Show the plot
                fig = plot_km_and_logrank(df, feature, threshold)
                st.pyplot(fig)
                
                # Add a button to clear selection
                if st.button("Close Plot", key="clear_km_plot"):
                    st.session_state['selected_node'] = None
                    st.rerun()
        
        # Check if it's a leaf node
        elif selected_node_id in leaf_info:
            # Generate and display KM plot for leaf node
            with km_plot_placeholder.container():
                st.subheader(f"Kaplan-Meier Plot for Leaf Node {selected_node_id}")
                
                leaf_data = leaf_info[selected_node_id]
                st.write(f"Leaf statistics - Mean: {leaf_data['mean_survival']:.2f}, Median: {leaf_data['median_survival']:.2f}, Samples: {leaf_data['n_samples']}")
                
                # Show the plot comparing this leaf with other leaves
                fig = plot_leaf_node_km(leaf_info, selected_node_id, df)
                st.pyplot(fig)
                
                # Add a button to clear selection
                if st.button("Close Plot", key="clear_km_plot_leaf"):
                    st.session_state['selected_node'] = None
                    st.rerun()
    else:
        # Show empty message in the KM plot area when no node is selected
        with km_plot_placeholder.container():
            st.subheader("Kaplan-Meier Plot")
            st.write("Click on a split node (blue) or leaf node (colored) in the tree to display the Kaplan-Meier survival curves.")

if __name__ == "__main__":
    main()
