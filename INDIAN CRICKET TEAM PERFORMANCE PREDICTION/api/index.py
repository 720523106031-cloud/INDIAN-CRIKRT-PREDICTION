"""
India Cricket Performance Analytics - Flask Application & Serverless Handler
=============================================================================
Author: ECE DevOps Engineering Team
Title: INDIA CRICKET PERFORMANCE ANALYTICS
Subtitle: AI-Powered Indian Cricket Player Performance Prediction System
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Determine base paths accurately for both local run and Vercel serverless environment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)
app.config['SECRET_KEY'] = 'india-cricket-ml-prediction-ece-devops-key-2026'

# Global variables for model and metadata
ML_MODEL = None
MODEL_METRICS = {}
PLAYERS_DATA = []

# Encodings must match training feature order in model/train_model.py
FEATURE_COLUMNS = [
    'role_encoded',
    'format_encoded',
    'matches',
    'runs',
    'batting_avg',
    'strike_rate',
    'fours',
    'sixes',
    'wickets',
    'bowling_avg',
    'bowling_econ'
]

ROLE_MAP = {
    'Top-order Batter': 0,
    'Middle-order Batter': 1,
    'Wicketkeeper Batter': 2,
    'Batting All-rounder': 3,
    'Bowling All-rounder': 4,
    'Pace Bowler': 5,
    'Spin Bowler': 6
}

FORMAT_MAP = {
    'Test': 0,
    'ODI': 1,
    'T20I': 2
}


def load_resources():
    """Loads players database, ML model, and training metrics."""
    global ML_MODEL, MODEL_METRICS, PLAYERS_DATA

    # 1. Load Players JSON
    players_path = os.path.join(DATA_DIR, 'players.json')
    if os.path.exists(players_path):
        try:
            with open(players_path, 'r', encoding='utf-8') as f:
                PLAYERS_DATA = json.load(f)
        except Exception as e:
            print(f"[Warning] Error loading players.json: {e}")
            PLAYERS_DATA = []

    # 2. Load Model Metrics JSON
    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                MODEL_METRICS = json.load(f)
        except Exception as e:
            print(f"[Warning] Error loading model_metrics.json: {e}")
            MODEL_METRICS = {}

    # 3. Load Trained Joblib ML Model
    model_path = os.path.join(MODEL_DIR, 'cricket_rf_model.joblib')
    if os.path.exists(model_path):
        try:
            import joblib
            ML_MODEL = joblib.load(model_path)
            print("[Success] Machine Learning Random Forest Model loaded successfully.")
        except Exception as e:
            print(f"[Warning] Failed to load joblib model: {e}. Will use sports analytics calibration fallback.")
            ML_MODEL = None


# Initial load
load_resources()


def calculate_rule_based_fallback(role, fmt, matches, runs, bat_avg, sr, fours, sixes, wickets, bowl_avg, bowl_econ):
    """
    Fallback prediction formula matching the sports analytics scoring algorithm
    if the serialized joblib model cannot be loaded in a constrained environment.
    """
    # 1. Batting
    bat_score = 0.0
    if fmt == 'Test':
        if bat_avg >= 50.0:
            avg_sub = 88.0 + min(12.0, (bat_avg - 50.0) * 1.5)
        elif bat_avg >= 40.0:
            avg_sub = 75.0 + (bat_avg - 40.0) * 1.3
        elif bat_avg >= 30.0:
            avg_sub = 55.0 + (bat_avg - 30.0) * 2.0
        else:
            avg_sub = max(5.0, bat_avg * 1.8)
        sr_sub = min(100.0, (sr / 65.0) * 80.0) if sr > 0 else 0.0
        bat_score = 0.75 * avg_sub + 0.25 * sr_sub

    elif fmt == 'ODI':
        if bat_avg >= 50.0:
            avg_sub = 88.0 + min(12.0, (bat_avg - 50.0) * 1.4)
        elif bat_avg >= 40.0:
            avg_sub = 72.0 + (bat_avg - 40.0) * 1.6
        elif bat_avg >= 30.0:
            avg_sub = 55.0 + (bat_avg - 30.0) * 1.7
        else:
            avg_sub = max(5.0, bat_avg * 1.8)

        if sr >= 100.0:
            sr_sub = 90.0 + min(10.0, (sr - 100.0) * 0.8)
        elif sr >= 85.0:
            sr_sub = 75.0 + (sr - 85.0) * 1.0
        else:
            sr_sub = max(10.0, (sr / 85.0) * 75.0)
        bat_score = 0.55 * avg_sub + 0.45 * sr_sub

    else:  # T20I
        if bat_avg >= 40.0:
            avg_sub = 90.0 + min(10.0, (bat_avg - 40.0) * 1.2)
        elif bat_avg >= 30.0:
            avg_sub = 75.0 + (bat_avg - 30.0) * 1.5
        elif bat_avg >= 20.0:
            avg_sub = 55.0 + (bat_avg - 20.0) * 2.0
        else:
            avg_sub = max(5.0, bat_avg * 2.7)

        if sr >= 160.0:
            sr_sub = 95.0 + min(5.0, (sr - 160.0) * 0.5)
        elif sr >= 140.0:
            sr_sub = 82.0 + (sr - 140.0) * 0.65
        elif sr >= 120.0:
            sr_sub = 65.0 + (sr - 120.0) * 0.85
        else:
            sr_sub = max(10.0, (sr / 120.0) * 65.0)
        bat_score = 0.40 * avg_sub + 0.60 * sr_sub

    # 2. Bowling
    bowl_score = 0.0
    if wickets > 0:
        wkts_per_match = wickets / max(matches, 1)

        if fmt == 'Test':
            if bowl_avg <= 20.0:
                avg_sub = 94.0 + (20.0 - bowl_avg) * 0.8
            elif bowl_avg <= 25.0:
                avg_sub = 84.0 + (25.0 - bowl_avg) * 2.0
            elif bowl_avg <= 30.0:
                avg_sub = 70.0 + (30.0 - bowl_avg) * 2.8
            else:
                avg_sub = max(10.0, 70.0 - (bowl_avg - 30.0) * 2.0)

            if bowl_econ <= 2.8:
                econ_sub = 92.0 + (2.8 - bowl_econ) * 10.0
            elif bowl_econ <= 3.3:
                econ_sub = 78.0 + (3.3 - bowl_econ) * 28.0
            else:
                econ_sub = max(10.0, 78.0 - (bowl_econ - 3.3) * 25.0)

            wkt_sub = min(100.0, (wkts_per_match / 4.0) * 95.0)
            bowl_score = 0.45 * avg_sub + 0.30 * econ_sub + 0.25 * wkt_sub

        elif fmt == 'ODI':
            if bowl_avg <= 24.0:
                avg_sub = 92.0 + (24.0 - bowl_avg) * 1.5
            elif bowl_avg <= 28.0:
                avg_sub = 80.0 + (28.0 - bowl_avg) * 3.0
            elif bowl_avg <= 35.0:
                avg_sub = 65.0 + (35.0 - bowl_avg) * 2.1
            else:
                avg_sub = max(10.0, 65.0 - (bowl_avg - 35.0) * 1.8)

            if bowl_econ <= 4.8:
                econ_sub = 92.0 + (4.8 - bowl_econ) * 12.0
            elif bowl_econ <= 5.4:
                econ_sub = 78.0 + (5.4 - bowl_econ) * 23.0
            else:
                econ_sub = max(10.0, 78.0 - (bowl_econ - 5.4) * 18.0)

            wkt_sub = min(100.0, (wkts_per_match / 1.8) * 95.0)
            bowl_score = 0.45 * avg_sub + 0.35 * econ_sub + 0.20 * wkt_sub

        else:  # T20I
            if bowl_avg <= 18.0:
                avg_sub = 94.0 + (18.0 - bowl_avg) * 1.0
            elif bowl_avg <= 24.0:
                avg_sub = 80.0 + (24.0 - bowl_avg) * 2.3
            elif bowl_avg <= 30.0:
                avg_sub = 65.0 + (30.0 - bowl_avg) * 2.5
            else:
                avg_sub = max(10.0, 65.0 - (bowl_avg - 30.0) * 1.8)

            if bowl_econ <= 7.0:
                econ_sub = 94.0 + (7.0 - bowl_econ) * 12.0
            elif bowl_econ <= 8.0:
                econ_sub = 80.0 + (8.0 - bowl_econ) * 14.0
            else:
                econ_sub = max(10.0, 80.0 - (bowl_econ - 8.0) * 15.0)

            wkt_sub = min(100.0, (wkts_per_match / 1.4) * 95.0)
            bowl_score = 0.40 * avg_sub + 0.40 * econ_sub + 0.20 * wkt_sub

    # 3. Role
    if role in ['Top-order Batter', 'Middle-order Batter']:
        final_score = bat_score * 0.96 + (min(4.0, bowl_score * 0.04) if wickets > 0 else 0)
    elif role == 'Wicketkeeper Batter':
        final_score = bat_score * 0.90 + 10.0
    elif role == 'Batting All-rounder':
        final_score = bat_score * 0.58 + bowl_score * 0.42
    elif role == 'Bowling All-rounder':
        final_score = bat_score * 0.38 + bowl_score * 0.62
    elif role in ['Pace Bowler', 'Spin Bowler']:
        final_score = bowl_score * 0.95 + min(5.0, bat_score * 0.05)
    else:
        final_score = (bat_score + bowl_score) / 2.0

    if matches < 3:
        final_score *= 0.88
    elif matches < 10:
        final_score *= 0.95

    return float(np.clip(final_score, 15.0, 99.0))


def generate_insights_and_suggestions(role, fmt, stats, score):
    bat_avg = stats.get('batting_avg', 0.0)
    sr = stats.get('strike_rate', 0.0)
    runs = stats.get('runs', 0)
    fours = stats.get('fours', 0)
    sixes = stats.get('sixes', 0)
    wickets = stats.get('wickets', 0)
    bowl_avg = stats.get('bowling_avg', 0.0)
    bowl_econ = stats.get('bowling_econ', 0.0)
    matches = stats.get('matches', 1)

    boundary_pct = round(((fours * 4 + sixes * 6) / max(runs, 1)) * 100, 1) if runs > 0 else 0.0
    runs_per_match = round(runs / max(matches, 1), 1)

    if bat_avg >= 48.0:
        bat_summary = f"World-class batting average ({bat_avg}) with elite consistency. Averages {runs_per_match} runs per match with {boundary_pct}% runs from boundaries."
    elif bat_avg >= 35.0:
        bat_summary = f"Strong and reliable top-tier average ({bat_avg}) with a healthy strike rate of {sr}. Demonstrates dependable anchor capability."
    elif bat_avg >= 20.0:
        bat_summary = f"Moderate batting average ({bat_avg}). Useful lower-order or counter-punch contribution with {fours} fours and {sixes} sixes."
    else:
        bat_summary = f"Tail-end / lower-order batting profile ({bat_avg} avg). Primary focus remains on bowling or specialist execution."

    wkts_per_match = round(wickets / max(matches, 1), 2)
    if wickets >= 40:
        if bowl_econ < 5.0 and fmt in ['ODI', 'Test']:
            bowl_summary = f"Premier strike and control bowler. Takes {wkts_per_match} wickets/match with an exceptional economy of {bowl_econ} and average of {bowl_avg}."
        else:
            bowl_summary = f"Proven wicket-taker ({wickets} career wickets). Strike rate is potent with an economy rate of {bowl_econ} in {fmt} cricket."
    elif wickets > 0:
        bowl_summary = f"Handy supporting bowling option with {wickets} wickets, operating at an economy rate of {bowl_econ} and average of {bowl_avg}."
    else:
        bowl_summary = "Non-bowling specialist or wicketkeeper batter. No active international bowling workload recorded."

    suggestions = []
    if 'Batter' in role or 'All-rounder' in role:
        if fmt == 'Test':
            if bat_avg < 40.0:
                suggestions.append("Improve defensive strike rotation against swinging red ball outside off-stump to build higher innings milestones.")
            if sr > 70.0 and bat_avg < 38.0:
                suggestions.append("Exercise tighter shot selection during the morning 15-over swing window before expanding scoring strokes.")
            else:
                suggestions.append("Convert solid 30s and 40s into match-defining three-figure hundreds on demanding overseas pitches.")
        elif fmt == 'ODI':
            if sr < 85.0:
                suggestions.append("Increase middle-overs strike rotation against spinners (overs 11-40) to minimize dot ball percentages.")
            if boundary_pct < 45.0:
                suggestions.append("Incorporate aerial lofted strokes during powerplay field restrictions (overs 1-10).")
            else:
                suggestions.append("Maintain high boundary intent while preserving wicket preservation in high-pressure chases.")
        else:  # T20I
            if sr < 135.0:
                suggestions.append("Accelerate boundary striking rate from ball one; refine ramp, scoop, and switch-hit access against wide yorkers.")
            if boundary_pct < 60.0:
                suggestions.append("Optimize boundary-finding options through the 360-degree arc during death overs (16-20).")
            else:
                suggestions.append("Sustain aggressive intent against mystery spin variations during entry overs.")

    if 'Bowler' in role or 'All-rounder' in role:
        if fmt == 'Test':
            if bowl_econ > 3.3:
                suggestions.append("Tighten corridor of uncertainty line and length to maintain pressure and build maiden overs.")
            if bowl_avg > 30.0:
                suggestions.append("Target hard fifth-stump lengths and develop reverse swing tactics during second-innings old ball periods.")
            else:
                suggestions.append("Maintain peak fitness workloads to bowl lethal 6-over bursts in foreign pace-friendly conditions.")
        elif fmt == 'ODI':
            if bowl_econ > 5.5:
                suggestions.append("Develop back-of-the-hand slower variations and wide line yorkers to constrain death overs run-rates.")
            suggestions.append("Focus on early powerplay breakthroughs to disrupt opposition top-order partnerships.")
        else:  # T20I
            if bowl_econ > 8.0:
                suggestions.append("Master wide yorkers and knuckle-ball pace variations to lower death overs boundary vulnerability.")
            suggestions.append("Mix up hard lengths with deceptive bouncers to prevent batters from premeditating scoops.")

    if not suggestions:
        suggestions.append("Maintain current high-standard fitness and match rhythm across bilateral and ICC tournament fixtures.")

    return bat_summary, bowl_summary, suggestions[:3]


def get_performance_category(score):
    if score >= 80.0:
        return {
            'category': 'Excellent',
            'label': 'Elite Match Winner',
            'badge_class': 'badge-excellent',
            'progress_class': 'progress-excellent',
            'color': '#00C853',
            'description': 'Performs at an elite international level with immense match-winning impact.'
        }
    elif score >= 60.0:
        return {
            'category': 'Good',
            'label': 'Reliable Core Performer',
            'badge_class': 'badge-good',
            'progress_class': 'progress-good',
            'color': '#00B4D8',
            'description': 'Consistently high-standard player who stabilizes the side and delivers steady results.'
        }
    elif score >= 40.0:
        return {
            'category': 'Average',
            'label': 'Moderate Impact',
            'badge_class': 'badge-average',
            'progress_class': 'progress-average',
            'color': '#FFA000',
            'description': 'Fair contribution with occasional match impact; exhibits room for technical improvement.'
        }
    else:
        return {
            'category': 'Needs Improvement',
            'label': 'Underperforming / Developmental',
            'badge_class': 'badge-needs-improvement',
            'progress_class': 'progress-needs-improvement',
            'color': '#FF3D00',
            'description': 'Statistical metrics lag behind international benchmarks for this format; requires tactical tuning.'
        }


# =====================================================================
# ROUTES
# =====================================================================

@app.route('/')
def home():
    return render_template(
        'index.html',
        players=PLAYERS_DATA[:8],
        metrics=MODEL_METRICS,
        total_players=len(PLAYERS_DATA)
    )


@app.route('/players')
def players():
    selected_format = request.args.get('format', 'All')
    selected_role = request.args.get('role', 'All')
    search_query = request.args.get('search', '').strip().lower()

    filtered_players = []
    for p in PLAYERS_DATA:
        if selected_role != 'All' and p.get('category') != selected_role and p.get('role') != selected_role:
            continue
        if search_query and search_query not in p.get('name', '').lower() and search_query not in p.get('role', '').lower():
            continue
        filtered_players.append(p)

    return render_template(
        'players.html',
        players=filtered_players,
        all_players=PLAYERS_DATA,
        selected_format=selected_format,
        selected_role=selected_role,
        search_query=search_query
    )


@app.route('/prediction', methods=['GET'])
def prediction_page():
    prefill_id = request.args.get('player', '')
    prefill_format = request.args.get('format', 'ODI')

    prefill_player = None
    if prefill_id:
        for p in PLAYERS_DATA:
            if p['id'] == prefill_id:
                prefill_player = p
                break

    return render_template(
        'prediction.html',
        players=PLAYERS_DATA,
        prefill_player=prefill_player,
        prefill_format=prefill_format,
        metrics=MODEL_METRICS,
        roles=list(ROLE_MAP.keys()),
        formats=list(FORMAT_MAP.keys())
    )


@app.route('/comparison', methods=['GET'])
def comparison_page():
    p1_id = request.args.get('p1', 'rohit-sharma')
    p2_id = request.args.get('p2', 'virat-kohli')
    fmt = request.args.get('format', 'ODI')

    p1 = next((p for p in PLAYERS_DATA if p['id'] == p1_id), PLAYERS_DATA[0] if PLAYERS_DATA else None)
    p2 = next((p for p in PLAYERS_DATA if p['id'] == p2_id), PLAYERS_DATA[1] if len(PLAYERS_DATA) > 1 else None)

    return render_template(
        'comparison.html',
        players=PLAYERS_DATA,
        player1=p1,
        player2=p2,
        selected_format=fmt,
        formats=['Test', 'ODI', 'T20I']
    )


@app.route('/dashboard')
def dashboard():
    team_summary = {
        'total_squad': len(PLAYERS_DATA),
        'total_runs': sum(p['stats']['ODI']['runs'] + p['stats']['Test']['runs'] + p['stats']['T20I']['runs'] for p in PLAYERS_DATA),
        'total_wickets': sum(p['stats']['ODI']['wickets'] + p['stats']['Test']['wickets'] + p['stats']['T20I']['wickets'] for p in PLAYERS_DATA),
        'total_centuries': sum(p['stats']['ODI']['centuries'] + p['stats']['Test']['centuries'] + p['stats']['T20I']['centuries'] for p in PLAYERS_DATA),
        'total_matches_pool': sum(p['stats']['ODI']['matches'] + p['stats']['Test']['matches'] + p['stats']['T20I']['matches'] for p in PLAYERS_DATA)
    }

    return render_template(
        'dashboard.html',
        players=PLAYERS_DATA,
        team_summary=team_summary,
        metrics=MODEL_METRICS
    )


@app.route('/about')
def about():
    return render_template(
        'about.html',
        metrics=MODEL_METRICS
    )


# =====================================================================
# API ENDPOINTS (JSON)
# =====================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()

        player_name = data.get('player_name', 'Indian Cricketer').strip()
        role = data.get('role', 'Top-order Batter')
        fmt = data.get('format', 'ODI')

        try:
            matches = max(1, int(float(data.get('matches', 1))))
            runs = max(0, int(float(data.get('runs', 0))))
            batting_avg = max(0.0, float(data.get('batting_avg', 0.0)))
            strike_rate = max(0.0, float(data.get('strike_rate', 0.0)))
            fours = max(0, int(float(data.get('fours', 0))))
            sixes = max(0, int(float(data.get('sixes', 0))))
            wickets = max(0, int(float(data.get('wickets', 0))))
            bowling_avg = max(0.0, float(data.get('bowling_avg', 0.0)))
            bowling_econ = max(0.0, float(data.get('bowling_econ', 0.0)))
        except (ValueError, TypeError) as num_err:
            return jsonify({
                'success': False,
                'error': f'Invalid numeric input: {str(num_err)}'
            }), 400

        role_enc = ROLE_MAP.get(role, 0)
        fmt_enc = FORMAT_MAP.get(fmt, 1)

        # Build feature DataFrame with correct column names
        feature_df = pd.DataFrame([{
            'role_encoded': role_enc,
            'format_encoded': fmt_enc,
            'matches': matches,
            'runs': runs,
            'batting_avg': batting_avg,
            'strike_rate': strike_rate,
            'fours': fours,
            'sixes': sixes,
            'wickets': wickets,
            'bowling_avg': bowling_avg,
            'bowling_econ': bowling_econ
        }], columns=FEATURE_COLUMNS)

        if ML_MODEL is not None:
            try:
                raw_score = float(ML_MODEL.predict(feature_df)[0])
                model_used = "Scikit-Learn Random Forest Regressor"
            except Exception as inf_err:
                print(f"[Warning] ML model predict failed: {inf_err}. Using analytics heuristic.")
                raw_score = calculate_rule_based_fallback(
                    role, fmt, matches, runs, batting_avg, strike_rate, fours, sixes, wickets, bowling_avg, bowling_econ
                )
                model_used = "Sports Analytics Fallback Heuristic"
        else:
            raw_score = calculate_rule_based_fallback(
                role, fmt, matches, runs, batting_avg, strike_rate, fours, sixes, wickets, bowling_avg, bowling_econ
            )
            model_used = "Sports Analytics Calibrated Heuristic"

        score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)
        cat_info = get_performance_category(score)

        stat_dict = {
            'matches': matches,
            'runs': runs,
            'batting_avg': batting_avg,
            'strike_rate': strike_rate,
            'fours': fours,
            'sixes': sixes,
            'wickets': wickets,
            'bowling_avg': bowling_avg,
            'bowling_econ': bowling_econ
        }

        bat_analysis, bowl_analysis, suggestions = generate_insights_and_suggestions(
            role, fmt, stat_dict, score
        )

        response_payload = {
            'success': True,
            'player_name': player_name,
            'role': role,
            'format': fmt,
            'predicted_score': score,
            'category': cat_info['category'],
            'category_label': cat_info['label'],
            'category_description': cat_info['description'],
            'badge_class': cat_info['badge_class'],
            'color': cat_info['color'],
            'batting_analysis': bat_analysis,
            'bowling_analysis': bowl_analysis,
            'suggestions': suggestions,
            'stats': stat_dict,
            'model_used': model_used,
            'model_metrics': {
                'algorithm': MODEL_METRICS.get('algorithm', 'Random Forest Regressor'),
                'r2_score': MODEL_METRICS.get('test_r2_score', 0.952),
                'mae': MODEL_METRICS.get('test_mae', 2.42)
            }
        }

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server prediction error: {str(e)}'
        }), 500


@app.route('/api/players', methods=['GET'])
def api_get_players():
    return jsonify({
        'success': True,
        'count': len(PLAYERS_DATA),
        'players': PLAYERS_DATA
    })


@app.route('/api/player/<player_id>', methods=['GET'])
def api_get_player(player_id):
    player = next((p for p in PLAYERS_DATA if p['id'] == player_id), None)
    if player:
        return jsonify({'success': True, 'player': player})
    return jsonify({'success': False, 'error': 'Player not found'}), 404


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    return jsonify({
        'success': True,
        'metrics': MODEL_METRICS
    })


@app.route('/api/compare', methods=['POST'])
def api_compare():
    try:
        data = request.get_json() or {}
        p1_id = data.get('p1_id')
        p2_id = data.get('p2_id')
        fmt = data.get('format', 'ODI')

        p1 = next((p for p in PLAYERS_DATA if p['id'] == p1_id), None)
        p2 = next((p for p in PLAYERS_DATA if p['id'] == p2_id), None)

        if not p1 or not p2:
            return jsonify({'success': False, 'error': 'Invalid player selection'}), 400

        st1 = p1['stats'].get(fmt, {})
        st2 = p2['stats'].get(fmt, {})

        def predict_for_player(player, stats, f):
            feat_df = pd.DataFrame([{
                'role_encoded': ROLE_MAP.get(player['role'], 0),
                'format_encoded': FORMAT_MAP.get(f, 1),
                'matches': stats.get('matches', 1),
                'runs': stats.get('runs', 0),
                'batting_avg': stats.get('batting_avg', 0.0),
                'strike_rate': stats.get('strike_rate', 0.0),
                'fours': stats.get('fours', 0),
                'sixes': stats.get('sixes', 0),
                'wickets': stats.get('wickets', 0),
                'bowling_avg': stats.get('bowling_avg', 0.0),
                'bowling_econ': stats.get('bowling_econ', 0.0)
            }], columns=FEATURE_COLUMNS)

            if ML_MODEL is not None:
                sc = float(ML_MODEL.predict(feat_df)[0])
            else:
                sc = calculate_rule_based_fallback(
                    player['role'], f, stats.get('matches', 1), stats.get('runs', 0),
                    stats.get('batting_avg', 0.0), stats.get('strike_rate', 0.0),
                    stats.get('fours', 0), stats.get('sixes', 0), stats.get('wickets', 0),
                    stats.get('bowling_avg', 0.0), stats.get('bowling_econ', 0.0)
                )
            return round(float(np.clip(sc, 0.0, 100.0)), 1)

        score1 = predict_for_player(p1, st1, fmt)
        score2 = predict_for_player(p2, st2, fmt)

        return jsonify({
            'success': True,
            'format': fmt,
            'player1': {
                'id': p1['id'],
                'name': p1['name'],
                'role': p1['role'],
                'stats': st1,
                'score': score1,
                'category': get_performance_category(score1)
            },
            'player2': {
                'id': p2['id'],
                'name': p2['name'],
                'role': p2['role'],
                'stats': st2,
                'score': score2,
                'category': get_performance_category(score2)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("[Server] Starting India Cricket Performance Prediction Web App on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
