"""
India Cricket Performance Prediction System - Comprehensive Test Suite
======================================================================
Tests all Flask routes, REST APIs, ML model inference, and error handling.
"""

import unittest
import json
import os
from api.index import app, ML_MODEL, PLAYERS_DATA, MODEL_METRICS

class TestCricketApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_resources_loaded(self):
        """Verify model, players database, and metrics are loaded."""
        self.assertIsNotNone(ML_MODEL, "ML model should be loaded successfully.")
        self.assertGreater(len(PLAYERS_DATA), 10, "Players database should contain at least 10 Indian cricketers.")
        self.assertIn('algorithm', MODEL_METRICS, "Metrics should contain algorithm specification.")
        self.assertIn('test_r2_score', MODEL_METRICS, "Metrics should contain test_r2_score.")

    def test_02_home_route(self):
        """Verify home page route loads with status 200."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'INDIA CRICKET PERFORMANCE', response.data)

    def test_03_players_route(self):
        """Verify players directory page and filtering."""
        response = self.client.get('/players')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Virat Kohli', response.data)
        self.assertIn(b'Rohit Sharma', response.data)

        # Filter by format
        response_test = self.client.get('/players?format=Test')
        self.assertEqual(response_test.status_code, 200)

    def test_04_prediction_page_route(self):
        """Verify prediction page loads and supports prefilling."""
        response = self.client.get('/prediction?player=virat-kohli&format=ODI')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Player Performance Prediction', response.data)

    def test_05_comparison_page_route(self):
        """Verify comparison page route loads."""
        response = self.client.get('/comparison?p1=rohit-sharma&p2=virat-kohli&format=ODI')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Indian Player Comparison', response.data)

    def test_06_dashboard_page_route(self):
        """Verify dashboard page route loads with aggregated statistics."""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'India National Team Analytics Dashboard', response.data)

    def test_07_about_page_route(self):
        """Verify about and DevOps documentation page loads."""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'DevOps', response.data)

    def test_08_api_players(self):
        """Verify GET /api/players endpoint returns valid JSON."""
        response = self.client.get('/api/players')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['count'], len(PLAYERS_DATA))

    def test_09_api_single_player(self):
        """Verify GET /api/player/<id> endpoint."""
        response = self.client.get('/api/player/virat-kohli')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['player']['name'], 'Virat Kohli')

        # Test non-existent player
        resp_404 = self.client.get('/api/player/non-existent-player')
        self.assertEqual(resp_404.status_code, 404)

    def test_10_api_model_info(self):
        """Verify GET /api/model-info endpoint."""
        response = self.client.get('/api/model-info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['metrics']['algorithm'], 'Random Forest Regressor')

    def test_11_api_predict_batter(self):
        """Verify POST /api/predict for a world-class batter (e.g. Virat Kohli ODI stats)."""
        payload = {
            'player_name': 'Virat Kohli',
            'role': 'Top-order Batter',
            'format': 'ODI',
            'matches': 295,
            'runs': 13906,
            'batting_avg': 58.18,
            'strike_rate': 93.54,
            'fours': 1302,
            'sixes': 151,
            'wickets': 5,
            'bowling_avg': 133.2,
            'bowling_econ': 6.22
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertGreaterEqual(data['predicted_score'], 80.0, "Elite batter score should be >= 80")
        self.assertEqual(data['category'], 'Excellent')
        self.assertTrue(len(data['suggestions']) > 0)
        print(f"\n[Test Result] Virat Kohli ODI Predicted Score: {data['predicted_score']}/100 ({data['category']})")

    def test_12_api_predict_bowler(self):
        """Verify POST /api/predict for a world-class bowler (e.g. Jasprit Bumrah Test stats)."""
        payload = {
            'player_name': 'Jasprit Bumrah',
            'role': 'Pace Bowler',
            'format': 'Test',
            'matches': 45,
            'runs': 310,
            'batting_avg': 7.04,
            'strike_rate': 40.52,
            'fours': 34,
            'sixes': 9,
            'wickets': 205,
            'bowling_avg': 19.49,
            'bowling_econ': 2.74
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertGreaterEqual(data['predicted_score'], 80.0, "Elite bowler score should be >= 80")
        self.assertEqual(data['category'], 'Excellent')
        print(f"[Test Result] Jasprit Bumrah Test Predicted Score: {data['predicted_score']}/100 ({data['category']})")

    def test_13_api_predict_validation_error(self):
        """Verify error handling on invalid inputs."""
        payload = {
            'player_name': 'Invalid Player',
            'matches': 'invalid_string'
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_14_api_compare(self):
        """Verify POST /api/compare endpoint comparing two players."""
        payload = {
            'p1_id': 'rohit-sharma',
            'p2_id': 'virat-kohli',
            'format': 'ODI'
        }
        response = self.client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['player1']['name'], 'Rohit Sharma')
        self.assertEqual(data['player2']['name'], 'Virat Kohli')
        self.assertIn('score', data['player1'])
        self.assertIn('score', data['player2'])
        print(f"[Test Result] Comparison in ODI: Rohit ({data['player1']['score']}) vs Virat ({data['player2']['score']})")


if __name__ == '__main__':
    unittest.main()

