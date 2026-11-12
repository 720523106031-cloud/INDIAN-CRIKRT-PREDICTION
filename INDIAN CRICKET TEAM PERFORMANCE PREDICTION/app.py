"""
Root Entry Point for India Cricket Performance Prediction Application
======================================================================
Convenient runner for local development and demonstration:
    python app.py
Or:
    python api/index.py
"""

from api.index import app

if __name__ == '__main__':
    print("=" * 65)
    print("  INDIA CRICKET PERFORMANCE PREDICTION SYSTEM")
    print("  ECE College DevOps Assignment Application")
    print("  Starting local server: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=True)

