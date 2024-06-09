from flask import Flask, render_template, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_cpr_helpcare', methods=['POST'])
def run_cpr_helpcare():
    try:
        # Run the CPR-helpcare script
        subprocess.run(['python', 'cpr_helpcare.py'], check=True)
        
        # Read the results from the JSON file
        with open('results.json', 'r') as f:
            results = json.load(f)
        
        return jsonify({"status": "success", "results": results})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
