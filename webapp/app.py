import os
import shutil
import uuid
import subprocess
import sys
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    # now you're handling non-HTTP exceptions only
    return jsonify(error=str(e)), 500
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    script_type = request.form.get('script_type', 'pixelate2')
    preset = request.form.get('preset', 'clear')

    # Create a unique working directory
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(app.config['PROCESSED_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join(job_dir, 'restored'), exist_ok=True)

    filename = secure_filename(file.filename)
    orig_path = os.path.join(job_dir, filename)
    file.save(orig_path)

    # Copy scripts to job_dir so pure_pix works in isolation
    for script in ['pure_pix.py', 'pixelate2.py', 'de.py', 'check_pixelated.py', 'layer7_verifier.py']:
        src_path = os.path.join(BASE_DIR, script)
        if os.path.exists(src_path):
            shutil.copy(src_path, job_dir)

    pixelated_file = None

    try:
        if script_type == 'pure_pix':
            from PIL import Image
            img = Image.open(orig_path)
            img.convert('RGB').save(os.path.join(job_dir, 'photo.jpg'), 'JPEG')
            subprocess.run([sys.executable, 'pure_pix.py'], cwd=job_dir, check=True)
            pixelated_file = 'pure_pix.jpg'
            subprocess.run([sys.executable, 'de.py', pixelated_file, '--restored-out', 'restored/', '--json', 'report.json'], cwd=job_dir, check=True)
        elif script_type == 'pixelate2':
            # pixelate2.py
            subprocess.run([sys.executable, 'pixelate2.py', filename, '-p', preset], cwd=job_dir, check=True)
            stem, ext = os.path.splitext(filename)
            pixelated_file = f"{stem}_pixelated{ext}"
            subprocess.run([sys.executable, 'de.py', pixelated_file, '--restored-out', 'restored/', '--json', 'report.json'], cwd=job_dir, check=True)
        elif script_type == 'de':
            # run only de.py
            subprocess.run([sys.executable, 'de.py', filename, '--restored-out', 'restored/', '--json', 'report.json'], cwd=job_dir, check=True)
            pixelated_file = filename
        elif script_type == 'check_pixelated':
            # run only check_pixelated.py
            output = subprocess.run([sys.executable, 'check_pixelated.py', filename], cwd=job_dir, capture_output=True, text=True, check=True)
            with open(os.path.join(job_dir, 'report.json'), 'w') as f:
                f.write(output.stdout)
            pixelated_file = filename

        # Collect outputs
        pixelated_path = os.path.join(job_dir, pixelated_file)
        restored_path = None
        
        # Check if de.py actually restored something
        restored_dir = os.path.join(job_dir, 'restored')
        if os.path.exists(restored_dir):
            restored_files = os.listdir(restored_dir)
            if restored_files:
                restored_path = os.path.join('restored', restored_files[0])

        # Prepare ZIP file for easy download
        zip_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{job_id}.zip")
        import zipfile
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            if os.path.exists(pixelated_path):
                zipf.write(pixelated_path, arcname=f"pixelated_{pixelated_file}")
            if restored_path and os.path.exists(os.path.join(job_dir, restored_path)):
                zipf.write(os.path.join(job_dir, restored_path), arcname=f"restored_{os.path.basename(restored_path)}")
            report_path = os.path.join(job_dir, 'report.json')
            if os.path.exists(report_path):
                zipf.write(report_path, arcname="report.json")

        return jsonify({
            'success': True,
            'job_id': job_id,
            'download_url': f"/download/{job_id}",
            'pixelated_url': f"/file/{job_id}/{pixelated_file}",
            'restored_url': f"/file/{job_id}/{restored_path}" if restored_path else None,
            'report_url': f"/file/{job_id}/report.json"
        })
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Script failed: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<job_id>')
def download_zip(job_id):
    zip_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{job_id}.zip")
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True, download_name="processed_images.zip")
    return "Not found", 404

@app.route('/file/<job_id>/<path:filename>')
def serve_file(job_id, filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], job_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return "Not found", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
