import os
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['CONFIG_FILE'] = 'config.json'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def is_configured():
    return os.path.exists(app.config['CONFIG_FILE'])

def get_config():
    if is_configured():
        with open(app.config['CONFIG_FILE'], 'r') as f:
            return json.load(f)
    return {}

def save_config(config_data):
    with open(app.config['CONFIG_FILE'], 'w') as f:
        json.dump(config_data, f, indent=4)

def get_models_config():
    config = get_config()
    return config.get('models', [])

@app.route('/')
def index():
    if not is_configured():
        return redirect(url_for('setup'))
    
    config = get_config()
    return render_template('inference.html', config=config, models=config.get('models', []))

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if is_configured() and request.method == 'GET':
        return redirect(url_for('index'))

    if request.method == 'POST':
        site_name = request.form.get('site_name')
        description = request.form.get('description')
        classes_str = request.form.get('classes')
        
        # New student profile fields
        nim = request.form.get('nim')
        student_name = request.form.get('student_name')
        prodi = request.form.get('prodi')
        supervisor = request.form.get('supervisor')
        theme_color = request.form.get('theme_color', '#6366f1')
        theme_mode = request.form.get('theme_mode', 'dark')
        input_type = request.form.get('input_type', 'image')
        
        model_names = request.form.getlist('model_names[]')
        model_files = request.files.getlist('model_files[]')
        
        saved_models = []
        
        for name, model_file in zip(model_names, model_files):
            if model_file and model_file.filename != '' and name.strip() != '':
                filename = secure_filename(model_file.filename)
                model_file.save(os.path.join(app.config['MODEL_FOLDER'], filename))
                
                saved_models.append({
                    'display_name': name,
                    'filename': filename
                })
        
        if saved_models:
            # Parse global classes 
            classes_list = [c.strip() for c in classes_str.split(',') if c.strip()] if classes_str else []
            
            config_data = {
                'site_name': site_name,
                'description': description,
                'classes': classes_list,
                'nim': nim,
                'student_name': student_name,
                'prodi': prodi,
                'supervisor': supervisor,
                'theme_color': theme_color,
                'theme_mode': theme_mode,
                'input_type': input_type,
                'is_setup': True,
                'models': saved_models
            }
            save_config(config_data)
            return redirect(url_for('index'))
        else:
            # Handle case where no models were uploaded
            return render_template('setup.html', error="At least one model is required.")
            
    return render_template('setup.html')

@app.route('/reset', methods=['POST'])
def reset():
    # Only allow reset if configured (simple safety)
    if is_configured():
        # Remove config file
        if os.path.exists(app.config['CONFIG_FILE']):
            os.remove(app.config['CONFIG_FILE'])
        
        # Clear uploads
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            if os.path.isfile(path):
                os.remove(path)
        
        # Clear models
        for f in os.listdir(app.config['MODEL_FOLDER']):
            path = os.path.join(app.config['MODEL_FOLDER'], f)
            if os.path.isfile(path):
                os.remove(path)
                
    return redirect(url_for('setup'))

@app.route('/process', methods=['POST'])
def process():
    config = get_config()
    input_type = config.get('input_type', 'image')
    model_filename = request.form.get('model')
    
    # Common variables
    prediction = ""
    confidence = 0
    image_url = None
    width, height = 0, 0

    if input_type == 'image':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            img = Image.open(filepath)
            width, height = img.size
            image_url = f'/uploads/{filename}'
        except Exception as e:
            return jsonify({'error': f'Invalid image: {str(e)}'}), 500
    else:
        # Text input
        text_input = request.form.get('text_input', '').strip()
        if not text_input:
            return jsonify({'error': 'No text provided'}), 400
        # For text, we don't have an image_url or size
    
    # Dummy processing logic using config classes
    try:
        available_classes = config.get('classes', ["Object"])
        prediction = np.random.choice(available_classes)
        confidence = round(np.random.uniform(0.7, 0.99), 2)
        
        response_data = {
            'prediction': prediction,
            'confidence': confidence,
            'metadata': {
                'model_used': model_filename,
                'input_type': input_type
            }
        }
        
        if input_type == 'image':
            response_data['image_url'] = image_url
            response_data['metadata']['width'] = width
            response_data['metadata']['height'] = height
            
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Using port 5001 to avoid AirPlay conflict on macOS (port 5000)
    app.run(debug=True, port=5001, host='0.0.0.0')
