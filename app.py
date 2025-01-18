import os
import time
import numpy as np
from PIL import Image
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory, flash, jsonify, Response
from werkzeug.utils import secure_filename
from tqdm import tqdm
import logging
import uuid
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import threading
import subprocess  # <-- For re-encoding with ffmpeg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure secret key

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
PREVIEW_FOLDER = os.path.join('static', 'previews')

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, PREVIEW_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB limit

progress = {}  # In-memory progress tracking

def allowed_file(filename, filetype='image'):
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if filetype == 'image':
            return ext in ALLOWED_IMAGE_EXTENSIONS
        elif filetype == 'video':
            return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

# ---------------------------------------
# RE-ENCODE WITH FFMPEG FOR BROWSER
# ---------------------------------------
def convert_for_browser(input_path):
    """
    Re-encode the given MP4 video (likely MP4V) to H.264 with 'faststart'
    so that it can be streamed in HTML5 <video>.
    Returns the path to the newly generated H.264 file.
    """
    # We'll produce a temporary file named something_h264.mp4
    output_for_browser = input_path.rsplit('.', 1)[0] + '_h264.mp4'
    command = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-movflags', '+faststart',
        output_for_browser
    ]
    logging.info(f"Re-encoding {input_path} to {output_for_browser} via ffmpeg...")
    subprocess.run(command, check=True)
    return output_for_browser


# ----------------- EFFECT FUNCTIONS ----------------- #

def pixelate(image, pixel_size):
    height, width, channels = image.shape
    temp = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated_image = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image

def apply_glitch(image, glitch_count=10, glitch_shift=20):
    height, width, channels = image.shape
    actual_glitches = np.random.randint(max(1, glitch_count-2), glitch_count+2)
    for _ in range(actual_glitches):
        y_start = np.random.randint(0, height)
        y_end = y_start + np.random.randint(1, height // 10)
        x_start = np.random.randint(0, width)
        x_end = x_start + np.random.randint(1, width // 10)

        if y_end > height:
            y_end = height
        if x_end > width:
            x_end = width

        shift = np.random.randint(-glitch_shift, glitch_shift)
        image[y_start:y_end, x_start:x_end] = np.roll(
            image[y_start:y_end, x_start:x_end],
            shift,
            axis=1
        )
    return image

def apply_distortion(image, distortion_x=20, distortion_y=20):
    height, width, channels = image.shape
    pts1 = np.float32([[0, 0], [width, 0], [0, height]])

    dx = np.random.randint(-distortion_x, distortion_x)
    dy = np.random.randint(-distortion_y, distortion_y)
    pts2 = np.float32([
        [0+dx, 0+dy],
        [width+dx, 0+dy],
        [0+dx, height+dy]
    ])

    matrix = cv2.getAffineTransform(pts1, pts2)
    distorted_image = cv2.warpAffine(image, matrix, (width, height))
    return distorted_image

def apply_color_bleeding(image, color_bleed_shift=2):
    height, width, channels = image.shape
    # shift red channel
    red = image[:, :, 2]
    shift_r = np.random.randint(-color_bleed_shift, color_bleed_shift+1)
    red_shifted = np.roll(red, shift_r, axis=1)
    image[:, :, 2] = red_shifted

    # shift blue channel
    blue = image[:, :, 0]
    shift_b = np.random.randint(-color_bleed_shift, color_bleed_shift+1)
    blue_shifted = np.roll(blue, shift_b, axis=1)
    image[:, :, 0] = blue_shifted

    return image

def apply_scan_lines(image, scan_line_gap=4, scan_line_darkness=0.7):
    height, width, channels = image.shape
    overlay = image.copy()
    for y in range(0, height, scan_line_gap):
        overlay[y:y+1, :] = (overlay[y:y+1, :] * scan_line_darkness).astype('uint8')
    blended = cv2.addWeighted(image, 0.9, overlay, 0.1, 0)
    return blended

def add_static_noise(image, intensity=0.02):
    try:
        noise = np.random.randint(0, 256, image.shape, dtype='uint8')
        mask = np.random.rand(*image.shape[:2], image.shape[2]) < intensity
        image[mask] = noise[mask]
        return image
    except Exception as e:
        logging.error(f"Error adding static noise: {e}")
        return image

def apply_flicker(image, flicker_min=0.95, flicker_max=1.05):
    factor = np.random.uniform(flicker_min, flicker_max)
    image = np.clip(image * factor, 0, 255).astype('uint8')
    return image

# --------------- NOISE APPLIERS (IMAGE & VIDEO) --------------- #

def add_noise_image(
    image_path, output_path, amount, strength, monochromatic, pixel_size=1,
    glitch=False, distortion=False, color_bleed=False, scan_lines=False,
    static=False, flicker=False,
    glitch_count=10, glitch_shift=20,
    distortion_x=20, distortion_y=20,
    color_bleed_shift=2,
    scan_line_gap=4, scan_line_darkness=0.7,
    static_intensity=0.02,
    flicker_min=0.95, flicker_max=1.05
):
    try:
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        np_image = np.array(image)

        total_pixels = np_image.shape[0] * np_image.shape[1]
        num_noisy_pixels = int((amount / 100.0) * total_pixels)
        indices = np.random.choice(total_pixels, num_noisy_pixels, replace=False)
        y_coords, x_coords = np.unravel_index(indices, np_image.shape[:2])

        if monochromatic:
            noise = np.random.randint(-strength, strength + 1, num_noisy_pixels).reshape(-1, 1)
            original = np_image[y_coords, x_coords].mean(axis=1, keepdims=True)
            new_values = np.clip(original + noise, 0, 255).astype('uint8')
            np_image[y_coords, x_coords] = np.repeat(new_values, 3, axis=1)
        else:
            noise = np.random.randint(-strength, strength + 1, (num_noisy_pixels, 3))
            original = np_image[y_coords, x_coords].astype(int)
            new_pixels = np.clip(original + noise, 0, 255).astype('uint8')
            np_image[y_coords, x_coords] = new_pixels

        if pixel_size > 1:
            np_image = pixelate(np_image, pixel_size)

        if glitch:
            np_image = apply_glitch(np_image, glitch_count=glitch_count, glitch_shift=glitch_shift)

        if distortion:
            np_image = apply_distortion(np_image, distortion_x=distortion_x, distortion_y=distortion_y)

        if color_bleed:
            np_image = apply_color_bleeding(np_image, color_bleed_shift=color_bleed_shift)

        if scan_lines:
            np_image = apply_scan_lines(np_image, scan_line_gap=scan_line_gap, scan_line_darkness=scan_line_darkness)

        if static:
            np_image = add_static_noise(np_image, intensity=static_intensity)

        if flicker:
            np_image = apply_flicker(np_image, flicker_min=flicker_min, flicker_max=flicker_max)

        noisy_image = Image.fromarray(np_image)
        noisy_image.save(output_path)
        logging.info(f"Noisy image saved to {output_path}")
        return f"Noisy image saved to {output_path}"
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return f"Error processing image: {e}"

def add_noise_to_frame(
    frame, amount, strength, monochromatic, pixel_size=1,
    glitch=False, distortion=False, color_bleed=False, scan_lines=False,
    static=False, flicker=False,
    glitch_count=10, glitch_shift=20,
    distortion_x=20, distortion_y=20,
    color_bleed_shift=2,
    scan_line_gap=4, scan_line_darkness=0.7,
    static_intensity=0.02,
    flicker_min=0.95, flicker_max=1.05
):
    noisy_frame = frame.copy()

    total_pixels = frame.shape[0] * frame.shape[1]
    num_noisy_pixels = int((amount / 100.0) * total_pixels)
    indices = np.random.choice(total_pixels, num_noisy_pixels, replace=False)
    y_coords, x_coords = np.unravel_index(indices, noisy_frame.shape[:2])

    if monochromatic:
        noise = np.random.randint(-strength, strength + 1, num_noisy_pixels).reshape(-1, 1)
        original = noisy_frame[y_coords, x_coords].mean(axis=1, keepdims=True)
        new_values = np.clip(original + noise, 0, 255).astype('uint8')
        noisy_frame[y_coords, x_coords] = np.repeat(new_values, 3, axis=1)
    else:
        noise = np.random.randint(-strength, strength + 1, (num_noisy_pixels, 3))
        original = noisy_frame[y_coords, x_coords].astype(int)
        new_pixels = np.clip(original + noise, 0, 255).astype('uint8')
        noisy_frame[y_coords, x_coords] = new_pixels

    if pixel_size > 1:
        noisy_frame = pixelate(noisy_frame, pixel_size)

    if glitch:
        noisy_frame = apply_glitch(noisy_frame, glitch_count=glitch_count, glitch_shift=glitch_shift)

    if distortion:
        noisy_frame = apply_distortion(noisy_frame, distortion_x=distortion_x, distortion_y=distortion_y)

    if color_bleed:
        noisy_frame = apply_color_bleeding(noisy_frame, color_bleed_shift=color_bleed_shift)

    if scan_lines:
        noisy_frame = apply_scan_lines(noisy_frame, scan_line_gap=scan_line_gap, scan_line_darkness=scan_line_darkness)

    if static:
        noisy_frame = add_static_noise(noisy_frame, intensity=static_intensity)

    if flicker:
        noisy_frame = apply_flicker(noisy_frame, flicker_min=flicker_min, flicker_max=flicker_max)

    return noisy_frame

def add_noise_video(
    input_path, output_path, amount, strength, monochromatic, pixel_size=1,
    glitch=False, distortion=False, color_bleed=False, scan_lines=False,
    static=False, flicker=False,
    glitch_count=10, glitch_shift=20,
    distortion_x=20, distortion_y=20,
    color_bleed_shift=2,
    scan_line_gap=4, scan_line_darkness=0.7,
    static_intensity=0.02,
    flicker_min=0.95, flicker_max=1.05
):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error("Cannot open video file.")
            return "Error: Cannot open video file."

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # By default, this is 'mp4v' -> Many browsers won't decode it.
        # We'll re-encode to H.264 afterwards.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Processing video: {input_path}")
        logging.info(f"Total frames: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

        for _ in tqdm(range(frame_count), desc="Processing frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            noisy_frame = add_noise_to_frame(
                frame, amount, strength, monochromatic, pixel_size,
                glitch, distortion, color_bleed, scan_lines, static, flicker,
                glitch_count=glitch_count, glitch_shift=glitch_shift,
                distortion_x=distortion_x, distortion_y=distortion_y,
                color_bleed_shift=color_bleed_shift,
                scan_line_gap=scan_line_gap, scan_line_darkness=scan_line_darkness,
                static_intensity=static_intensity,
                flicker_min=flicker_min, flicker_max=flicker_max
            )
            out.write(noisy_frame)

        cap.release()
        out.release()
        logging.info(f"Noisy video saved to {output_path}")
        return f"Noisy video saved to {output_path}"
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return f"Error processing video: {e}"

def process_video_preview(
    input_path, preview_path, amount, strength, monochromatic, pixel_size=1,
    glitch=False, distortion=False, color_bleed=False, scan_lines=False,
    static=False, flicker=False,
    glitch_count=10, glitch_shift=20,
    distortion_x=20, distortion_y=20,
    color_bleed_shift=2,
    scan_line_gap=4, scan_line_darkness=0.7,
    static_intensity=0.02,
    flicker_min=0.95, flicker_max=1.05
):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error("Cannot open video file for preview.")
            return "Error: Cannot open video file for preview."

        ret, frame = cap.read()
        if not ret:
            logging.error("Cannot read frame from video for preview.")
            cap.release()
            return "Error: Cannot read frame from video for preview."

        noisy_frame = add_noise_to_frame(
            frame, amount, strength, monochromatic, pixel_size,
            glitch, distortion, color_bleed, scan_lines, static, flicker,
            glitch_count=glitch_count, glitch_shift=glitch_shift,
            distortion_x=distortion_x, distortion_y=distortion_y,
            color_bleed_shift=color_bleed_shift,
            scan_line_gap=scan_line_gap, scan_line_darkness=scan_line_darkness,
            static_intensity=static_intensity,
            flicker_min=flicker_min, flicker_max=flicker_max
        )
        noisy_frame_rgb = cv2.cvtColor(noisy_frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        noisy_image = Image.fromarray(noisy_frame_rgb)
        noisy_image.save(preview_path)
        logging.info(f"Video preview saved to {preview_path}")
        cap.release()
        return f"Video preview saved to {preview_path}"
    except Exception as e:
        logging.error(f"Error creating video preview: {e}")
        return f"Error creating video preview: {e}"

# ----------------- ASYNC PROCESSING ----------------- #

def process_video_async(
    task_id, input_path, output_path, amount, strength, monochromatic, pixel_size,
    glitch, distortion, color_bleed, scan_lines, static, flicker,
    glitch_count, glitch_shift,
    distortion_x, distortion_y,
    color_bleed_shift,
    scan_line_gap, scan_line_darkness,
    static_intensity,
    flicker_min, flicker_max
):
    """
    Called in a thread: run add_noise_video, then re-encode with ffmpeg.
    """
    result = add_noise_video(
        input_path, output_path, amount, strength, monochromatic, pixel_size,
        glitch, distortion, color_bleed, scan_lines, static, flicker,
        glitch_count=glitch_count, glitch_shift=glitch_shift,
        distortion_x=distortion_x, distortion_y=distortion_y,
        color_bleed_shift=color_bleed_shift,
        scan_line_gap=scan_line_gap, scan_line_darkness=scan_line_darkness,
        static_intensity=static_intensity,
        flicker_min=flicker_min, flicker_max=flicker_max
    )

    if result.startswith("Noisy video saved to"):
        # Re-encode the MP4V file to H.264 so the browser can play it
        try:
            final_path = convert_for_browser(output_path)
            # Overwrite original with the H.264 file
            os.remove(output_path)
            os.rename(final_path, output_path)
            logging.info(f"Re-encoded video saved to {output_path}")

            result = f"Noisy video saved and re-encoded for browser to {output_path}"
        except Exception as e:
            logging.error(f"FFmpeg re-encode error: {e}")
            result = f"Noisy video saved, but re-encode failed: {e}"

    filename = os.path.basename(output_path)
    progress[task_id]['status'] = 'completed'
    progress[task_id]['result'] = filename

# ----------------- ROUTES ----------------- #

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload_preview', methods=['POST'])
def upload_preview():
    if 'input_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    file = request.files['input_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file:
        mode = request.form.get('mode')
        amount = int(request.form.get('amount', 10))
        strength = int(request.form.get('strength', 10))
        pixel_size = int(request.form.get('pixel_size', 1))
        monochromatic = 'monochromatic' in request.form

        glitch = 'glitch' in request.form
        distortion = 'distortion' in request.form
        color_bleed = 'color_bleed' in request.form
        scan_lines = 'scan_lines' in request.form
        static = 'static' in request.form
        flicker = 'flicker' in request.form

        glitch_count = int(request.form.get('glitch_count', 10))
        glitch_shift = int(request.form.get('glitch_shift', 20))
        distortion_x = int(request.form.get('distortion_x', 20))
        distortion_y = int(request.form.get('distortion_y', 20))
        color_bleed_shift = int(request.form.get('color_bleed_shift', 2))
        scan_line_gap = int(request.form.get('scan_line_gap', 4))
        scan_line_darkness = float(request.form.get('scan_line_darkness', 0.7))
        static_intensity = float(request.form.get('static_intensity', 0.02))
        flicker_min = float(request.form.get('flicker_min', 0.95))
        flicker_max = float(request.form.get('flicker_max', 1.05))

        # Validate
        if not (0 <= amount <= 100) or not (0 <= strength <= 100):
            return jsonify({
                'status': 'error', 
                'message': 'Amount and Strength must be between 0 and 100.'
            }), 400

        filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex
        input_filename = f"{unique_id}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        logging.info(f"Uploaded file saved to {input_path}")

        file_ext = filename.rsplit('.', 1)[1].lower()
        if mode == 'image' and file_ext not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({'status': 'error', 'message': 'Invalid image file type.'}), 400
        if mode == 'video' and file_ext not in ALLOWED_VIDEO_EXTENSIONS:
            return jsonify({'status': 'error', 'message': 'Invalid video file type.'}), 400

        output_filename = f"noisy_{unique_id}_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        if mode == 'image':
            result = add_noise_image(
                input_path, output_path, amount, strength, monochromatic, pixel_size,
                glitch, distortion, color_bleed, scan_lines, static, flicker,
                glitch_count=glitch_count, glitch_shift=glitch_shift,
                distortion_x=distortion_x, distortion_y=distortion_y,
                color_bleed_shift=color_bleed_shift,
                scan_line_gap=scan_line_gap, scan_line_darkness=scan_line_darkness,
                static_intensity=static_intensity,
                flicker_min=flicker_min, flicker_max=flicker_max
            )
            if result.startswith("Error"):
                return jsonify({'status': 'error', 'message': result}), 500
            else:
                return jsonify({
                    'status': 'success',
                    'preview_url': url_for('download_file', filename=output_filename)
                }), 200
        elif mode == 'video':
            preview_filename = f"preview_{unique_id}_{filename.rsplit('.',1)[0]}.jpg"
            preview_path = os.path.join(app.config['PREVIEW_FOLDER'], preview_filename)
            preview_result = process_video_preview(
                input_path, preview_path, amount, strength, monochromatic, pixel_size,
                glitch, distortion, color_bleed, scan_lines, static, flicker,
                glitch_count=glitch_count, glitch_shift=glitch_shift,
                distortion_x=distortion_x, distortion_y=distortion_y,
                color_bleed_shift=color_bleed_shift,
                scan_line_gap=scan_line_gap, scan_line_darkness=scan_line_darkness,
                static_intensity=static_intensity,
                flicker_min=flicker_min, flicker_max=flicker_max
            )
            if preview_result.startswith("Error"):
                return jsonify({'status': 'error', 'message': preview_result}), 500
            else:
                preview_image_url = url_for('static', filename=f"previews/{preview_filename}")
                return jsonify({
                    'status': 'success',
                    'preview_image': preview_image_url,
                    'input_file': input_filename,
                    'output_file': output_filename,
                    'amount': amount,
                    'strength': strength,
                    'pixel_size': pixel_size,
                    'monochromatic': monochromatic,
                    'glitch': glitch,
                    'distortion': distortion,
                    'color_bleed': color_bleed,
                    'scan_lines': scan_lines,
                    'static': static,
                    'flicker': flicker,
                    'glitch_count': glitch_count,
                    'glitch_shift': glitch_shift,
                    'distortion_x': distortion_x,
                    'distortion_y': distortion_y,
                    'color_bleed_shift': color_bleed_shift,
                    'scan_line_gap': scan_line_gap,
                    'scan_line_darkness': scan_line_darkness,
                    'static_intensity': static_intensity,
                    'flicker_min': flicker_min,
                    'flicker_max': flicker_max
                }), 200
        else:
            return jsonify({'status': 'error', 'message': 'Invalid mode selected.'}), 400


@app.route('/process_video_async', methods=['POST'])
def process_video_async_route():
    data = request.json

    input_file = data.get('input_file')
    output_file = data.get('output_file')
    amount = int(data.get('amount', 10))
    strength = int(data.get('strength', 10))
    pixel_size = int(data.get('pixel_size', 1))
    monochromatic = bool(data.get('monochromatic'))

    glitch = bool(data.get('glitch'))
    distortion = bool(data.get('distortion'))
    color_bleed = bool(data.get('color_bleed'))
    scan_lines = bool(data.get('scan_lines'))
    static = bool(data.get('static'))
    flicker = bool(data.get('flicker'))

    glitch_count = int(data.get('glitch_count', 10))
    glitch_shift = int(data.get('glitch_shift', 20))
    distortion_x = int(data.get('distortion_x', 20))
    distortion_y = int(data.get('distortion_y', 20))
    color_bleed_shift = int(data.get('color_bleed_shift', 2))
    scan_line_gap = int(data.get('scan_line_gap', 4))
    scan_line_darkness = float(data.get('scan_line_darkness', 0.7))
    static_intensity = float(data.get('static_intensity', 0.02))
    flicker_min = float(data.get('flicker_min', 0.95))
    flicker_max = float(data.get('flicker_max', 1.05))

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_file)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)

    task_id = uuid.uuid4().hex
    progress[task_id] = {'status': 'processing', 'result': None}

    thread = threading.Thread(
        target=process_video_async,
        args=(
            task_id, input_path, output_path, amount, strength, monochromatic, pixel_size,
            glitch, distortion, color_bleed, scan_lines, static, flicker,
            glitch_count, glitch_shift, distortion_x, distortion_y,
            color_bleed_shift, scan_line_gap, scan_line_darkness,
            static_intensity, flicker_min, flicker_max
        )
    )
    thread.start()

    return jsonify({'task_id': task_id}), 202

@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    if task_id in progress:
        return jsonify(progress[task_id])
    else:
        return jsonify({'status': 'unknown'}), 404

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# ------------- BROWSER-FRIENDLY VIDEO STREAMING ------------- #
@app.route('/video/<filename>')
def serve_video(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return "File not found", 404

    range_header = request.headers.get('Range', None)
    if not range_header:
        # No 'Range' header; send entire file
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, mimetype='video/mp4', as_attachment=False)

    size = os.stat(file_path).st_size
    byte1, byte2 = 0, None

    if '=' in range_header:
        range_val = range_header.split('=')[1]
        if '-' in range_val:
            parts = range_val.split('-')
            if parts[0].strip():
                byte1 = int(parts[0])
            if len(parts) > 1 and parts[1].strip():
                byte2 = int(parts[1])

    if byte2 is None or byte2 >= size:
        byte2 = size - 1

    length = byte2 - byte1 + 1
    with open(file_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype='video/mp4', content_type='video/mp4')
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv

# ------------- CLEANUP SCHEDULER ------------- #
def cleanup_files():
    now = time.time()
    cutoff = now - (24 * 60 * 60)  # 1 day
    folders = [UPLOAD_FOLDER, OUTPUT_FOLDER, PREVIEW_FOLDER]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                if os.path.getmtime(file_path) < cutoff:
                    try:
                        os.remove(file_path)
                        logging.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error deleting file {file_path}: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_files, 'interval', hours=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    app.run(debug=False)
