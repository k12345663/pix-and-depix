# Ministry of Truth | Pix & Depix

This project is an advanced image forensics and steganography tool. It is designed to pixelate an image for visual censorship while mathematically multiplexing (hiding) the original, high-resolution details into orthogonal frequency domains. This allows the tool to later perfectly demultiplex and recover the original image, all while bypassing standard AI forensic detection.

## Project Structure

The project is split into two primary components: the **Core Mathematics Engine** and the **Web Interface**.

### 1. Core Mathematics Engine
These Python scripts handle the actual mathematical transformations, steganography, and forensic detection. They rely heavily on `scipy` (for Fast Fourier Transforms and Discrete Cosine Transforms) and `numpy`.

* **`pixelate.py`** *(formerly pixelate2.py)*: The Multiplexer. This script takes a source image and applies a pixelation effect. However, instead of destroying the data, it hides the original high-resolution details within the invisible frequency domains of the image.
* **`depixelate.py`** *(formerly de.py)*: The Demultiplexer & Verifier. This script processes a seemingly pixelated image, scans the frequency domains for hidden data, and extracts it to reconstruct the original image.
* **`check_pixelation.py`** *(formerly check_pixelated.py)*: The Detector. A forensic utility script used to determine if an image has been artificially pixelated or degraded by calculating block sizes, grid strengths, and flat tile ratios.
* **`pure_pix.py`**: The Control Baseline. A simple script that performs traditional, destructive pixelation (resizing down and back up). This is used as a control image to compare against the output of `pixelate.py` to ensure the mathematical artifacts are visually undetectable and to calibrate the AI detectors.
* **`layer7_verifier.py`**: An advanced verification script to audit the layers of the image for hidden steganographic data.
* **`PROJECT_REPORT.md`**: An in-depth technical paper explaining the complex mathematics, the architecture, and the techniques used to evade AI forensic manipulation detectors.
* **`explanation.md`**: A simplified, high-level overview of how the math works for non-technical readers.

### 2. Web Interface (`webapp/`)
A Flask-based web application that provides a beautiful, modern UI to interact with the core engine.

* **`app.py`**: The Flask backend server. It provides the `/process` API endpoint which securely handles file uploads, routes the image to the correct mathematical script based on user input, and zips the results for download.
* **`templates/index.html`**: The frontend HTML structure, utilizing a glass-morphism aesthetic.
* **`static/js/main.js`**: The frontend logic that handles the drag-and-drop interface, API requests, and graceful error handling.
* **`static/css/style.css`**: The styling and animations for the web application.
* **`requirements.txt`**: The Python dependencies required to run the backend and the core mathematical scripts.

---

## How to Run Locally

Because the mathematical scripts are computationally heavy, ensure you are running this on a machine with a decent CPU and at least 1GB of free RAM.

1. Install dependencies:
   ```bash
   cd webapp
   pip install -r requirements.txt
   ```

2. Start the Flask server:
   ```bash
   python3 app.py
   ```

3. Open your browser and navigate to:
   **http://127.0.0.1:5000**
