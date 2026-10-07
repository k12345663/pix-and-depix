# Ministry of Truth | Pix & Depix
## A Simple Explanation of the Project

At its core, this project is a **secret-hiding (steganography) and forensics tool**. 

Normally, when you "pixelate" or blur an image to censor it, the original details are destroyed forever. This project does something magical: it makes an image *look* pixelated, but it secretly hides the original, crystal-clear details inside the image itself. 

Later on, the tool can perfectly recover the original image from the pixelated version.

---

### How is it approached? (The Magic Math)

The secret lies in "Frequency Domains" (using math called Fast Fourier Transforms). 
Imagine a song: you have the loud bass (low frequencies) and the quiet cymbals (high frequencies). 
Images have frequencies too! 

1. **Hiding (Multiplexing):** When you pixelate an image with this tool, it takes the original clear image and shifts its data into invisible "frequencies." It then places a standard pixelated grid over the visible frequencies. To the human eye, it looks like a normal, low-quality pixelated image.
2. **Recovering (Demultiplexing):** When you run the depixelate script, the computer ignores the visible pixelated blocks, searches the invisible frequencies, and extracts the hidden high-resolution details to reconstruct the original photo.

---

### What is happening and where? (The Codebase)

The project is split into two main parts: The **Web App** (User Interface) and the **Core Engine** (The Math).

#### 1. The Core Engine (The Math Scripts)
These files live in the main folder and do all the heavy lifting:
* **`pixelate.py`**: The "Hider". It takes a normal image, hides the details using complex math (`scipy` and `numpy`), and spits out a pixelated version.
* **`depixelate.py`**: The "Recoverer". It takes the pixelated image, reads the hidden frequencies, and spits out the restored original image.
* **`check_pixelation.py`**: The "Detector". A utility script that analyzes an image's blocks to determine if it has been artificially pixelated or degraded.

#### 2. The Web App (The Interface)
These files live inside the `webapp/` folder and make the tool easy to use in a browser:
* **`webapp/app.py` (The Boss):** This is the Flask backend server. When you click "Execute Operation", this file receives your image, decides which of the math scripts to run, creates a temporary folder to run them safely, and packages the results into a ZIP file.
* **`webapp/templates/index.html` (The Look):** The structure of the beautiful glass-morphism website you see.
* **`webapp/static/js/main.js` (The Logic):** This handles the drag-and-drop animations, talks to `app.py` to send the image, and gracefully displays error messages or the final recovered images on your screen without reloading the page.

---

### Summary of the Workflow

1. You upload a photo on the website (`index.html`).
2. The frontend (`main.js`) sends it to the backend (`app.py`).
3. The backend puts the image in a temporary folder and runs the heavy math scripts (`pixelate.py` or `depixelate.py`).
4. The math script spends 10-30 seconds crunching millions of numbers to either hide or recover data.
5. The backend gathers the resulting images, zips them up, and sends them back to the frontend to display!
