document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    
    const radioInputs = document.querySelectorAll('input[name="script_type"]');
    const presetGroup = document.getElementById('presetGroup');
    
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = document.getElementById('spinner');
    
    const resultSection = document.getElementById('resultSection');
    const pixelatedCard = document.getElementById('pixelatedCard');
    const pixelatedResult = document.getElementById('pixelatedResult');
    const restoredCard = document.getElementById('restoredCard');
    const restoredResult = document.getElementById('restoredResult');
    const downloadAllBtn = document.getElementById('downloadAllBtn');

    // Drag and Drop Logic
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('is-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('is-active');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        if(files.length > 0) {
            fileInput.files = files;
            handleFiles(files[0]);
        }
    });

    fileInput.addEventListener('change', function() {
        if(this.files.length > 0) {
            handleFiles(this.files[0]);
        }
    });

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }
        
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            dropArea.style.display = 'none';
            previewContainer.style.display = 'flex';
        }
    }

    removeBtn.addEventListener('click', () => {
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.style.display = 'none';
        dropArea.style.display = 'flex';
        resultSection.style.display = 'none';
    });

    // Radio logic for preset
    radioInputs.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'pixelate2') {
                presetGroup.style.display = 'block';
                presetGroup.style.animation = 'slideUp 0.3s ease-out';
            } else {
                presetGroup.style.display = 'none';
            }
        });
    });

    // Initial state trigger
    const initialRadio = document.querySelector('input[name="script_type"]:checked');
    if (initialRadio && initialRadio.value !== 'pixelate2') {
        presetGroup.style.display = 'none';
    }

    // Form Submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert('Please select an image first.');
            return;
        }

        const formData = new FormData(uploadForm);
        
        // UI Loading State
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        resultSection.style.display = 'none';

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });
            
            let data;
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.indexOf("application/json") !== -1) {
                data = await response.json();
            } else {
                const text = await response.text();
                throw new Error(`Server error (${response.status}): ${text.substring(0, 100)}...`);
            }
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error');
            }

            if (data.success) {
                const scriptType = document.querySelector('input[name="script_type"]:checked').value;
                const isDeOnly = scriptType === 'de';
                const isCheck = scriptType === 'check_pixelated';
                
                const checkResultWrapper = document.getElementById('checkResultWrapper');
                const pixelatedWrapper = document.getElementById('pixelatedWrapper');
                const outputBadge = document.getElementById('outputBadge');
                const checkResultText = document.getElementById('checkResultText');
                const checkResultDetails = document.getElementById('checkResultDetails');

                if (isCheck) {
                    // Fetch report.json to show results
                    pixelatedWrapper.style.display = 'none';
                    checkResultWrapper.style.display = 'block';
                    outputBadge.textContent = 'Analyzed';
                    pixelatedCard.style.display = 'block';
                    restoredCard.style.display = 'none';
                    
                    try {
                        const reportResponse = await fetch(data.report_url);
                        const report = await reportResponse.json();
                        
                        if (report.pixelated) {
                            checkResultText.textContent = '⚠️ Image is Pixelated';
                            checkResultText.style.color = '#f87171';
                            checkResultDetails.textContent = `Block size: ${report.block_size}px, Strength: ${report.strength}, Flat Ratio: ${report.flat_ratio}`;
                        } else {
                            checkResultText.textContent = '✅ Image is NOT Pixelated';
                            checkResultText.style.color = '#34d399';
                            checkResultDetails.textContent = 'No degradation grid detected.';
                        }
                    } catch (e) {
                        checkResultText.textContent = 'Error parsing report';
                        checkResultDetails.textContent = e.message;
                    }

                } else if (isDeOnly) {
                    pixelatedCard.style.display = 'none';
                    checkResultWrapper.style.display = 'none';
                    pixelatedWrapper.style.display = 'block';
                    outputBadge.textContent = 'Processed';
                } else {
                    pixelatedCard.style.display = 'block';
                    checkResultWrapper.style.display = 'none';
                    pixelatedWrapper.style.display = 'block';
                    outputBadge.textContent = 'Processed';
                    pixelatedResult.src = data.pixelated_url;
                }
                
                if (!isCheck) {
                    if (data.restored_url) {
                        restoredResult.src = data.restored_url;
                        restoredCard.style.display = 'block';
                    } else {
                        restoredCard.style.display = 'none';
                        if (isDeOnly) {
                            alert("Depixelate check complete. No degradation detected, so no restored image was generated.");
                        }
                    }
                }
                
                downloadAllBtn.href = data.download_url;
                resultSection.style.display = 'block';
                
                // Scroll to results
                setTimeout(() => {
                    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
        } catch (error) {
            alert('Error processing image: ' + error.message);
        } finally {
            // Restore UI State
            submitBtn.disabled = false;
            btnText.style.display = 'block';
            spinner.style.display = 'none';
        }
    });
});
