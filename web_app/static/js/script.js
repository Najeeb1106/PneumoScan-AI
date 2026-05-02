const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loader = document.getElementById('loader');
const previewSection = document.getElementById('preview-section');
const previewImg = document.getElementById('preview-img');
const initialState = document.getElementById('initial-state');
const resultsDisplay = document.getElementById('results-display');
const resultBadge = document.getElementById('result-badge');
const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');

let probabilityChart = null;

// Chart.js global defaults for dark mode
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
fileInput.addEventListener('change', handleFileSelect, false);

function handleDrop(e) {
    handleFiles(e.dataTransfer.files);
}

function handleFileSelect(e) {
    handleFiles(e.target.files);
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            processImage(file);
        } else {
            alert('Please upload an image file (JPEG or PNG).');
        }
    }
}

function processImage(file) {
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        previewSection.style.display = 'flex'; // Changed to flex for new UI
        dropZone.style.display = 'none';
    };
    reader.readAsDataURL(file);

    initialState.style.display = 'none';
    resultsDisplay.style.display = 'none';
    loader.style.display = 'flex';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        setTimeout(() => {
            loader.style.display = 'none';
            if (data.error) {
                alert('Error: ' + data.error);
                resetUpload();
                return;
            }
            displayResults(data);
        }, 800); // 800ms delay for premium feel
    })
    .catch(err => {
        console.error(err);
        loader.style.display = 'none';
        alert('Analysis failed.');
        resetUpload();
    });
}

function displayResults(data) {
    resultsDisplay.style.display = 'block';
    
    const confVal = parseFloat(data.confidence);
    predictionText.textContent = data.prediction;
    confidenceText.textContent = `Confidence: ${data.confidence}`;
    predictionText.className = 'prediction-title';
    resultBadge.className = 'prediction-badge';

    let normalProb = 0;
    let pneumoProb = 0;

    if (data.prediction.includes('Pneumonia')) {
        predictionText.classList.add('text-danger');
        resultBadge.classList.add('badge-danger');
        resultBadge.innerHTML = '<i class="fa-solid fa-circle-exclamation"></i> Pathological Detect';
        pneumoProb = confVal;
        normalProb = 100 - confVal;
    } else {
        predictionText.classList.add('text-success');
        resultBadge.classList.add('badge-success');
        resultBadge.innerHTML = '<i class="fa-solid fa-check"></i> Normal Clear';
        normalProb = confVal;
        pneumoProb = 100 - confVal;
    }

    renderChart(normalProb, pneumoProb);
}

function renderChart(normalProb, pneumoProb) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    
    if (probabilityChart) {
        probabilityChart.destroy();
    }

    probabilityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Pneumonia'],
            datasets: [{
                data: [normalProb, pneumoProb],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)', // Emerald
                    'rgba(239, 68, 68, 0.8)'   // Red
                ],
                borderColor: '#18181b', // Match surface background
                borderWidth: 4,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    position: 'bottom',
                    labels: { color: '#94a3b8' }
                },
                tooltip: {
                    backgroundColor: 'rgba(9, 9, 11, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return ` ${context.label}: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            },
            cutout: '75%' // Thinner premium ring
        }
    });
}

function resetUpload() {
    fileInput.value = '';
    dropZone.style.display = 'flex';
    previewSection.style.display = 'none';
    
    loader.style.display = 'none';
    resultsDisplay.style.display = 'none';
    initialState.style.display = 'flex';
    
    if (probabilityChart) {
        probabilityChart.destroy();
        probabilityChart = null;
    }
}
