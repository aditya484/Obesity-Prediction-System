document.addEventListener('DOMContentLoaded', () => {
    // ---- DOM Elements ----
    
    // Navigation & Views
    const navNew = document.getElementById('nav-new-prediction');
    const navHistory = document.getElementById('nav-history');
    const viewPrediction = document.getElementById('prediction-view');
    const viewResult = document.getElementById('result-view');
    const viewHistory = document.getElementById('history-view');
    
    // Wizard Elements
    const steps = [document.getElementById('step1'), document.getElementById('step2'), document.getElementById('step3')];
    const stepIndicators = document.querySelectorAll('.step-indicator');
    const progressFill = document.getElementById('progress-fill');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitBtn');
    const form = document.getElementById('predictionForm');
    
    let currentStep = 0; // 0, 1, 2
    
    // No remaining dual-synced slider elements
    
    // Theme
    const themeCheckbox = document.getElementById('checkbox');
    const body = document.body;
    
    // Live BMI
    const heightInput = document.getElementById('Height');
    const weightInput = document.getElementById('Weight');
    const bmiLiveValue = document.getElementById('bmiLiveValue');
    
    // Results
    const retakeBtn = document.getElementById('retakeBtn');
    let radarChartObj = null;

    // ---- Theme Toggle ----
    // Check locally saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    body.setAttribute('data-theme', savedTheme);
    themeCheckbox.checked = savedTheme === 'dark';
    
    themeCheckbox.addEventListener('change', () => {
        const newTheme = themeCheckbox.checked ? 'dark' : 'light';
        body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        if(radarChartObj) {
            radarChartObj.options.scales.r.grid.color = newTheme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
            radarChartObj.options.scales.r.angleLines.color = newTheme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
            radarChartObj.options.plugins.legend.labels.color = newTheme === 'dark' ? '#f8fafc' : '#0f172a';
            radarChartObj.update();
        }
    });

    // ---- Navigation Views ----
    function switchView(viewId) {
        viewPrediction.classList.add('hidden');
        viewResult.classList.add('hidden');
        viewHistory.classList.add('hidden');
        
        navNew.classList.remove('active');
        navHistory.classList.remove('active');
        
        if (viewId === 'prediction') {
            viewPrediction.classList.remove('hidden');
            navNew.classList.add('active');
        } else if (viewId === 'result') {
            viewResult.classList.remove('hidden');
        } else if (viewId === 'history') {
            viewHistory.classList.remove('hidden');
            navHistory.classList.add('active');
            renderHistory();
        }
    }

    navNew.addEventListener('click', () => {
        resetWizard();
        switchView('prediction');
    });
    
    navHistory.addEventListener('click', () => switchView('history'));
    retakeBtn.addEventListener('click', () => {
        resetWizard();
        switchView('prediction');
    });

    // Form variables already directly correspond to the precise number input values now.

    // ---- Live BMI ----
    function updateBMI() {
        const h = parseFloat(heightInput.value);
        const w = parseFloat(weightInput.value);
        if (h > 0 && w > 0) {
            const bmi = (w / (h * h)).toFixed(1);
            bmiLiveValue.textContent = bmi;
            if (bmi < 18.5) bmiLiveValue.style.color = 'var(--c-primary)';
            else if (bmi < 25) bmiLiveValue.style.color = '#4ade80';
            else if (bmi < 30) bmiLiveValue.style.color = '#fbbf24';
            else bmiLiveValue.style.color = 'var(--c-accent)';
        }
    }
    heightInput.addEventListener('input', updateBMI);
    weightInput.addEventListener('input', updateBMI);
    updateBMI();

    // ---- Wizard Logic ----
    function updateWizard() {
        // Update DOM visibility
        steps.forEach((s, idx) => {
            if (idx === currentStep) s.classList.add('active-step');
            else s.classList.remove('active-step');
        });

        // Update Indicators
        stepIndicators.forEach((ind, idx) => {
            if (idx <= currentStep) ind.classList.add('active');
            else ind.classList.remove('active');
        });

        // Update Progress Bar
        const progressPercent = ((currentStep + 1) / steps.length) * 100;
        progressFill.style.width = `${progressPercent}%`;

        // Update Buttons
        if (currentStep === 0) {
            prevBtn.style.display = 'none';
        } else {
            prevBtn.style.display = 'flex';
        }

        if (currentStep === steps.length - 1) {
            nextBtn.style.display = 'none';
            submitBtn.style.display = 'flex';
        } else {
            nextBtn.style.display = 'flex';
            submitBtn.style.display = 'none';
        }
    }

    function resetWizard() {
        currentStep = 0;
        form.reset();
        updateBMI();
        updateWizard();
    }

    nextBtn.addEventListener('click', () => {
        if (currentStep < steps.length - 1) {
            currentStep++;
            updateWizard();
        }
    });

    prevBtn.addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            updateWizard();
        }
    });

    updateWizard();

    // ---- Form Submission & API Call ----
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        submitBtn.innerHTML = 'Analyzing...';
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            // Radio buttons and selectors return strings, ensure parsing for numerical keys
            const numKeys = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'];
            if (numKeys.includes(key)) {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                // Save to history
                saveToHistory(data, result);
                
                // Render Result Page
                renderResult(data, result);
                switchView('result');
            } else {
                alert('Error computing prediction: ' + result.error);
            }

        } catch (error) {
            console.error('Error:', error);
            alert('A network error occurred while communicating with the server.');
        } finally {
            submitBtn.innerHTML = 'Generate Report <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14m-7-7 7 7-7 7"></path></svg>';
            submitBtn.disabled = false;
        }
    });

    // ---- History Management ----
    function saveToHistory(inputData, result) {
        const history = JSON.parse(localStorage.getItem('obesityHistory') || '[]');
        const record = {
            id: Date.now(),
            date: new Date().toLocaleDateString() + ' ' + new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
            inputs: inputData,
            result: result
        };
        history.unshift(record);
        if(history.length > 20) history.pop(); // Keep last 20
        localStorage.setItem('obesityHistory', JSON.stringify(history));
    }

    function renderHistory() {
        const list = document.getElementById('historyList');
        const history = JSON.parse(localStorage.getItem('obesityHistory') || '[]');
        
        if (history.length === 0) {
            list.innerHTML = '<p style="color: var(--text-secondary); text-align:center; padding: 2rem;">No previous assessments found.</p>';
            return;
        }

        list.innerHTML = history.map(item => `
            <div class="history-item">
                <div>
                    <div class="hist-date">${item.date}</div>
                    <div class="hist-label gradient-text" style="font-size: 1.1rem; margin:0;">${item.result.obesity_level.replace(/_/g, ' ')}</div>
                </div>
                <div class="hist-bmi">BMI: ${item.result.calculated_bmi}</div>
            </div>
        `).join('');
    }

    document.getElementById('clearHistoryBtn').addEventListener('click', () => {
        if(confirm('Are you sure you want to clear your assessment history?')) {
            localStorage.removeItem('obesityHistory');
            renderHistory();
        }
    });

    // ---- Result Rendering & Chart ----
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', () => {
            const resultElement = document.getElementById('result-view');
            const btnsWrapper = document.querySelector('.result-header div');
            
            // Temporarily hide buttons for the print
            if (btnsWrapper) btnsWrapper.style.display = 'none';
            // Store original background to enforce light background for PDF if they are in dark mode
            const origBg = resultElement.style.backgroundColor;
            resultElement.style.padding = '2rem';
            resultElement.style.backgroundColor = document.body.getAttribute('data-theme') === 'dark' ? '#1e293b' : '#ffffff';

            const opt = {
                margin:       0.2, // smaller margin
                filename:     'ObesiCheck_Health_Report.pdf',
                image:        { type: 'jpeg', quality: 0.98 },
                html2canvas:  { scale: 2, useCORS: true },
                jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
            };
            
            html2pdf().set(opt).from(resultElement).save().then(() => {
                // Restore UI state
                if (btnsWrapper) btnsWrapper.style.display = 'flex';
                resultElement.style.padding = '';
                resultElement.style.backgroundColor = origBg;
            });
        });
    }

    function renderResult(inputs, result) {
        const formattedLabel = result.obesity_level.replace(/_/g, ' ');
        document.getElementById('resultLabel').textContent = formattedLabel;
        document.getElementById('resultBmi').textContent = result.calculated_bmi;
        document.getElementById('suggestionText').textContent = result.suggestion;

        const statusIndicator = document.getElementById('statusIndicator');
        let color = '#4ade80'; // default green
        if(formattedLabel.toLowerCase().includes('underweight')) color = '#60a5fa'; // blue
        else if(formattedLabel.toLowerCase().includes('overweight')) color = '#fbbf24'; // yellow
        else if(formattedLabel.toLowerCase().includes('obesity')) color = '#f43f5e'; // red
        statusIndicator.style.backgroundColor = color;

        drawRadarChart(inputs, color);
    }

    function drawRadarChart(inputs, colorHex) {
        const ctx = document.getElementById('radarChart').getContext('2d');
        
        const dataVals = [
            inputs.FCVC,        // Veggies (1-3) -> scale to ~1-5 base
            inputs.CH2O * 1.5,  // Water (1-3) -> scale
            inputs.FAF * 1.5,   // Physical Activity (0-3)
            inputs.Age / 20,    // Scaled down visually
            inputs.NCP          // Meals (1-4)
        ];

        const isDark = body.getAttribute('data-theme') === 'dark';
        const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
        const textColor = isDark ? '#f8fafc' : '#0f172a';

        if(radarChartObj) {
            radarChartObj.destroy();
        }

        radarChartObj = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Vegetable Intake', 'Hydration', 'Physical Activity', 'Age Factor', 'Daily Meals'],
                datasets: [{
                    label: 'Your Profile Matrix',
                    data: dataVals,
                    backgroundColor: `${colorHex}40`, // 25% opacity
                    borderColor: colorHex,
                    pointBackgroundColor: colorHex,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: colorHex,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: gridColor },
                        grid: { color: gridColor },
                        pointLabels: { color: textColor, font: {family: 'Outfit', size: 12} },
                        ticks: { display: false } // Hide numbers on scale
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: textColor, font: {family: 'Outfit'} }
                    }
                }
            }
        });
    }

});
