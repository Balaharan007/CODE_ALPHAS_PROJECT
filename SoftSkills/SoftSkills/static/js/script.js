document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    const recordingIndicator = document.getElementById('recording-indicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const aboutLink = document.getElementById('aboutLink');
    const aboutModal = document.getElementById('aboutModal');
    const closeModal = document.getElementById('closeModal');

    // Results elements
    const emotionResult = document.getElementById('emotionResult');
    const sentimentResult = document.getElementById('sentimentResult');
    const pitchResult = document.getElementById('pitchResult');
    const intensityResult = document.getElementById('intensityResult');
    const transcriptionText = document.getElementById('transcriptionText');
    const emotionScore = document.getElementById('emotionScore');
    const emotionScoreBar = document.getElementById('emotionScoreBar');
    const sentimentScore = document.getElementById('sentimentScore');
    const sentimentScoreBar = document.getElementById('sentimentScoreBar');
    const pitchScore = document.getElementById('pitchScore');
    const pitchScoreBar = document.getElementById('pitchScoreBar');
    const intensityScore = document.getElementById('intensityScore');
    const intensityScoreBar = document.getElementById('intensityScoreBar');
    const finalScore = document.getElementById('finalScore');
    const suggestionsText = document.getElementById('suggestionsText');

    // Event listeners
    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    aboutLink.addEventListener('click', openAboutModal);
    closeModal.addEventListener('click', closeAboutModal);

    // Modal close on click outside
    window.addEventListener('click', function(event) {
        if (event.target === aboutModal) {
            closeAboutModal();
        }
    });

    // Functions
    function startRecording() {
        console.log("Starting recording...");
        fetch('/start_recording', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update UI
                startBtn.disabled = true;
                stopBtn.disabled = false;
                recordingIndicator.classList.remove('hidden');
                resultsContainer.classList.add('hidden');
                console.log("Recording started successfully");
            }
        })
        .catch(error => {
            console.error('Error starting recording:', error);
            alert('Failed to start recording. Please try again.');
        });
    }

    function stopRecording() {
        console.log("Stopping recording and starting analysis...");
        loadingOverlay.classList.remove('hidden');
        
        fetch('/stop_recording', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Analysis data received:", data.analysis);
            if (data.status === 'success') {
                displayResults(data.analysis);
            } else {
                console.error("Analysis failed:", data);
                alert('Analysis failed. Please try again.');
            }
        })
        .catch(error => {
            console.error('Error stopping recording:', error);
            alert('Failed to complete analysis. Please try again.');
        })
        .finally(() => {
            // Always reset UI state regardless of success/failure
            loadingOverlay.classList.add('hidden');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            recordingIndicator.classList.add('hidden');
        });
    }

    function displayResults(analysis) {
        // Display basic results
        emotionResult.textContent = analysis.emotion || 'Neutral';
        sentimentResult.textContent = analysis.sentiment || 'Neutral';
        pitchResult.textContent = `${analysis.pitch.toFixed(2)} Hz`;
        intensityResult.textContent = analysis.intensity.toFixed(4);
        
        // Display transcription
        if (analysis.transcribed_text && analysis.transcribed_text.trim() !== '') {
            transcriptionText.textContent = analysis.transcribed_text;
        } else {
            transcriptionText.textContent = 'No speech detected';
        }
        
        // Display scores
        updateScoreDisplay(emotionScore, emotionScoreBar, analysis.scores.emotion_score);
        updateScoreDisplay(sentimentScore, sentimentScoreBar, analysis.scores.sentiment_score);
        updateScoreDisplay(pitchScore, pitchScoreBar, analysis.scores.pitch_score);
        updateScoreDisplay(intensityScore, intensityScoreBar, analysis.scores.intensity_score);
        
        // Create dynamic HTML for additional scores if they exist (voice variability and tempo)
        if (analysis.scores.voice_variability_score !== undefined && analysis.scores.tempo_score !== undefined) {
            const scoresContainer = document.querySelector('.scores-container');
            
            // Only add new score elements if they don't already exist
            if (!document.getElementById('voiceVariabilityScore')) {
                // Voice Variability Score
                const voiceVariabilityHTML = createScoreItemHTML(
                    'Voice Variability', 
                    'voiceVariabilityScore', 
                    'voiceVariabilityScoreBar',
                    analysis.scores.voice_variability_score
                );
                
                // Tempo Score
                const tempoScoreHTML = createScoreItemHTML(
                    'Speech Tempo', 
                    'tempoScore', 
                    'tempoScoreBar',
                    analysis.scores.tempo_score
                );
                
                // Insert new elements at the end of scores container
                scoresContainer.insertAdjacentHTML('beforeend', voiceVariabilityHTML + tempoScoreHTML);
                
                // Initialize the new score bars
                updateScoreDisplay(
                    document.getElementById('voiceVariabilityScore'), 
                    document.getElementById('voiceVariabilityScoreBar'), 
                    analysis.scores.voice_variability_score
                );
                
                updateScoreDisplay(
                    document.getElementById('tempoScore'), 
                    document.getElementById('tempoScoreBar'), 
                    analysis.scores.tempo_score
                );
            } else {
                // Just update existing score elements
                updateScoreDisplay(
                    document.getElementById('voiceVariabilityScore'), 
                    document.getElementById('voiceVariabilityScoreBar'), 
                    analysis.scores.voice_variability_score
                );
                
                updateScoreDisplay(
                    document.getElementById('tempoScore'), 
                    document.getElementById('tempoScoreBar'), 
                    analysis.scores.tempo_score
                );
            }
        }
        
        // Display final score
        finalScore.textContent = analysis.scores.final_score.toFixed(1);
        
        // Display suggestions
        suggestionsText.innerHTML = formatSuggestions(analysis.suggestions);
        
        // Show results container
        resultsContainer.classList.remove('hidden');
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    function createScoreItemHTML(label, scoreId, barId, scoreValue) {
        return `
            <div class="score-item">
                <div class="score-label">${label} Score</div>
                <div class="score-bar-container">
                    <div class="score-bar" id="${barId}"></div>
                    <span class="score-value" id="${scoreId}">${scoreValue}/10</span>
                </div>
            </div>
        `;
    }

    function updateScoreDisplay(scoreElement, barElement, score) {
        scoreElement.textContent = `${score}/10`;
        barElement.style.width = `${score * 10}%`;
        
        // Change color based on score
        if (score >= 8) {
            barElement.style.background = 'linear-gradient(to right, #4CAF50, #2E7D32)';
        } else if (score >= 5) {
            barElement.style.background = 'linear-gradient(to right, #FFC107, #FF9800)';
        } else {
            barElement.style.background = 'linear-gradient(to right, #FF5722, #F44336)';
        }
    }

    function formatSuggestions(suggestions) {
        if (!suggestions) return 'No suggestions available.';
        
        // Replace markdown-style headers with HTML
        let formatted = suggestions
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
            
        return formatted;
    }

    function resetAnalysis() {
        resultsContainer.classList.add('hidden');
        
        // Reset all result fields
        emotionResult.textContent = '-';
        sentimentResult.textContent = '-';
        pitchResult.textContent = '-';
        intensityResult.textContent = '-';
        transcriptionText.textContent = 'No speech detected';
        emotionScore.textContent = '-/10';
        emotionScoreBar.style.width = '0%';
        sentimentScore.textContent = '-/10';
        sentimentScoreBar.style.width = '0%';
        pitchScore.textContent = '-/10';
        pitchScoreBar.style.width = '0%';
        intensityScore.textContent = '-/10';
        intensityScoreBar.style.width = '0%';
        
        // Reset additional score elements if they exist
        const voiceVariabilityScore = document.getElementById('voiceVariabilityScore');
        const voiceVariabilityScoreBar = document.getElementById('voiceVariabilityScoreBar');
        const tempoScore = document.getElementById('tempoScore');
        const tempoScoreBar = document.getElementById('tempoScoreBar');
        
        if (voiceVariabilityScore) {
            voiceVariabilityScore.textContent = '-/10';
            voiceVariabilityScoreBar.style.width = '0%';
        }
        
        if (tempoScore) {
            tempoScore.textContent = '-/10';
            tempoScoreBar.style.width = '0%';
        }
        
        finalScore.textContent = '-';
        suggestionsText.textContent = 'Analyze yourself to get personalized improvement suggestions.';
        
        // Scroll back to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function openAboutModal() {
        aboutModal.classList.remove('hidden');
    }

    function closeAboutModal() {
        aboutModal.classList.add('hidden');
    }
}); 