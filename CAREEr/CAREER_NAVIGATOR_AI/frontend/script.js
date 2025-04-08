// Career Navigator AI - Student Dashboard Script
// Handles dynamic data loading and interactive features

document.addEventListener('DOMContentLoaded', function() {
    // Reference to UI elements
    const leaderboardBtn = document.getElementById('leaderboardBtn');
    const leaderboardSection = document.getElementById('leaderboardSection');
    const leaderboardBody = document.getElementById('leaderboardBody');
    const searchBtn = document.getElementById('searchBtn');
    const aiAssistantBtn = document.getElementById('aiAssistantBtn');
    const profileBtn = document.getElementById('profileBtn');
    const profileSection = document.getElementById('profileSection');
    const closeProfileBtn = document.querySelector('.close-profile-btn');
    const addPhotoBtn = document.getElementById('addPhotoBtn');
    const photoInput = document.getElementById('photoInput');
    const homeBtn = document.getElementById('homeBtn');
    const aiTestBtn = document.getElementById('aiTestBtn');
    const aiTestSection = document.getElementById('aiTestSection');
    
    // Create student progress container if it doesn't exist
    let studentProgressView = document.getElementById('studentProgressView');
    if (!studentProgressView) {
        studentProgressView = document.createElement('div');
        studentProgressView.id = 'studentProgressView';
        studentProgressView.className = 'student-progress-view';
        studentProgressView.style.display = 'none';
        document.querySelector('.dashboard-container').appendChild(studentProgressView);
    }
    
    // Toggle profile visibility
    profileBtn.addEventListener('click', function() {
        hideAllSections();
        profileSection.style.display = 'block';
        setTimeout(() => {
            profileSection.classList.add('fade-in');
        }, 50);
    });
    
    // Close profile when close button is clicked
    closeProfileBtn.addEventListener('click', function() {
        profileSection.classList.remove('fade-in');
        setTimeout(() => {
            profileSection.style.display = 'none';
        }, 300);
    });
    
    // Handle "Add Photo" button click
    addPhotoBtn.addEventListener('click', function() {
        photoInput.click();
    });
    
    // Handle photo upload
    photoInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('userPhoto').src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Handle home button
    homeBtn.addEventListener('click', hideAllSections);
    
    // Handle AI Test button
    aiTestBtn.addEventListener('click', function() {
        hideAllSections();
        aiTestSection.style.display = 'block';
        setTimeout(() => {
            aiTestSection.classList.add('fade-in');
        }, 50);
    });
    
    // Event listeners
    leaderboardBtn.addEventListener('click', function() {
        hideAllSections();
        leaderboardSection.style.display = 'block';
        
        // Load leaderboard data
        loadLeaderboardData();
        
        // Add fade-in animation to leaderboard
        leaderboardSection.classList.add('fade-in');
    });
    
    // Fix Search button navigation to roadmapGen app
    searchBtn.addEventListener('click', function() {
        // Run the roadmapGen app and redirect to it
        try {
            var shell = new ActiveXObject("WScript.Shell");
            shell.Run(
                "C:\\Users\\haran\\OneDrive\\Desktop\\CAREEr\\CAREER_NAVIGATOR_AI\\roadmapGen\\app.py",
                1,
                false
            );
            // Give the app a moment to start
            setTimeout(function () {
                window.location.href = "http://127.0.0.1:6000/";
            }, 2000);
        } catch (e) {
            // Fallback if ActiveX is not available or fails
            console.error("Could not start the application:", e);
            // Still try to redirect to the app
            window.location.href = "http://127.0.0.1:6000/";
        }
    });
    
    // Fix AI Assistant button functionality
    aiAssistantBtn.addEventListener('click', function() {
        // Start the chatbot application and redirect to it
        try {
            var shell = new ActiveXObject("WScript.Shell");
            shell.Run(
                "C:\\Users\\haran\\OneDrive\\Desktop\\CAREEr\\CAREER_NAVIGATOR_AI\\chatbot\\app.py",
                1,
                false
            );
            // Give the app a moment to start
            setTimeout(function () {
                window.location.href = "http://127.0.0.1:5001/";
            }, 2000);
        } catch (e) {
            // Fallback if ActiveX is not available or fails
            console.error("Could not start the application:", e);
            // Still try to redirect to the app
            window.location.href = "http://127.0.0.1:5001/";
        }
    });
    
    // Function to hide all sections
    function hideAllSections() {
        // Get all visible sections
        const sections = document.querySelectorAll('.search-box, .profile-section, .ai-test-section, .leaderboard-section, .student-progress-view');
        sections.forEach(section => {
            section.style.display = 'none';
            if (section.classList.contains('fade-in')) {
                section.classList.remove('fade-in');
            }
        });
    }
    
    // Function to load leaderboard data from CSV
    async function loadLeaderboardData() {
        try {
            const response = await fetch('../UPDATED_USER_DETAIL_NEW.csv');
            if (!response.ok) {
                throw new Error('Failed to load CSV file');
            }
            
            const data = await response.text();
            
            // Parse CSV data
            const rows = data.split('\n');
            const headers = rows[0].split(',');
            
            // Clear existing leaderboard
            leaderboardBody.innerHTML = '';
            
            // Update table header to match requirements
            const tableHeader = leaderboardSection.querySelector('thead tr');
            if (tableHeader) {
                tableHeader.innerHTML = `
                    <th>Rank</th>
                    <th>Name</th>
                    <th>Courses Completed</th>
                    <th>Skill Level</th>
                    <th>Badges Earned</th>
                `;
            }
            
            // Find index of required columns
            const nameIndex = headers.indexOf('Name');
            const rankIndex = headers.indexOf('Rank');
            const coursesCompletedIndex = headers.indexOf('Course completed %');
            const skillLevelIndex = headers.indexOf('Skill Level');
            const badgesEarnedIndex = headers.indexOf('Batch earned');
            
            // Process each row and add to leaderboard
            for (let i = 1; i < rows.length; i++) {
                if (rows[i].trim() === '') continue;
                
                const rowData = rows[i].split(',');
                if (rowData.length <= 1) continue; // Skip invalid rows
                
                const name = nameIndex >= 0 ? rowData[nameIndex] : 'Unknown';
                const rank = rankIndex >= 0 ? rowData[rankIndex] : 'N/A';
                const coursesCompleted = coursesCompletedIndex >= 0 ? rowData[coursesCompletedIndex] : '0';
                const skillLevel = skillLevelIndex >= 0 ? rowData[skillLevelIndex] : 'Beginner';
                const badgesEarned = badgesEarnedIndex >= 0 ? rowData[badgesEarnedIndex] : '0';
                
                if (!name || !rank) continue;
                
                // Create row with animation delay based on position
                const row = document.createElement('tr');
                row.className = 'leaderboard-row';
                row.style.animationDelay = `${(i - 1) * 0.1}s`;
                row.innerHTML = `
                    <td>${rank}</td>
                    <td>${name}</td>
                    <td>${coursesCompleted || '0'}%</td>
                    <td>${skillLevel || 'Beginner'}</td>
                    <td>${badgesEarned || '0'}</td>
                `;
                
                // Store full user data as dataset for future use
                row.dataset.userData = JSON.stringify(rowData);
                row.dataset.headers = JSON.stringify(headers);
                
                // Add click event to show student progress
                row.addEventListener('click', function() {
                    showStudentProgress(this);
                });
                
                leaderboardBody.appendChild(row);
            }
            
            // Add CSS animations for leaderboard rows
            document.head.insertAdjacentHTML('beforeend', `
                <style>
                    .leaderboard-row {
                        animation: fadeInRow 0.5s ease forwards;
                        opacity: 0;
                        transform: translateY(10px);
                    }
                    
                    @keyframes fadeInRow {
                        to {
                            opacity: 1;
                            transform: translateY(0);
                        }
                    }
                    
                    .leaderboard-row:hover {
                        background-color: rgba(124, 58, 237, 0.05) !important;
                        transform: translateY(-2px);
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    
                    .student-progress-view {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0, 0, 0, 0.75);
                        z-index: 1000;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        opacity: 0;
                        visibility: hidden;
                        transition: opacity 0.3s ease, visibility 0.3s ease;
                    }
                    
                    .student-progress-view.fade-in {
                        opacity: 1;
                        visibility: visible;
                    }
                    
                    .progress-card {
                        background-color: white;
                        width: 90%;
                        max-width: 800px;
                        border-radius: 16px;
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                        overflow: hidden;
                        transform: scale(0.9);
                        transition: transform 0.3s ease;
                    }
                    
                    .student-progress-view.fade-in .progress-card {
                        transform: scale(1);
                    }
                    
                    .card-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 20px 25px;
                        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                        color: white;
                    }
                    
                    .card-header h2 {
                        margin: 0;
                        font-size: 1.5rem;
                        font-weight: 500;
                        color: white;
                    }
                    
                    .close-btn {
                        background: none;
                        border: none;
                        color: white;
                        font-size: 1.8rem;
                        cursor: pointer;
                        transition: transform 0.2s ease;
                    }
                    
                    .close-btn:hover {
                        transform: rotate(90deg);
                    }
                    
                    .progress-statistics {
                        display: flex;
                        flex-wrap: wrap;
                        padding: 25px;
                        justify-content: space-between;
                        background-color: var(--light-bg);
                    }
                    
                    .stat-item {
                        flex: 1 1 150px;
                        margin: 10px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        text-align: center;
                    }
                    
                    .stat-circle-container {
                        width: 120px;
                        height: 120px;
                        margin-bottom: 10px;
                        position: relative;
                    }
                    
                    .stat-circle {
                        width: 100%;
                        height: 100%;
                        position: relative;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                    
                    .stat-value {
                        font-size: 1.5rem;
                        font-weight: 700;
                        color: var(--primary-color);
                    }
                    
                    .stat-label {
                        font-size: 0.9rem;
                        color: var(--text-secondary);
                        margin-top: 5px;
                    }
                    
                    .nft-status {
                        font-size: 1.2rem;
                    }
                    
                    .current-courses {
                        padding: 25px;
                    }
                    
                    .current-courses h3 {
                        margin-top: 0;
                        margin-bottom: 20px;
                        color: var(--text-primary);
                        font-weight: 500;
                    }
                    
                    .courses-container {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 15px;
                    }
                    
                    .course-badge {
                        display: flex;
                        align-items: center;
                        padding: 10px 15px;
                        background-color: var(--light-bg);
                        border-radius: 30px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                        transition: all 0.2s ease;
                        cursor: default;
                    }
                    
                    .course-badge:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        background-color: rgba(124, 58, 237, 0.1);
                    }
                    
                    .course-icon {
                        margin-right: 10px;
                        font-size: 1.2rem;
                    }
                    
                    .course-name {
                        font-size: 0.9rem;
                    }
                    
                    .no-courses {
                        padding: 15px;
                        color: var(--text-secondary);
                        font-style: italic;
                    }
                </style>
            `);
        } catch (error) {
            console.error('Error loading leaderboard data:', error);
            leaderboardBody.innerHTML = '<tr><td colspan="5">Error loading leaderboard data. Please try again.</td></tr>';
        }
    }
    
    // Function to show student progress when clicking on leaderboard entry
    function showStudentProgress(row) {
        // Extract user data
        const userData = JSON.parse(row.dataset.userData);
        const headers = JSON.parse(row.dataset.headers);
        
        // Find indices for all required fields
        const findIndex = (field) => {
            const index = headers.indexOf(field);
            return index >= 0 ? index : -1;
        };
        
        const nameIndex = findIndex('Name');
        const totalProgressIndex = findIndex('Total Progress (%)');
        const skillLevelIndex = findIndex('Skill Level');
        const streakIndex = findIndex('Streak');
        const courseLearningIndex = findIndex('course_learning');
        const totalDaysIndex = findIndex('total_days activated');
        const rankIndex = findIndex('Rank');
        const nftStatusIndex = findIndex('NFT Verification');
        
        // Get specific data fields with fallbacks
        const name = nameIndex >= 0 ? userData[nameIndex] : 'Unknown';
        const totalProgress = totalProgressIndex >= 0 ? userData[totalProgressIndex] : '0';
        const skillLevel = skillLevelIndex >= 0 ? userData[skillLevelIndex] : 'Beginner';
        const streak = streakIndex >= 0 ? userData[streakIndex] : '0';
        const courseLearning = courseLearningIndex >= 0 ? userData[courseLearningIndex] : '';
        const totalDays = totalDaysIndex >= 0 ? userData[totalDaysIndex] : '0';
        const rank = rankIndex >= 0 ? userData[rankIndex] : 'N/A';
        const nftStatus = nftStatusIndex >= 0 ? userData[nftStatusIndex] : 'No';
        
        // Prepare courses learning array (handle quoted strings correctly)
        let coursesArray = [];
        if (courseLearning) {
            if (courseLearning.startsWith('"') && courseLearning.endsWith('"')) {
                // Remove outer quotes and split by commas
                const cleanedString = courseLearning.substring(1, courseLearning.length - 1);
                coursesArray = cleanedString.split(',').map(course => course.trim());
            } else {
                coursesArray = courseLearning.split(',').map(course => course.trim());
            }
        }
        
        // Create and show the student progress view
        studentProgressView.innerHTML = `
            <div class="progress-card">
                <div class="card-header">
                    <h2>${name}'s Learning Progress</h2>
                    <button class="close-btn">&times;</button>
                </div>
                <div class="progress-statistics">
                    <div class="stat-item">
                        <div class="stat-circle-container">
                            <div class="stat-circle" data-progress="${totalProgress || 0}">
                                <svg class="progress-ring" width="120" height="120" viewBox="0 0 120 120">
                                    <circle class="progress-ring-circle-bg" cx="60" cy="60" r="54" stroke="#e2e8f0" stroke-width="6" fill="none"></circle>
                                    <circle class="progress-ring-circle" cx="60" cy="60" r="54" stroke="url(#gradient)" stroke-width="6" fill="none" stroke-linecap="round" transform="rotate(-90 60 60)"></circle>
                                </svg>
                                <div class="stat-value">${totalProgress || 0}%</div>
                                
                                <svg width="0" height="0" style="position:absolute">
                                    <defs>
                                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" stop-color="#7c3aed" />
                                            <stop offset="100%" stop-color="#3b82f6" />
                                        </linearGradient>
                                    </defs>
                                </svg>
                            </div>
                        </div>
                        <div class="stat-label">Total Progress</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-info">
                            <div class="stat-value">${skillLevel || 'Beginner'}</div>
                            <div class="stat-label">Skill Level</div>
                        </div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-info">
                            <div class="stat-value">${streak || 0}</div>
                            <div class="stat-label">Day Streak</div>
                        </div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-info">
                            <div class="stat-value">${totalDays || 0}</div>
                            <div class="stat-label">Total Days</div>
                        </div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-info">
                            <div class="stat-value">#${rank || 'N/A'}</div>
                            <div class="stat-label">Rank</div>
                        </div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-info">
                            <div class="stat-value nft-status" style="color: ${nftStatus === 'Yes' ? '#16a34a' : '#ef4444'}">
                                ${nftStatus === 'Yes' ? '✓ Verified' : '✗ Not Verified'}
                            </div>
                            <div class="stat-label">NFT Certification</div>
                        </div>
                    </div>
                </div>
                
                <div class="current-courses">
                    <h3>Currently Learning</h3>
                    <div class="courses-container">
                        ${coursesArray.length > 0 ? 
                            coursesArray.map(course => `
                                <div class="course-badge">
                                    <div class="course-icon"><i class="fas fa-book"></i></div>
                                    <div class="course-name">${course}</div>
                                </div>
                            `).join('') : 
                            '<div class="no-courses">No courses currently in progress</div>'
                        }
                    </div>
                </div>
            </div>
        `;
        
        // Display student progress with animation
        studentProgressView.style.display = 'flex';
        setTimeout(() => {
            studentProgressView.classList.add('fade-in');
            
            // Initialize progress circles
            initializeProgressCircles();
        }, 50);
        
        // Add close button event listener
        studentProgressView.querySelector('.close-btn').addEventListener('click', function() {
            studentProgressView.classList.remove('fade-in');
            setTimeout(() => {
                studentProgressView.style.display = 'none';
            }, 300);
        });
    }
    
    // Function to initialize animated progress circles
    function initializeProgressCircles() {
        const circles = document.querySelectorAll('.stat-circle');
        circles.forEach(circle => {
            const progress = parseInt(circle.dataset.progress) || 0;
            
            // Get the progress circle element
            const progressCircle = circle.querySelector('.progress-ring-circle');
            if (progressCircle) {
                const radius = progressCircle.r.baseVal.value;
                const circumference = radius * 2 * Math.PI;
                const offset = circumference - (progress / 100) * circumference;
                
                progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
                progressCircle.style.strokeDashoffset = circumference; // Start at 0%
                
                // Animate the progress
                setTimeout(() => {
                    progressCircle.style.transition = 'stroke-dashoffset 1.5s ease-in-out';
                    progressCircle.style.strokeDashoffset = offset;
                }, 100);
            }
        });
    }

    // Initialize UI components
    initializeCircularProgress();
    registerEventListeners();
    animateStats();
});

// Initialize circular progress bars
function initializeCircularProgress() {
    const progressBars = document.querySelectorAll('.circular-progress');
    
    progressBars.forEach(bar => {
        const progress = bar.getAttribute('data-progress');
        const circle = bar.querySelector('.circle');
        
        if (circle) {
            const radius = circle.r.baseVal.value;
            const circumference = radius * 2 * Math.PI;
            
            circle.style.strokeDasharray = `${circumference} ${circumference}`;
            circle.style.strokeDashoffset = circumference;
            
            // Animate the progress bar
            setTimeout(() => {
                const offset = circumference - (progress / 100) * circumference;
                circle.style.strokeDashoffset = offset;
            }, 500);
        }
    });
}

// Register event listeners for UI components
function registerEventListeners() {
    // Navigation buttons
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(button => {
        button.addEventListener('mouseenter', createRippleEffect);
    });
    
    // Add hover effects to stat cards
    const statCards = document.querySelectorAll('.stat-card, .metric-box');
    statCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-5px)';
            card.style.boxShadow = '0 15px 30px rgba(0, 0, 0, 0.15)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
            card.style.boxShadow = '';
        });
    });
    
    // Add photo button hover effect
    const addPhotoBtn = document.getElementById('addPhotoBtn');
    if (addPhotoBtn) {
        addPhotoBtn.addEventListener('mouseenter', () => {
            addPhotoBtn.style.transform = 'scale(1.05)';
        });
        
        addPhotoBtn.addEventListener('mouseleave', () => {
            addPhotoBtn.style.transform = '';
        });
    }

    // Profile close button animation
    const closeProfileBtn = document.querySelector('.close-profile-btn');
    if (closeProfileBtn) {
        closeProfileBtn.addEventListener('mouseenter', () => {
            closeProfileBtn.style.transform = 'rotate(90deg)';
        });
        
        closeProfileBtn.addEventListener('mouseleave', () => {
            closeProfileBtn.style.transform = '';
        });
    }
}

// Create ripple effect on button click
function createRippleEffect(e) {
    const button = e.currentTarget;
    
    const ripple = document.createElement('span');
    ripple.classList.add('ripple-effect');
    
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    
    ripple.style.width = ripple.style.height = `${size}px`;
    ripple.style.left = `${e.clientX - rect.left - size / 2}px`;
    ripple.style.top = `${e.clientY - rect.top - size / 2}px`;
    
    button.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
    }, 600);
}

// Animate stats with counting effect
function animateStats() {
    const streakValue = document.querySelector('.streak-value');
    const timeValue = document.querySelector('.time-value');
    
    if (streakValue) {
        animateCounter(streakValue, 0, 51, 1500);
    }
    
    if (timeValue) {
        animateCounter(timeValue, 0, 751, 2000);
    }
    
    // Animate metric numbers
    const metricNumbers = document.querySelectorAll('.metric-number');
    metricNumbers.forEach(number => {
        const finalValue = parseInt(number.textContent);
        animateCounter(number, 0, finalValue, 1500);
    });
}

// Animate counter from start to end value
function animateCounter(element, start, end, duration) {
    let startTime = null;
    
    function updateCounter(timestamp) {
        if (!startTime) startTime = timestamp;
        
        const progress = Math.min((timestamp - startTime) / duration, 1);
        const currentValue = Math.floor(progress * (end - start) + start);
        
        element.textContent = currentValue;
        
        if (progress < 1) {
            window.requestAnimationFrame(updateCounter);
        } else {
            element.textContent = end;
        }
    }
    
    window.requestAnimationFrame(updateCounter);
}

// Toggle profile section display
function toggleProfileSection() {
    const profileSection = document.getElementById('profileSection');
    
    if (profileSection.style.display === 'none' || !profileSection.style.display) {
        // Show profile with animation
        profileSection.style.display = 'flex';
        setTimeout(() => {
            profileSection.style.opacity = '1';
        }, 10);
    } else {
        // Hide profile with animation
        profileSection.style.opacity = '0';
        setTimeout(() => {
            profileSection.style.display = 'none';
        }, 300);
    }
}

// Update profile photo when selected
function updateProfilePhoto(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            document.getElementById('userPhoto').src = e.target.result;
        };
        
        reader.readAsDataURL(input.files[0]);
    }
}

// Add ripple effect style
document.head.insertAdjacentHTML('beforeend', `
    <style>
        .ripple-effect {
            position: absolute;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.3);
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        }
        
        @keyframes ripple {
            to {
                transform: scale(2);
                opacity: 0;
            }
        }
        
        .profile-section {
            transition: opacity 0.3s ease;
        }
        
        .stat-card, .metric-box {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .close-profile-btn, .add-photo-btn {
            transition: transform 0.3s ease;
        }
        
        .leaderboard-section {
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .progress-ring-circle {
            transition: stroke-dashoffset 1.5s ease-in-out;
        }
    </style>
`); 