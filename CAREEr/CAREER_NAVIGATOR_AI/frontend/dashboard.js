document.addEventListener('DOMContentLoaded', function () {
    // Check if user is logged in
    const token = localStorage.getItem('token');
    const userName = localStorage.getItem('userName');

    if (!token) {
        window.location.href = 'login.html'; // Redirect to login if not authenticated
        return;
    }

    // Logout functionality
    document.getElementById('logoutBtn').addEventListener('click', function () {
        localStorage.removeItem('token');
        localStorage.removeItem('userName');
        window.location.href = 'login.html';
    });

    // Redirect to Student page
    document.getElementById('studentBtn').addEventListener('click', function () {
        window.location.href = 'student.html';
    });

    // Redirect to Job Seeker page
    document.getElementById('jobSeekerBtn').addEventListener('click', function () {
        window.location.href = 'jobseeker.html';
    });

    // Handle Recruiter button click
    document.getElementById('recruiterBtn').addEventListener('click', function () {
        const recruiterOptions = document.querySelector('.recruiter-options');
        if (recruiterOptions.style.display === 'none') {
            recruiterOptions.style.display = 'block';
        }
    });

    // Add event listeners for Candidate Search and Hiring Automation buttons
    document.addEventListener('click', function(event) {
        // Handle Candidate Search button click
        if (event.target.id === 'candidateSearchBtn') {
            window.location.href = 'recruiter.html';
        }
        
        // Handle Hiring Automation button click
        if (event.target.id === 'hiringAutomationBtn') {
            // Directly navigate to the malpractice detection app using the exact address
            window.open('http://127.0.0.1:3000/', '_blank');
        }
    });
});