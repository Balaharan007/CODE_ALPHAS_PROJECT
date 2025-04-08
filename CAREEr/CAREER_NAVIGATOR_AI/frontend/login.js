function showLogin(type) {
    document.getElementById('login-form').classList.remove('hidden');
    document.getElementById('signup-form').classList.add('hidden');
    document.getElementById('form-title').innerText = type === 'user' ? "User Login" : "Admin Login";
}

function showSignup() {
    document.getElementById('signup-form').classList.remove('hidden');
    document.getElementById('login-form').classList.add('hidden');
}

function showLoginScreen() {
    document.getElementById('login-form').classList.remove('hidden');
    document.getElementById('signup-form').classList.add('hidden');
}

document.getElementById('loginForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const message = document.getElementById('message');

    message.textContent = "Logging in...";

    try {
        const response = await fetch('http://localhost:3000/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();
        if (response.ok) {
            // Store user data in localStorage
            localStorage.setItem('token', data.token);
            localStorage.setItem('userName', data.user.name);
            
            message.textContent = "Login successful! Redirecting to dashboard...";
            message.style.color = "green";
            
            // Redirect to dashboard
            setTimeout(() => {
                window.location.href = "dashboard.html";
            }, 1000);
        } else {
            message.textContent = data.message || "Login failed.";
            message.style.color = "red";
        }
    } catch (error) {
        console.error('Login error:', error);
        message.textContent = "Server error. Make sure the backend server is running.";
        message.style.color = "red";
    }
});
