document.getElementById('signupForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const name = document.getElementById('signup-name').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    const confirmPassword = document.getElementById('signup-confirm-password').value;
    const message = document.getElementById('signup-message');

    if (password !== confirmPassword) {
        message.textContent = "Passwords do not match!";
        return;
    }

    message.textContent = "Signing up...";

    try {
        const response = await fetch('http://localhost:3000/api/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, password })
        });

        const data = await response.json();
        if (response.ok) {
            message.textContent = "Signup successful! Please login.";
            setTimeout(() => showLoginScreen(), 1500);
        } else {
            message.textContent = data.message || "Signup failed.";
        }
    } catch (error) {
        message.textContent = "Server error. Try again later.";
    }
});
