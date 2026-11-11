
const form = document.getElementById("login-form");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");

const loginButton = document.getElementById("login-button");
const loginButtonText = document.getElementById("login-button-text");

const messageBox = document.getElementById("login-message");
const togglePassword = document.getElementById("toggle-password");

// =============================================
// MESSAGE
// =============================================

function showMessage(message) {
    messageBox.textContent = message;
    messageBox.hidden = false;
}

function hideMessage() {
    messageBox.textContent = "";
    messageBox.hidden = true;
}

// =============================================
// PASSWORD VISIBILITY
// =============================================

togglePassword.addEventListener("click", () => {
    const visible = passwordInput.type === "text";

    passwordInput.type = visible ? "password" : "text";
    togglePassword.textContent = visible ? "Show" : "Hide";

    togglePassword.setAttribute(
        "aria-label",
        visible ? "Tampilkan password" : "Sembunyikan password"
    );
});

// =============================================
// LOGIN
// =============================================

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    hideMessage();

    const username = usernameInput.value.trim();
    const password = passwordInput.value;

    if (!username || !password) {
        showMessage("Username dan password wajib diisi.");
        return;
    }

    loginButton.disabled = true;
    loginButtonText.textContent = "Authenticating...";

    try {
        const response = await fetch("/api/auth/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            credentials: "same-origin",
            body: JSON.stringify({
                username,
                password
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.message || "Login gagal."
            );
        }

        loginButtonText.textContent = "Login Successful...";

        window.location.replace(data.redirect || "/");

    } catch (error) {
        showMessage(error.message);

        loginButton.disabled = false;
        loginButtonText.textContent = "Sign In";

        passwordInput.value = "";
        passwordInput.focus();
    }
});
