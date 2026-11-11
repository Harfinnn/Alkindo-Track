
const LOGIN_URL = "/login.html";
const CHECK_INTERVAL = 30000;

let checking = false;

async function checkAuthentication() {
    if (checking) return false;

    checking = true;

    try {
        const response = await fetch("/api/auth/me", {
            method: "GET",
            cache: "no-store",
            credentials: "same-origin"
        });

        if (response.status === 401) {
            window.location.replace(LOGIN_URL);
            return false;
        }

        if (!response.ok) {
            console.error(
                "[GateWatch Auth] Authentication check failed:",
                response.status
            );
            return false;
        }

        return true;

    } catch (error) {
        console.error(
            "[GateWatch Auth] Connection error:",
            error
        );

        return false;

    } finally {
        checking = false;
    }
}

// Periksa sesi ketika halaman dibuka.
checkAuthentication();

// Periksa ulang setiap 30 detik.
const authInterval = window.setInterval(
    checkAuthentication,
    CHECK_INTERVAL
);

// Logout menggunakan event delegation,
// sehingga tetap bekerja pada sidebar yang dimuat dinamis.
document.addEventListener("click", async (event) => {
    const logoutButton = event.target.closest("[data-logout]");

    if (!logoutButton) return;

    logoutButton.disabled = true;

    try {
        await fetch("/api/auth/logout", {
            method: "POST",
            credentials: "same-origin"
        });
    } catch (error) {
        console.error("[GateWatch Auth] Logout:", error);
    }

    window.clearInterval(authInterval);
    window.location.replace(LOGIN_URL);
});
