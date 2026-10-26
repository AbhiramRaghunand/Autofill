// frontend/js/auth.js

function saveAccessToken(token) {
    localStorage.setItem("access_token", token);
}

function getAccessToken() {
    return localStorage.getItem("access_token");
}

function clearAuthTokens() {
    localStorage.removeItem("access_token");
    // Clean up refresh tokens saved by older app versions.
    localStorage.removeItem("refresh_token");
}

function isTokenExpired(token) {
    try {
        const payload = token.split(".")[1];
        const decodedPayload = JSON.parse(
            atob(payload.replace(/-/g, "+").replace(/_/g, "/"))
        );

        return !decodedPayload.exp || decodedPayload.exp * 1000 <= Date.now();
    } catch {
        return true;
    }
}

function getUserRole() {
    const token = getAccessToken();
    if (!token) return null;
    try {
        const payload = token.split(".")[1];
        const decodedPayload = JSON.parse(
            atob(payload.replace(/-/g, "+").replace(/_/g, "/"))
        );
        return decodedPayload.role || "patient";
    } catch {
        return null;
    }
}

function isAdmin() {
    return getUserRole() === "admin";
}

function isAuthenticated() {
    const token = getAccessToken();

    // Keep protected pages reachable when the short-lived access token expires.
    // Their first API request will attempt cookie-based refresh. If that fails,
    // apiRequest clears the access token and redirects to login.
    return Boolean(token);
}

document.querySelectorAll(".brand[data-authenticated-href]").forEach((brand) => {
    const landingPage = brand.getAttribute("href");
    const dashboard = brand.dataset.authenticatedHref;

    const updateDestination = () => {
        brand.setAttribute("href", isAuthenticated() ? dashboard : landingPage);
    };

    updateDestination();
    brand.addEventListener("click", updateDestination);
});

async function logout() {
    const token = getAccessToken();
    try {
        if (typeof apiRequest === "function") {
            // Call the server even without a local access token: the cookie may
            // still exist, and apiRequest can refresh before retrying logout.
            await apiRequest("/api/auth/logout", { method: "POST" });
        }
    } catch (err) {
        console.warn("Server logout notification failed:", err);
    } finally {
        clearAuthTokens();
        window.location.href = "../login/login.html";
    }
}

/**
 * Use this at the beginning of protected pages.
 */
function requireAuthentication() {
    if (!isAuthenticated()) {
        window.location.replace("../login/login.html");
        return false;
    }
    return true;
}

/**
 * Use this at the beginning of admin-only pages.
 */
function requireAdmin() {
    if (!isAuthenticated()) {
        window.location.replace("../login/login.html");
        return false;
    }
    if (!isAdmin()) {
        window.location.replace("../dashboard/dashboard.html");
        return false;
    }
    return true;
}

/**
 * Use this on login/register pages if you don't want an already
 * authenticated user to see them.
 */
function redirectIfAuthenticated() {
    if (isAuthenticated()) {
        window.location.replace("../dashboard/dashboard.html");
        return true;
    }
    return false;
}
