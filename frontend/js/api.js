// frontend/js/api.js

const API_BASE_URL = String(window.MEDFORM_API_BASE_URL || "").replace(/\/$/, "");
// Share one refresh request among concurrent API calls.
let refreshPromise = null;

function showSecurityToast(message, isError = false) {
    let container = document.getElementById("securityToastContainer");
    if (!container) {
        container = document.createElement("div");
        container.id = "securityToastContainer";
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
        `;
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.style.cssText = `
        pointer-events: auto;
        padding: 12px 18px;
        border-radius: 10px;
        background: ${isError ? "#1e293b" : "#0f172a"};
        color: #ffffff;
        font-family: "Inter", sans-serif;
        font-size: 11px;
        font-weight: 500;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        border-left: 4px solid ${isError ? "#ef4444" : "#10b981"};
        opacity: 0;
        transform: translateY(-8px);
        transition: all 0.25s ease;
        max-width: 320px;
        line-height: 1.45;
    `;
    toast.textContent = message;
    container.appendChild(toast);

    requestAnimationFrame(() => {
        toast.style.opacity = "1";
        toast.style.transform = "translateY(0)";
    });

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(-8px)";
        setTimeout(() => toast.remove(), 250);
    }, 4500);
}

async function apiRequest(endpoint, options = {}, hasRetried = false) {
    const token = localStorage.getItem("access_token");

    const headers = {
        ...(options.headers || {}),
    };

    if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
        headers["Content-Type"] = "application/json";
    }

    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers,
        // Needed for the HttpOnly refresh cookie. Same-origin requests also work.
        credentials: "include",
    });

    const contentType = response.headers.get("content-type") || "";
    let data;

    if (contentType.includes("application/json")) {
        data = await response.json();
    } else {
        data = await response.text();
    }

    // Handle 401 Unauthorized with cookie-based automatic token refresh.
    // The refresh token is never read by JavaScript or sent in a JSON body.
    const isLoginEndpoint = endpoint.includes("/api/auth/login");
    const isRefreshEndpoint = endpoint.includes("/api/auth/refresh");

    if (response.status === 401 && !hasRetried && !isLoginEndpoint && !isRefreshEndpoint) {
        const isSuperseded = (
            data &&
            typeof data === "object" &&
            (data.detail?.error === "SESSION_SUPERSEDED" || data.error === "SESSION_SUPERSEDED")
        );

        let newAccessToken = null;

        if (!isSuperseded) {
            try {
                if (!refreshPromise) {
                    refreshPromise = (async () => {
                        const refreshRes = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
                            method: "POST",
                            credentials: "include",
                        });

                        if (!refreshRes.ok) {
                            throw new Error("Refresh session is no longer valid");
                        }

                        const refreshData = await refreshRes.json();
                        if (!refreshData.access_token) {
                            throw new Error("Refresh response did not contain an access token");
                        }

                        return refreshData.access_token;
                    })();
                }

                newAccessToken = await refreshPromise;
            } catch (refreshErr) {
                console.warn("Silent token refresh failed:", refreshErr);
            } finally {
                refreshPromise = null;
            }
        }

        if (newAccessToken) {
            localStorage.setItem("access_token", newAccessToken);

            // Retry once only; preserve caller options and all existing request headers.
            return apiRequest(
                endpoint,
                {
                    ...options,
                    headers: {
                        ...(options.headers || {}),
                        Authorization: `Bearer ${newAccessToken}`,
                    },
                },
                true
            );
        }

        // Remove any legacy refresh token left by an older version of the app.
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");

        if (window.location.pathname.includes("/pages/") && !window.location.pathname.includes("/login/")) {
            const noticeMsg = isSuperseded
                ? "You have been signed out because this account was logged in on another device or platform."
                : "Your session has expired. Please sign in again.";
            showSecurityToast(noticeMsg, true);
            setTimeout(() => {
                window.location.href = "../login/login.html";
            }, 1800);
        }
    }

    if (!response.ok) {
        if (response.status === 429) {
            showSecurityToast("Rate limit reached. Please wait a moment before trying again.", true);
        }

        let message = "Something went wrong.";
        let errorDetail = data?.detail || data;

        if (typeof data === "object" && data?.detail) {
            if (typeof data.detail === "string") {
                message = data.detail;
            } else if (typeof data.detail === "object" && data.detail.message) {
                message = data.detail.message;
            }
        } else if (typeof data === "string" && data.trim()) {
            message = data;
        }

        const customError = new Error(message);
        customError.status = response.status;
        customError.data = data;
        customError.detail = errorDetail;
        throw customError;
    }

    return data;
}
