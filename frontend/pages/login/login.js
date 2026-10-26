// frontend/pages/login/login.js

document.addEventListener("DOMContentLoaded", () => {
    redirectIfAuthenticated();

    const form = document.getElementById("loginForm");
    const phoneInput = document.getElementById("phone");
    const passwordInput = document.getElementById("password");

    const phoneError = document.getElementById("phoneError");
    const passwordError = document.getElementById("passwordError");
    const formError = document.getElementById("formError");
    const attemptWarning = document.getElementById("attemptWarning");

    const loginButton = document.getElementById("loginButton");
    const loginButtonText = document.getElementById("loginButtonText");
    const loginSpinner = document.getElementById("loginSpinner");

    const showPasswordButton = document.getElementById("showPassword");

    // Lockout modal elements
    const lockoutModal = document.getElementById("lockoutModal");
    const lockoutCountdown = document.getElementById("lockoutCountdown");
    const lockoutDismissBtn = document.getElementById("lockoutDismissBtn");
    const lockoutBackdrop = document.getElementById("lockoutBackdrop");

    let lockoutTimer = null;

    function clearErrors() {
        phoneError.textContent = "";
        passwordError.textContent = "";
        formError.textContent = "";
        if (attemptWarning) {
            attemptWarning.textContent = "";
            attemptWarning.hidden = true;
        }

        phoneInput.classList.remove("invalid");
        passwordInput.classList.remove("invalid");
    }

    function validate() {
        let valid = true;

        const phone = phoneInput.value.trim();
        const password = passwordInput.value;

        if (!phone) {
            phoneError.textContent = "Phone number is required.";
            phoneInput.classList.add("invalid");
            valid = false;
        }

        if (!password) {
            passwordError.textContent = "Password is required.";
            passwordInput.classList.add("invalid");
            valid = false;
        }

        return valid;
    }

    function setLoading(loading) {
        loginButton.disabled = loading;
        loginButtonText.textContent = loading ? "Signing in..." : "Sign in";
        loginSpinner.hidden = !loading;
    }

    function setInputsDisabled(disabled) {
        phoneInput.disabled = disabled;
        passwordInput.disabled = disabled;
        loginButton.disabled = disabled;
        if (showPasswordButton) {
            showPasswordButton.disabled = disabled;
        }
    }

    function startLockoutCountdown(totalSeconds) {
        if (lockoutTimer) {
            clearInterval(lockoutTimer);
        }

        let secondsLeft = Math.max(1, totalSeconds || 900);
        setInputsDisabled(true);

        function updateDisplay() {
            const minutes = Math.floor(secondsLeft / 60);
            const seconds = secondsLeft % 60;
            const displayStr = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

            if (lockoutCountdown) {
                lockoutCountdown.textContent = displayStr;
            }

            if (secondsLeft <= 0) {
                clearInterval(lockoutTimer);
                lockoutTimer = null;
                closeLockoutModal();
                setInputsDisabled(false);
                if (formError) {
                    formError.textContent = "Lockout period expired. You may now attempt to sign in again.";
                    formError.style.color = "var(--navy)";
                }
            } else {
                secondsLeft--;
            }
        }

        updateDisplay();
        lockoutTimer = setInterval(updateDisplay, 1000);
    }

    function openLockoutModal(remainingSeconds) {
        if (!lockoutModal) return;
        lockoutModal.classList.add("active");
        lockoutModal.setAttribute("aria-hidden", "false");
        startLockoutCountdown(remainingSeconds);
    }

    function closeLockoutModal() {
        if (!lockoutModal) return;
        lockoutModal.classList.remove("active");
        lockoutModal.setAttribute("aria-hidden", "true");
    }

    if (lockoutDismissBtn) {
        lockoutDismissBtn.addEventListener("click", closeLockoutModal);
    }
    if (lockoutBackdrop) {
        lockoutBackdrop.addEventListener("click", closeLockoutModal);
    }

    showPasswordButton.addEventListener("click", () => {
        const isPassword = passwordInput.type === "password";
        passwordInput.type = isPassword ? "text" : "password";
        showPasswordButton.textContent = isPassword ? "Hide" : "Show";
    });

    phoneInput.addEventListener("input", clearErrors);
    passwordInput.addEventListener("input", clearErrors);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        clearErrors();

        if (!validate()) {
            return;
        }

        setLoading(true);

        try {
            const data = await apiRequest("/api/auth/login", {
                method: "POST",
                body: JSON.stringify({
                    phone: phoneInput.value.trim(),
                    password: passwordInput.value,
                }),
            });

            if (!data.access_token) {
                throw new Error("Login succeeded, but no access token was returned.");
            }

            // The backend stores the refresh token in an HttpOnly cookie.
            // JavaScript only keeps the short-lived access token.
            saveAccessToken(data.access_token);

            window.location.href = "../dashboard/dashboard.html";

        } catch (error) {
            console.error("Login attempt failed:", error);

            const status = error.status || 0;
            const errData = error.data || error.payload || {};
            const detail = typeof error.detail === "object" ? error.detail : (errData.detail || {});

            const isLocked = status === 423 || detail.locked === true || errData.locked === true;
            const isConcurrentBlocked = status === 409 || detail.error === "CONCURRENT_SESSION_BLOCKED" || errData.error === "CONCURRENT_SESSION_BLOCKED";

            if (isLocked) {
                const remaining = detail.remaining_seconds || errData.remaining_seconds || 900;
                openLockoutModal(remaining);
                if (formError) {
                    formError.textContent = detail.message || "Account temporarily locked.";
                }
            } else if (isConcurrentBlocked) {
                if (formError) {
                    formError.textContent = detail.message || "This account is currently active on another device or platform. Please sign out from that device first.";
                    formError.style.color = "var(--error)";
                }
            } else {
                const attemptsLeft = detail.attempts_left !== undefined ? detail.attempts_left : errData.attempts_left;

                if (attemptsLeft !== undefined && attemptsLeft <= 3 && attemptWarning) {
                    attemptWarning.textContent = `Warning: ${attemptsLeft} attempt${attemptsLeft === 1 ? "" : "s"} remaining before temporary account lockout.`;
                    attemptWarning.hidden = false;
                }

                if (formError) {
                    formError.textContent = (typeof detail === "string" ? detail : detail.message) || error.message || "Invalid phone or password.";
                    formError.style.color = "var(--error)";
                }
            }

        } finally {
            setLoading(false);
        }
    });
});
