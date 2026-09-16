/**
 * SpamGuard AI — Interactive Client Application
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const htmlElement = document.documentElement;
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const messageInput = document.getElementById("messageInput");
    const charCount = document.getElementById("charCount");
    const clearTextBtn = document.getElementById("clearTextBtn");
    const sampleSpamBtn = document.getElementById("sampleSpamBtn");
    const sampleHamBtn = document.getElementById("sampleHamBtn");
    const classificationForm = document.getElementById("classificationForm");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const loadingSpinner = document.getElementById("loadingSpinner");
    const errorBanner = document.getElementById("errorBanner");
    const errorMessage = document.getElementById("errorMessage");
    const idleState = document.getElementById("idleState");
    const activeResult = document.getElementById("activeResult");

    // Result Elements
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictTag = document.getElementById("verdictTag");
    const confidencePill = document.getElementById("confidencePill");
    const verdictExplanation = document.getElementById("verdictExplanation");
    const metricVerdict = document.getElementById("metricVerdict");
    const metricConfidence = document.getElementById("metricConfidence");
    const progressPercentLabel = document.getElementById("progressPercentLabel");
    const progressFill = document.getElementById("progressFill");
    const spamProbBar = document.getElementById("spamProbBar");
    const spamProbVal = document.getElementById("spamProbVal");
    const hamProbBar = document.getElementById("hamProbBar");
    const hamProbVal = document.getElementById("hamProbVal");
    const cleanedTokens = document.getElementById("cleanedTokens");
    const copyTokensBtn = document.getElementById("copyTokensBtn");
    const systemStatusBadge = document.getElementById("systemStatusBadge");

    // Samples
    const SAMPLES = {
        spam: "Congratulations! You have won a free iPhone. Click here to claim your prize.",
        ham: "Hi Deepak, please attend the project meeting tomorrow at 10 AM.",
    };

    // --- 1. Theme Management ---
    const savedTheme = localStorage.getItem("spamguard_theme") || "dark";
    htmlElement.setAttribute("data-theme", savedTheme);

    themeToggleBtn.addEventListener("click", () => {
        const currentTheme = htmlElement.getAttribute("data-theme");
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        htmlElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("spamguard_theme", newTheme);
    });

    // --- 2. Character Counter & Input Handling ---
    function updateCharCount() {
        const length = messageInput.value.length;
        charCount.textContent = length.toLocaleString();
    }

    messageInput.addEventListener("input", () => {
        updateCharCount();
        hideError();
    });

    // --- 3. Test Samples ---
    sampleSpamBtn.addEventListener("click", () => {
        messageInput.value = SAMPLES.spam;
        updateCharCount();
        hideError();
        messageInput.focus();
    });

    sampleHamBtn.addEventListener("click", () => {
        messageInput.value = SAMPLES.ham;
        updateCharCount();
        hideError();
        messageInput.focus();
    });

    // --- 4. Clear Text ---
    clearTextBtn.addEventListener("click", () => {
        messageInput.value = "";
        updateCharCount();
        hideError();
        messageInput.focus();
    });

    // --- 5. Error Banner Controls ---
    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.remove("hidden");
    }

    function hideError() {
        errorBanner.classList.add("hidden");
    }

    // --- 6. Form Submission & API Integration ---
    classificationForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        hideError();

        const rawText = messageInput.value.trim();
        if (!rawText) {
            showError("Please enter a message before analyzing.");
            messageInput.focus();
            return;
        }

        // Set Loading State
        setLoadingState(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: rawText }),
            });

            const data = await response.json();

            if (!response.ok) {
                const errDetail = data.detail || "An unexpected error occurred during prediction.";
                showError(errDetail);
                setLoadingState(false);
                return;
            }

            // Render Results
            renderResults(data);
        } catch (error) {
            console.error("Fetch error:", error);
            showError("Could not connect to the SpamGuard prediction server. Please verify the application is running.");
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            loadingSpinner.classList.remove("hidden");
        } else {
            analyzeBtn.disabled = false;
            loadingSpinner.classList.add("hidden");
        }
    }

    // --- 7. Render Result Data ---
    function renderResults(data) {
        idleState.classList.add("hidden");
        activeResult.classList.remove("hidden");

        const isSpam = data.is_spam;
        const confPct = data.confidence_percentage.toFixed(1) + "%";

        // Toggle Spam vs Ham classes
        verdictBanner.className = `verdict-banner ${isSpam ? "is-spam" : "is-ham"}`;
        metricVerdict.className = `metric-value highlight ${isSpam ? "is-spam" : "is-ham"}`;
        progressFill.parentElement.parentElement.className = `confidence-section ${isSpam ? "is-spam" : "is-ham"}`;

        // Verdict Icon
        if (isSpam) {
            verdictIcon.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>`;
        } else {
            verdictIcon.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>`;
        }

        // Text & Numbers
        verdictTag.textContent = data.prediction.toUpperCase();
        confidencePill.textContent = `${confPct} Match`;
        verdictExplanation.textContent = data.explanation;

        metricVerdict.textContent = data.prediction.toUpperCase();
        metricConfidence.textContent = confPct;

        // Confidence Progress Bar
        progressPercentLabel.textContent = confPct;
        progressFill.style.width = "0%";
        setTimeout(() => {
            progressFill.style.width = confPct;
        }, 50);

        // Probabilities Breakdown
        const spamProb = ((data.probabilities?.spam || 0) * 100).toFixed(1);
        const hamProb = ((data.probabilities?.ham || 0) * 100).toFixed(1);

        spamProbVal.textContent = `${spamProb}%`;
        hamProbVal.textContent = `${hamProb}%`;

        spamProbBar.style.width = "0%";
        hamProbBar.style.width = "0%";
        setTimeout(() => {
            spamProbBar.style.width = `${spamProb}%`;
            hamProbBar.style.width = `${hamProb}%`;
        }, 100);

        // NLP Processed Tokens
        cleanedTokens.textContent = data.cleaned_text || "(No relevant words remaining after stop-word filtration)";
    }

    // --- 8. Copy Processed Tokens ---
    copyTokensBtn.addEventListener("click", () => {
        const tokensText = cleanedTokens.textContent;
        if (!tokensText) return;
        navigator.clipboard.writeText(tokensText).then(() => {
            const originalText = copyTokensBtn.textContent;
            copyTokensBtn.textContent = "Copied!";
            setTimeout(() => {
                copyTokensBtn.textContent = originalText;
            }, 1800);
        });
    });

    // --- 9. Check Server Health on Load ---
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                if (data.status === "healthy") {
                    systemStatusBadge.querySelector(".status-indicator-dot").style.background = "#10b981";
                    systemStatusBadge.querySelector(".status-indicator-text").textContent = "ML Model Active";
                } else {
                    systemStatusBadge.querySelector(".status-indicator-dot").style.background = "#f59e0b";
                    systemStatusBadge.querySelector(".status-indicator-text").textContent = "Degraded Pipeline";
                }
            }
        } catch (e) {
            systemStatusBadge.querySelector(".status-indicator-dot").style.background = "#ef4444";
            systemStatusBadge.querySelector(".status-indicator-text").textContent = "Offline";
        }
    }

    checkHealth();
    updateCharCount();
});
