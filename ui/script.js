document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const input = document.getElementById('news-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const historyContainer = document.getElementById('results-history');
    const welcomeSection = document.getElementById('welcome-section');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    // Auth Elements
    const authButtons = document.getElementById('auth-buttons');
    const userProfile = document.getElementById('user-profile');
    const displayUsername = document.getElementById('display-username');
    const logoutBtn = document.getElementById('logout-btn');
    
    // Modal Elements
    const authModal = document.getElementById('auth-modal');
    const authForm = document.getElementById('auth-form');
    const modalTitle = document.getElementById('modal-title');
    const authSubmit = document.getElementById('auth-submit');
    const switchAuthLink = document.getElementById('switch-to-signup');
    const closeModal = document.querySelector('.close-modal');
    
    let isLoginMode = true;

    // --- Init ---
    checkAuthStatus();

    // --- Auth Handlers ---
    async function checkAuthStatus() {
        try {
            const res = await fetch('/api/user');
            const data = await res.json();
            if (data.logged_in) {
                setLoggedIn(data.username);
            } else {
                setLoggedOut();
            }
        } catch (e) { console.error("Auth check failed"); }
    }

    function setLoggedIn(username) {
        authButtons.classList.add('hidden');
        userProfile.classList.remove('hidden');
        displayUsername.textContent = username;
    }

    function setLoggedOut() {
        authButtons.classList.remove('hidden');
        userProfile.classList.add('hidden');
    }

    document.getElementById('open-login').onclick = () => openModal(true);
    document.getElementById('open-signup').onclick = () => openModal(false);
    closeModal.onclick = () => authModal.classList.add('hidden');

    function openModal(loginMode) {
        isLoginMode = loginMode;
        modalTitle.textContent = isLoginMode ? "Connexion" : "Inscription";
        authSubmit.textContent = isLoginMode ? "Se connecter" : "S'inscrire";
        switchAuthLink.parentElement.innerHTML = isLoginMode ? 
            `Pas encore de compte ? <a href="#" id="switch-to-signup">S'inscrire</a>` :
            `Déjà un compte ? <a href="#" id="switch-to-login">Se connecter</a>`;
        
        // Re-attach listeners to dynamic links
        const link = document.getElementById(isLoginMode ? 'switch-to-signup' : 'switch-to-login');
        link.onclick = (e) => { e.preventDefault(); openModal(!isLoginMode); };
        
        authModal.classList.remove('hidden');
    }

    authForm.onsubmit = async (e) => {
        e.preventDefault();
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const endpoint = isLoginMode ? '/api/login' : '/api/signup';

        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await res.json();
            if (data.success) {
                setLoggedIn(data.username);
                authModal.classList.add('hidden');
                authForm.reset();
            } else {
                alert(data.error || "Erreur d'authentification");
            }
        } catch (err) { alert("Erreur serveur"); }
    };

    logoutBtn.onclick = async () => {
        await fetch('/api/logout', { method: 'POST' });
        setLoggedOut();
    };

    // --- Prediction Handlers ---
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        analyzeBtn.disabled = !this.value.trim();
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleAnalyze();
        }
    });

    analyzeBtn.addEventListener('click', handleAnalyze);

    async function handleAnalyze() {
        const text = input.value.trim();
        if (!text) return;

        if (welcomeSection) welcomeSection.style.display = 'none';
        addMessage(text, 'user');
        
        input.value = '';
        input.style.height = 'auto';
        analyzeBtn.disabled = true;
        loadingOverlay.classList.remove('hidden');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            const data = await response.json();
            if (response.ok) {
                addResultMessage(data);
            } else {
                addMessage(`Erreur: ${data.error}`, 'bot', true);
            }
        } catch (error) {
            addMessage(`Erreur de connexion serveur.`, 'bot', true);
        } finally {
            loadingOverlay.classList.add('hidden');
            scrollToBottom();
        }
    }

    function addMessage(text, side, isError = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${side}`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        if (isError) contentDiv.style.color = '#f85149';
        contentDiv.textContent = text;
        msgDiv.appendChild(contentDiv);
        historyContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    function addResultMessage(data) {
        const isTrue = data.is_true;
        const confidencePct = (data.confidence * 100).toFixed(1);
        const msgDiv = document.createElement('div');
        msgDiv.className = `message bot`;
        const cardDiv = document.createElement('div');
        cardDiv.className = `content result-card ${isTrue ? 'true' : 'fake'}`;
        cardDiv.innerHTML = `
            <div class="result-header">
                <span class="status-label">${data.prediction}</span>
            </div>
            <div class="confidence-box">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-secondary);">
                    <span>Confiance</span>
                    <span>${confidencePct}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: 0%"></div>
                </div>
            </div>
        `;
        msgDiv.appendChild(cardDiv);
        historyContainer.appendChild(msgDiv);
        setTimeout(() => {
            cardDiv.querySelector('.confidence-bar-fill').style.width = `${confidencePct}%`;
        }, 100);
        scrollToBottom();
    }

    function scrollToBottom() {
        historyContainer.scrollTop = historyContainer.scrollHeight;
    }
});
