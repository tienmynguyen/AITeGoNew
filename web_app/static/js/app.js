/**
 * AI Sign Language Recognition & Gemini LLM Sentence Builder Frontend Application
 * Hỗ trợ Dual-Camera: Browser Webcam (HTML5 getUserMedia) & Server Webcam (OpenCV MJPEG)
 */

// Application State
const state = {
    ws: null,
    isConnected: false,
    cameraSource: "browser", // "browser" hoặc "server"
    cameraRunning: false,
    browserStream: null,
    sendFrameInterval: null,
    deviceId: 0,
    activeBranch: "AUTO",
    fps: 0,
    fpsCalcTime: Date.now(),
    frameCount: 0,
    confidenceThreshold: 0.68,
    cooldown: 1.0,
    holdDuration: (function() {
        const saved = parseFloat(localStorage.getItem("vsl_hold_duration"));
        return (!isNaN(saved) && saved >= 0.2) ? saved : 1.0;
    })(),
    drawLandmarks: true,
    wordQueue: [],
    history: [],
    geminiApiKey: localStorage.getItem("vsl_gemini_api_key") || "",
    geminiModel: (function() {
        let m = localStorage.getItem("vsl_gemini_model") || "gemini-3.5-flash";
        if (["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-3.8-pro"].includes(m)) {
            m = "gemini-3.5-flash";
            localStorage.setItem("vsl_gemini_model", m);
        }
        return m;
    })(),
    sentenceStyle: "natural",
    autoSentence: true,
    autoSentenceTimer: null,
    labels: {
        one_hand: [],
        two_hands: []
    }
};

// DOM Elements
const DOM = {
    // Header & Badges
    cameraStatusBadge: document.getElementById("cameraStatusBadge"),
    cameraStatusText: document.getElementById("cameraStatusText"),
    branchStatusBadge: document.getElementById("branchStatusBadge"),
    branchStatusText: document.getElementById("branchStatusText"),
    geminiStatusBadge: document.getElementById("geminiStatusBadge"),
    geminiStatusText: document.getElementById("geminiStatusText"),
    fpsCounter: document.getElementById("fpsCounter"),

    // Camera & Stream Controls
    tabBrowserCam: document.getElementById("tabBrowserCam"),
    tabServerCam: document.getElementById("tabServerCam"),
    btnToggleCam: document.getElementById("btnToggleCam"),
    iconToggleCam: document.getElementById("iconToggleCam"),
    textToggleCam: document.getElementById("textToggleCam"),
    btnStartCamBig: document.getElementById("btnStartCamBig"),

    // Video Elements
    browserVideo: document.getElementById("browserVideo"),
    overlayCanvas: document.getElementById("overlayCanvas"),
    serverVideoFeed: document.getElementById("serverVideoFeed"),
    camOffPlaceholder: document.getElementById("camOffPlaceholder"),
    
    // Live Overlay (Hold Progress 2s)
    livePredictionOverlay: document.getElementById("livePredictionOverlay"),
    predBranchTag: document.getElementById("predBranchTag"),
    predWordText: document.getElementById("predWordText"),
    predConfLabel: document.getElementById("predConfLabel"),
    predHoldStatus: document.getElementById("predHoldStatus"),
    predHoldBar: document.getElementById("predHoldBar"),

    // Branch buttons
    btnBranchAuto: document.getElementById("btnBranchAuto"),
    btnBranchOne: document.getElementById("btnBranchOne"),
    btnBranchTwo: document.getElementById("btnBranchTwo"),

    // Settings
    sliderHoldDuration: document.getElementById("sliderHoldDuration"),
    valHoldDuration: document.getElementById("valHoldDuration"),
    sliderConfThreshold: document.getElementById("sliderConfThreshold"),
    valConfThreshold: document.getElementById("valConfThreshold"),
    sliderCooldown: document.getElementById("sliderCooldown"),
    valCooldown: document.getElementById("valCooldown"),
    toggleDrawLandmarks: document.getElementById("toggleDrawLandmarks"),

    // Dictionary
    dictToggleHeader: document.getElementById("dictToggleHeader"),
    dictBody: document.getElementById("dictBody"),
    dictArrow: document.getElementById("dictArrow"),
    chipsOneHand: document.getElementById("chipsOneHand"),
    chipsTwoHands: document.getElementById("chipsTwoHands"),

    // Word Queue
    wordQueueContainer: document.getElementById("wordQueueContainer"),
    emptyQueueNotice: document.getElementById("emptyQueueNotice"),
    queueChips: document.getElementById("queueChips"),
    queueCountBadge: document.getElementById("queueCountBadge"),
    btnUndoWord: document.getElementById("btnUndoWord"),
    btnClearQueue: document.getElementById("btnClearQueue"),
    manualWordInput: document.getElementById("manualWordInput"),
    btnAddManualWord: document.getElementById("btnAddManualWord"),

    // Gemini
    btnToggleSettings: document.getElementById("btnToggleSettings"),
    geminiSettingsPanel: document.getElementById("geminiSettingsPanel"),
    geminiApiKeyInput: document.getElementById("geminiApiKeyInput"),
    btnToggleKeyMask: document.getElementById("btnToggleKeyMask"),
    btnSaveKey: document.getElementById("btnSaveKey"),
    btnTestKey: document.getElementById("btnTestKey"),
    apiKeyNotice: document.getElementById("apiKeyNotice"),
    geminiModelSelect: document.getElementById("geminiModelSelect"),
    sentenceStyleSelect: document.getElementById("sentenceStyleSelect"),
    customPromptInput: document.getElementById("customPromptInput"),
    toggleAutoSentence: document.getElementById("toggleAutoSentence"),
    btnGenerateSentence: document.getElementById("btnGenerateSentence"),
    btnGenerateText: document.getElementById("btnGenerateText"),

    // Sentence Output
    sentenceDisplayText: document.getElementById("sentenceDisplayText"),
    sentenceDetails: document.getElementById("sentenceDetails"),
    englishTranslationText: document.getElementById("englishTranslationText"),
    explanationText: document.getElementById("explanationText"),
    btnSpeakSentence: document.getElementById("btnSpeakSentence"),
    btnCopySentence: document.getElementById("btnCopySentence"),

    // History
    historyList: document.getElementById("historyList"),
    emptyHistoryNotice: document.getElementById("emptyHistoryNotice"),
    btnClearHistory: document.getElementById("btnClearHistory"),
};

// Canvas offscreen để trích xuất frame từ Video element
const offscreenCanvas = document.createElement("canvas");
offscreenCanvas.width = 640;
offscreenCanvas.height = 480;
const offscreenCtx = offscreenCanvas.getContext("2d");

// -------------------------------------------------------------
// WEBSOCKET CONNECTION
// -------------------------------------------------------------
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log("[WS] Đã kết nối với Web Server.");
        state.isConnected = true;
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWsMessage(data);
        } catch (e) {
            console.error("[WS] Lỗi parse JSON:", e);
        }
    };

    state.ws.onclose = () => {
        console.warn("[WS] Mất kết nối. Đang thử kết nối lại sau 2s...");
        state.isConnected = false;
        setTimeout(initWebSocket, 2000);
    };

    state.ws.onerror = (err) => {
        console.error("[WS] Lỗi:", err);
    };
}

function sendWs(payload) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(payload));
    }
}

function handleWsMessage(data) {
    switch (data.type) {
        case "init":
            if (data.labels) {
                state.labels = data.labels;
                renderVocabularyDictionary();
            }
            if (data.gemini && data.gemini.has_key && !state.geminiApiKey) {
                updateGeminiBadge(true);
            }
            break;

        case "status":
            if (state.cameraSource === "server") {
                syncServerStatus(data.status);
            }
            break;

        case "word_recognized":
            onWordRecognized(data);
            break;

        case "frame_result":
            // Kết quả nhận diện trả về từ frame của Browser Webcam
            handleFrameResult(data);
            break;
    }
}

function handleFrameResult(result) {
    // Tính toán FPS hiển thị
    state.frameCount++;
    const now = Date.now();
    if (now - state.fpsCalcTime >= 1000) {
        state.fps = state.frameCount;
        state.frameCount = 0;
        state.fpsCalcTime = now;
        DOM.fpsCounter.textContent = state.fps;
    }

    // Cập nhật Nhánh hoạt động
    updateBranchBadge(result.active_branch, !!result.branch_override);

    // Vẽ Landmark lên Canvas overlay nếu được bật
    if (state.drawLandmarks && result.landmarks) {
        drawLandmarksOnCanvas(result.landmarks);
    } else {
        clearOverlayCanvas();
    }

    // Hiển thị Tiến trình duy trì cử chỉ (2s) hoặc kết quả nhận diện
    if (result.holding && result.holding.word && (result.holding.progress > 0 || result.holding.confirmed)) {
        showHoldProgress(result.holding, result.active_branch, result.confidence);
    } else if (result.word && result.confidence > 0) {
        showHoldProgress({
            word: result.word,
            branch: result.active_branch,
            seconds: 0,
            target: state.holdDuration,
            progress: 0,
            confirmed: false
        }, result.active_branch, result.confidence);
    } else {
        hideHoldProgress();
    }
}

function syncServerStatus(st) {
    DOM.fpsCounter.textContent = Math.round(st.fps);
    updateCameraBadge(st.is_running);
    const branchName = st.branch_override || st.active_branch || "AUTO";
    updateBranchBadge(branchName, !!st.branch_override);

    // Đồng bộ thanh tiến trình giữ cử chỉ (Hold Progress 2s) cho Server Camera
    if (state.cameraSource === "server") {
        if (st.holding && st.holding.word && (st.holding.progress > 0 || st.holding.confirmed)) {
            showHoldProgress(st.holding, st.active_branch, st.last_confidence || 0);
        } else if (st.last_word && st.last_confidence > 0) {
            showHoldProgress({
                word: st.last_word,
                branch: st.active_branch,
                seconds: 0,
                target: state.holdDuration,
                progress: 0,
                confirmed: false
            }, st.active_branch, st.last_confidence);
        } else {
            hideHoldProgress();
        }
    }
}

// -------------------------------------------------------------
// BROWSER WEBCAM (HTML5 getUserMedia)
// -------------------------------------------------------------
async function startBrowserCamera() {
    try {
        updateCameraBadgeLoading();

        // Yêu cầu quyền truy cập Camera từ Trình duyệt
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        state.browserStream = stream;
        DOM.browserVideo.srcObject = stream;
        await DOM.browserVideo.play();

        // Chỉnh kích thước overlay canvas theo video
        DOM.browserVideo.style.display = "block";
        DOM.overlayCanvas.style.display = "block";
        DOM.serverVideoFeed.style.display = "none";
        DOM.camOffPlaceholder.style.display = "none";

        fitCanvasToVideo();

        state.cameraRunning = true;
        updateCameraBadge(true);

        // Bắt đầu vòng lặp gửi frame nhị phân tới Server qua WebSocket (~22 FPS)
        if (state.sendFrameInterval) clearInterval(state.sendFrameInterval);
        state.sendFrameInterval = setInterval(captureAndSendFrame, 45);

        console.log("[CAMERA] Browser Webcam đã bật thành công!");
    } catch (err) {
        console.error("[CAMERA] Không thể truy cập Camera:", err);
        state.cameraRunning = false;
        updateCameraBadge(false);
        DOM.camOffPlaceholder.style.display = "flex";
        alert(`Không thể mở Camera: ${err.message}\n\nHãy đảm bảo bạn đã cấp quyền (Allow) cho trình duyệt sử dụng Camera và không có ứng dụng nào khác đang chiếm quyền camera!`);
    }
}

function stopBrowserCamera() {
    if (state.sendFrameInterval) {
        clearInterval(state.sendFrameInterval);
        state.sendFrameInterval = null;
    }
    if (state.browserStream) {
        state.browserStream.getTracks().forEach(track => track.stop());
        state.browserStream = null;
    }
    DOM.browserVideo.srcObject = null;
    DOM.browserVideo.style.display = "none";
    DOM.overlayCanvas.style.display = "none";
    DOM.camOffPlaceholder.style.display = "flex";

    state.cameraRunning = false;
    updateCameraBadge(false);
    clearOverlayCanvas();
}

function captureAndSendFrame() {
    if (!state.cameraRunning || !DOM.browserVideo.videoWidth) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

    // Chống tích lũy tràn bộ đệm WebSocket và tránh giật lag trình duyệt
    if (state.ws.bufferedAmount > 80000) {
        return;
    }

    try {
        // Vẽ frame từ video vào canvas ẩn
        offscreenCtx.drawImage(DOM.browserVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        
        // Chuyển frame thành Blob JPEG và gửi trực tiếp qua WebSocket
        offscreenCanvas.toBlob((blob) => {
            if (blob && state.ws && state.ws.readyState === WebSocket.OPEN) {
                if (state.ws.bufferedAmount <= 80000) {
                    state.ws.send(blob);
                }
            }
        }, "image/jpeg", 0.6);
    } catch (e) {
        // bỏ qua lỗi nếu frame tạm thời chưa sẵn sàng
    }
}

function fitCanvasToVideo() {
    const rect = DOM.browserVideo.getBoundingClientRect();
    DOM.overlayCanvas.width = DOM.browserVideo.videoWidth || 640;
    DOM.overlayCanvas.height = DOM.browserVideo.videoHeight || 480;
}

// -------------------------------------------------------------
// VẼ LANDMARK LÊN OVERLAY CANVAS
// -------------------------------------------------------------
function drawLandmarksOnCanvas(landmarks) {
    const ctx = DOM.overlayCanvas.getContext("2d");
    const w = DOM.overlayCanvas.width;
    const h = DOM.overlayCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Vẽ điểm Pose thân trên (Màu vàng kim / Cyan)
    if (landmarks.pose && landmarks.pose.length > 0) {
        ctx.fillStyle = "#38bdf8";
        landmarks.pose.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * w, pt.y * h, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    // Vẽ bàn tay trái (Màu Xanh lục tươi)
    if (landmarks.left_hand && landmarks.left_hand.length > 0) {
        ctx.fillStyle = "#10b981";
        landmarks.left_hand.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * w, pt.y * h, 4, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    // Vẽ bàn tay phải (Màu Tím / Hồng)
    if (landmarks.right_hand && landmarks.right_hand.length > 0) {
        ctx.fillStyle = "#c084fc";
        landmarks.right_hand.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * w, pt.y * h, 4, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
}

function clearOverlayCanvas() {
    const ctx = DOM.overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, DOM.overlayCanvas.width, DOM.overlayCanvas.height);
}

// -------------------------------------------------------------
// SERVER OPENCV CAMERA STREAM
// -------------------------------------------------------------
function startServerCamera() {
    DOM.browserVideo.style.display = "none";
    DOM.overlayCanvas.style.display = "none";
    DOM.camOffPlaceholder.style.display = "none";
    DOM.serverVideoFeed.style.display = "block";

    // Reload stream img src
    DOM.serverVideoFeed.src = `/video_feed?t=${Date.now()}`;
    sendWs({ action: "toggle_camera", state: true, device_id: state.deviceId });

    state.cameraRunning = true;
    updateCameraBadge(true);
}

function stopServerCamera() {
    sendWs({ action: "toggle_camera", state: false });
    DOM.serverVideoFeed.style.display = "none";
    DOM.camOffPlaceholder.style.display = "flex";

    state.cameraRunning = false;
    updateCameraBadge(false);
}

// -------------------------------------------------------------
// WORD RECOGNITION & QUEUE (DUY TRÌ 2S & TỰ ĐỘNG GHÉP CÂU)
// -------------------------------------------------------------
function onWordRecognized(event) {
    const { word, confidence, branch } = event;

    // Hiển thị Overlay trên video với thông báo xác nhận đã chốt từ
    if (DOM.predWordText) DOM.predWordText.textContent = word;
    const branchName = branch === "ONE_HAND" ? "1 TAY" : (branch === "TWO_HANDS" ? "2 TAY" : "TAY");
    if (DOM.predBranchTag) {
        DOM.predBranchTag.textContent = branchName;
        DOM.predBranchTag.style.background = branch === "ONE_HAND" ? "#3b82f6" : "#8b5cf6";
    }
    if (DOM.predHoldBar) DOM.predHoldBar.style.width = "100%";
    if (DOM.predHoldStatus) DOM.predHoldStatus.textContent = `✅ Đã thêm "${word}" vào hàng đợi!`;

    if (DOM.livePredictionOverlay) {
        DOM.livePredictionOverlay.classList.add("hold-confirmed");
        DOM.livePredictionOverlay.classList.remove("hidden");

        setTimeout(() => {
            if (DOM.livePredictionOverlay) {
                DOM.livePredictionOverlay.classList.remove("hold-confirmed");
            }
        }, 1200);
    }

    // Thêm vào Hàng đợi từ vựng (tự động kích hoạt ghép câu)
    addWordToQueue(word, branch, confidence);
}

let overlayHideTimeout = null;

function showHoldProgress(holding, activeBranch, conf = 0) {
    if (!DOM.livePredictionOverlay) return;

    const word = holding.word || "";
    if (!word) {
        hideHoldProgress();
        return;
    }

    DOM.predWordText.textContent = word;
    const branchName = holding.branch || activeBranch;
    DOM.predBranchTag.textContent = branchName === "ONE_HAND" ? "1 TAY" : (branchName === "TWO_HANDS" ? "2 TAY" : "TAY");
    DOM.predBranchTag.style.background = branchName === "ONE_HAND" ? "#3b82f6" : "#8b5cf6";

    const pct = Math.round((holding.progress || 0) * 100);
    if (DOM.predHoldBar) {
        DOM.predHoldBar.style.width = `${pct}%`;
    }

    if (DOM.predConfLabel && conf > 0) {
        DOM.predConfLabel.textContent = `${Math.round(conf * 100)}%`;
        DOM.predConfLabel.style.display = "inline-block";
    }

    if (holding.confirmed) {
        if (DOM.predHoldStatus) {
            DOM.predHoldStatus.textContent = `✅ Đã chốt "${word}"!`;
        }
        DOM.livePredictionOverlay.classList.add("hold-confirmed");
    } else {
        DOM.livePredictionOverlay.classList.remove("hold-confirmed");
        if (DOM.predHoldStatus) {
            const secs = typeof holding.seconds === 'number' ? holding.seconds.toFixed(1) : "0.0";
            const tgt = typeof holding.target === 'number' ? holding.target.toFixed(1) : state.holdDuration.toFixed(1);
            DOM.predHoldStatus.textContent = `⏳ Giữ cử chỉ: ${secs}s / ${tgt}s (${pct}%)`;
        }
    }

    DOM.livePredictionOverlay.classList.remove("hidden");

    if (overlayHideTimeout) clearTimeout(overlayHideTimeout);
    overlayHideTimeout = setTimeout(() => {
        if (DOM.livePredictionOverlay && !DOM.livePredictionOverlay.classList.contains("hold-confirmed")) {
            DOM.livePredictionOverlay.classList.add("hidden");
        }
    }, 2000);
}

function hideHoldProgress() {
    if (DOM.livePredictionOverlay && !DOM.livePredictionOverlay.classList.contains("hold-confirmed")) {
        DOM.livePredictionOverlay.classList.add("hidden");
    }
}

// Bộ đếm Debounce tự động ghép câu sau mỗi lần thêm từ hoặc chỉnh sửa
let autoSentenceDebounceTimer = null;

function triggerAutoSentence(delay = 400) {
    if (autoSentenceDebounceTimer) {
        clearTimeout(autoSentenceDebounceTimer);
        autoSentenceDebounceTimer = null;
    }

    if (!state.autoSentence) return;

    if (state.wordQueue.length === 0) {
        resetSentenceDisplay();
        return;
    }

    autoSentenceDebounceTimer = setTimeout(() => {
        generateSentence();
    }, delay);
}

function resetSentenceDisplay() {
    DOM.sentenceDisplayText.innerHTML = `<em>Chưa có câu nào được tạo. Hãy làm cử chỉ trước camera để bắt đầu!</em>`;
    DOM.sentenceDetails.style.display = "none";
}

function addWordToQueue(word, branch = "MANUAL", conf = 1.0) {
    state.wordQueue.push({
        id: Date.now() + Math.random(),
        word: word,
        branch: branch,
        conf: conf
    });
    renderWordQueue();
    triggerAutoSentence(350);
}

function removeWordFromQueue(id) {
    state.wordQueue = state.wordQueue.filter(item => item.id !== id);
    renderWordQueue();
    triggerAutoSentence(250);
}

function undoLastWord() {
    if (state.wordQueue.length > 0) {
        state.wordQueue.pop();
        renderWordQueue();
        triggerAutoSentence(250);
    }
}

function clearQueue() {
    state.wordQueue = [];
    renderWordQueue();
    resetSentenceDisplay();
}

function renderWordQueue() {
    DOM.queueCountBadge.textContent = `${state.wordQueue.length} từ`;

    if (state.wordQueue.length === 0) {
        DOM.emptyQueueNotice.style.display = "flex";
        DOM.queueChips.innerHTML = "";
        return;
    }

    DOM.emptyQueueNotice.style.display = "none";
    DOM.queueChips.innerHTML = state.wordQueue.map((item) => `
        <div class="word-chip" data-id="${item.id}">
            <span>${escapeHtml(item.word)}</span>
            <span class="word-chip-badge">${item.branch === "ONE_HAND" ? "1T" : (item.branch === "TWO_HANDS" ? "2T" : "Thủ công")}</span>
            <span class="chip-delete" onclick="removeWordFromQueue(${item.id})">&times;</span>
        </div>
    `).join("");
}

// -------------------------------------------------------------
// VOCABULARY DICTIONARY (38 Labels)
// -------------------------------------------------------------
function renderVocabularyDictionary() {
    // 1 Hand Labels
    DOM.chipsOneHand.innerHTML = state.labels.one_hand.map(word => `
        <button class="vocab-chip" onclick="addWordToQueue('${escapeHtml(word)}', 'ONE_HAND', 1.0)">
            ${escapeHtml(word)}
        </button>
    `).join("");

    // 2 Hands Labels
    DOM.chipsTwoHands.innerHTML = state.labels.two_hands.map(word => `
        <button class="vocab-chip" onclick="addWordToQueue('${escapeHtml(word)}', 'TWO_HANDS', 1.0)">
            ${escapeHtml(word)}
        </button>
    `).join("");
}

// -------------------------------------------------------------
// GEMINI LLM SENTENCE RECONSTRUCTION (NON-BLOCKING ASYNC)
// -------------------------------------------------------------
let currentGenerateAbortController = null;

async function generateSentence(isAuto = false) {
    if (state.wordQueue.length === 0) {
        if (!isAuto) {
            alert("Hàng đợi từ vựng đang rỗng! Hãy làm cử chỉ trước camera hoặc click vào từ trong Từ điển.");
        }
        return;
    }

    // Nếu đang có một yêu cầu ghép câu trước đó đang chạy, hủy ngay để không lag và không dồn ứ API
    if (currentGenerateAbortController) {
        try {
            currentGenerateAbortController.abort();
        } catch (_) {}
    }
    currentGenerateAbortController = new AbortController();
    const signal = currentGenerateAbortController.signal;

    const words = state.wordQueue.map(item => item.word);
    const style = DOM.sentenceStyleSelect.value;
    let model = DOM.geminiModelSelect.value;
    if (["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-3.8-pro"].includes(model)) {
        model = "gemini-3.5-flash";
        DOM.geminiModelSelect.value = "gemini-3.5-flash";
        state.geminiModel = "gemini-3.5-flash";
        localStorage.setItem("vsl_gemini_model", "gemini-3.5-flash");
    }
    const customPrompt = DOM.customPromptInput.value.trim();
    const apiKey = state.geminiApiKey;

    // UI Loading state - CHỈ disable nút bấm khi người dùng nhấn thủ công, KHÔNG khóa UI khi tự động!
    if (!isAuto) {
        DOM.btnGenerateSentence.disabled = true;
        DOM.btnGenerateText.textContent = "Đang ghép câu với Gemini AI...";
        DOM.sentenceDisplayText.innerHTML = `<em>✨ Đang phân tích ngữ cảnh và ghép câu từ ${words.length} từ...</em>`;
    } else {
        DOM.sentenceDisplayText.classList.add("sentence-loading-pulse");
    }

    try {
        const payload = {
            words: words,
            style: style,
            custom_prompt: customPrompt,
            api_key: apiKey,
            model: model
        };

        let data = null;

        // 1. Thử gọi backend server (chạy non-blocking trên thread pool)
        try {
            const response = await fetch("/api/llm/combine", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal: signal
            });
            if (response.ok) {
                data = await response.json();
            }
        } catch (serverErr) {
            if (serverErr.name === "AbortError") return;
            console.warn("Backend API error, falling back to direct browser Gemini call:", serverErr);
        }

        if (signal.aborted) return;

        // 2. Nếu backend offline hoặc lỗi mạng, gọi trực tiếp từ trình duyệt
        if (!data || (!data.success && data.is_fallback && apiKey)) {
            data = await callGeminiDirectFromBrowser(words, style, customPrompt, apiKey, model, signal);
        }

        if (signal.aborted) return;

        if (data && data.sentence) {
            if (data.model) {
                state.geminiModel = data.model;
                localStorage.setItem("vsl_gemini_model", data.model);
                if (DOM.geminiModelSelect && DOM.geminiModelSelect.value !== data.model) {
                    DOM.geminiModelSelect.value = data.model;
                }
            }
            displaySentenceResult(data);
            addToHistory(data.sentence, words, data.english_translation);
            if (!isAuto) {
                speakVietnamese(data.sentence);
            }
        } else {
            DOM.sentenceDisplayText.textContent = data ? (data.sentence || "Không tạo được câu.") : "Lỗi kết nối.";
        }

    } catch (err) {
        if (err.name === "AbortError") return;
        console.error("Lỗi khi ghép câu:", err);
        DOM.sentenceDisplayText.innerHTML = `<span style="color: var(--accent-rose);">Lỗi: ${escapeHtml(err.message)}</span>`;
    } finally {
        if (!isAuto) {
            DOM.btnGenerateSentence.disabled = false;
            DOM.btnGenerateText.textContent = "Ghép thành câu hoàn chỉnh";
        }
        DOM.sentenceDisplayText.classList.remove("sentence-loading-pulse");
    }
}

/**
 * Tiền xử lý chuỗi từ ký hiệu thô:
 * 1. Gộp từ lặp liên tiếp.
 * 2. Đổi 'có / đúng' -> 'có' hoặc 'đúng'.
 * 3. Ghép số chuỗi liên tiếp (ví dụ: '1', '00', '000' -> '100.000').
 */
function preprocessVslTokens(words) {
    if (!words || !words.length) return [];

    const deduped = [];
    for (const w of words) {
        const clean = String(w).trim();
        if (!clean) continue;
        if (!deduped.length || deduped[deduped.length - 1] !== clean) {
            deduped.push(clean);
        }
    }

    const merged = [];
    let i = 0;
    while (i < deduped.length) {
        const token = deduped[i];
        if (/^\d+$/.test(token)) {
            const numTokens = [token];
            let j = i + 1;
            while (j < deduped.length && /^\d+$/.test(deduped[j])) {
                numTokens.push(deduped[j]);
                j++;
            }
            if (numTokens.length > 1) {
                const fullNumStr = numTokens.join("");
                const val = parseInt(fullNumStr, 10);
                if (!isNaN(val)) {
                    merged.push(val.toLocaleString("vi-VN"));
                } else {
                    merged.push(fullNumStr);
                }
                i = j;
                continue;
            } else {
                merged.push(token);
            }
        } else {
            if (token === "có / đúng") {
                merged.push("có");
            } else {
                merged.push(token);
            }
        }
        i++;
    }
    return merged;
}

/**
 * Fallback trực tiếp gọi Gemini REST API từ trình duyệt
 */
async function callGeminiDirectFromBrowser(words, style, customPrompt, apiKey, model, signal = null) {
    const cleanTokens = preprocessVslTokens(words);

    if (!apiKey) {
        let fallbackText = cleanTokens.join(" ");
        fallbackText = fallbackText.charAt(0).toUpperCase() + fallbackText.slice(1);
        const isQuestion = ["cái gì", "ở đâu", "khi nào", "ai", "tại sao", "bao nhiêu", "không"].some(q => fallbackText.toLowerCase().includes(q));
        if (!/[.?!]$/.test(fallbackText)) {
            fallbackText += isQuestion ? "?" : ".";
        }
        return {
            success: false,
            sentence: fallbackText,
            original_words: words,
            cleaned_words: cleanTokens
        };
    }

    const systemPrompt = `Bạn là một chuyên gia ngôn ngữ học và phiên dịch viên Ngôn ngữ Ký hiệu Việt Nam (VSL - Vietnamese Sign Language) sang Tiếng Việt giao tiếp tự nhiên hàng đầu.

Bối cảnh:
Các từ nhận diện được trích xuất từ cử chỉ tay của người khiếm thính. Người sử dụng ký hiệu chỉ biểu thị các từ khóa cốt lõi (Keywords) và lược bỏ hầu hết ngữ pháp, hư từ, liên từ hoặc trợ từ.

Nguyên tắc BẮT BUỘC khi tạo câu:
1. XỬ LÝ NHÃN KÉP "có / đúng":
   - Tự động chọn từ "có" hoặc "đúng" phù hợp nhất với ngữ cảnh câu (ví dụ trong "bạn có / đúng cái gì" -> chọn "có").
   - TUYỆT ĐỐI KHÔNG để nguyên dạng "có / đúng" trong câu!
2. GHÉP CÁC CHỮ SỐ KÝ HIỆU (NUMBER CONCATENATION):
   - Chuỗi số liên tiếp như ["1", "00", "000"] phải được ghép thành "100.000" (hoặc "100.000 đồng" khi nói về giá cả).
3. VĂN PHONG TỰ NHIÊN, KHÔNG DỊCH THÔ CỨNG:
   - Viết thành câu nói chuyện đời thường mượt mà, đúng ngữ pháp tiếng Việt.
   - Thêm các từ nối, hư từ, trợ từ phù hợp (thì, là, mà, ở, với, tiền, đồng, món, ạ, nhé, vậy, thế, được không...).
   - Nếu câu có từ để hỏi (cái gì, ở đâu, khi nào, ai, tại sao, bao nhiêu, không) -> tạo câu hỏi tự nhiên với dấu chấm hỏi "?".
   - Lọc bỏ các từ lặp lại vô nghĩa.

CÁC VÍ DỤ MẪU (FEW-SHOT EXAMPLES):
- Từ nhận diện: ["xin chào", "bạn", "có / đúng", "cái gì", "giá", "1", "00", "000", "không"]
  -> Câu hoàn chỉnh: "Xin chào, bạn có cái gì giá 100.000 không?"
- Từ nhận diện: ["tên", "bạn", "cái gì"]
  -> Câu hoàn chỉnh: "Bạn tên là gì vậy?"
- Từ nhận diện: ["giá", "cái gì", "bao nhiêu"]
  -> Câu hoàn chỉnh: "Cái này giá bao nhiêu tiền vậy?"
- Từ nhận diện: ["giá", "2", "00", "000"]
  -> Câu hoàn chỉnh: "Món này giá 200.000 đồng."
- Từ nhận diện: ["cảm ơn", "bạn", "rất", "tốt"]
  -> Câu hoàn chỉnh: "Cảm ơn bạn nhiều, bạn tốt bụng quá!"
- Từ nhận diện: ["ở đâu", "giúp", "chúng tôi", "làm ơn"]
  -> Câu hoàn chỉnh: "Có ai ở đây có thể làm ơn giúp chúng tôi không?"

Từ khóa ký hiệu nhận diện: ${JSON.stringify(cleanTokens)}
(Từ gốc: ${JSON.stringify(words)})
Phong cách yêu cầu: ${style}
${customPrompt ? "Yêu cầu bổ sung: " + customPrompt : ""}

CHỈ TRẢ VỀ JSON:
{
  "sentence": "<câu hoàn chỉnh tiếng Việt tự nhiên>",
  "english_translation": "<bản dịch tiếng Anh tương đương>",
  "explanation": "<giải thích ngắn gọn>"
}`;

    let requestedModel = (model || "gemini-3.8-flash").replace(/^models\//, "");
    if (["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-exp"].includes(requestedModel)) {
        requestedModel = "gemini-3.8-flash";
    }

    const candidateModels = [
        requestedModel,
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-1.5-flash",
        "gemini-3.8-pro"
    ].filter(m => !["gemini-2.5-flash", "gemini-2.0-flash"].includes(m));
    const uniqueCandidates = [...new Set(candidateModels)];

    let lastError = null;

    for (const currentModel of uniqueCandidates) {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${currentModel}:generateContent?key=${apiKey}`;
        try {
            const fetchOptions = {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    contents: [{ parts: [{ text: systemPrompt }] }],
                    generationConfig: {
                        temperature: 0.3,
                        maxOutputTokens: 400,
                        responseMimeType: "application/json"
                    }
                })
            };
            if (signal) fetchOptions.signal = signal;

            const res = await fetch(url, fetchOptions);

            if (res.ok) {
                const resData = await res.json();
                const rawText = resData.candidates?.[0]?.content?.parts?.[0]?.text || "";
                try {
                    const parsed = JSON.parse(rawText);
                    return {
                        success: true,
                        sentence: parsed.sentence,
                        english_translation: parsed.english_translation,
                        explanation: parsed.explanation || (currentModel !== requestedModel ? `Đã tự động chuyển sang ${currentModel} do ${requestedModel} đang quá tải` : ""),
                        original_words: words,
                        cleaned_words: cleanTokens,
                        model: currentModel
                    };
                } catch {
                    return {
                        success: true,
                        sentence: rawText.trim(),
                        original_words: words,
                        cleaned_words: cleanTokens,
                        model: currentModel
                    };
                }
            } else {
                const errJson = await res.json().catch(() => ({}));
                const errMsg = errJson.error ? errJson.error.message : `HTTP ${res.status}`;
                lastError = new Error(errMsg);

                const isRetryable = (
                    res.status === 400 ||
                    res.status === 404 ||
                    res.status === 429 ||
                    res.status === 503 ||
                    errMsg.toLowerCase().includes("no longer available") ||
                    errMsg.toLowerCase().includes("update your code to use") ||
                    errMsg.toLowerCase().includes("high demand") ||
                    errMsg.toLowerCase().includes("spikes in demand") ||
                    errMsg.toLowerCase().includes("temporarily unavailable") ||
                    errMsg.toLowerCase().includes("not found")
                );

                if (isRetryable) {
                    console.warn(`[GEMINI] Model '${currentModel}' không khả dụng (${errMsg}). Tự động thử model dự phòng tiếp theo...`);
                    continue;
                } else {
                    throw lastError;
                }
            }
        } catch (fetchErr) {
            if (fetchErr.name === "AbortError") throw fetchErr;
            lastError = fetchErr;
            const errMsg = fetchErr.message || "";
            if (errMsg.toLowerCase().includes("high demand") || errMsg.includes("503") || errMsg.includes("429") || errMsg.includes("404")) {
                continue;
            }
            throw fetchErr;
        }
    }

    throw lastError || new Error("Các mô hình Gemini hiện đang quá tải, vui lòng thử lại sau giây lát.");
}

function displaySentenceResult(data) {
    DOM.sentenceDisplayText.textContent = `"${data.sentence}"`;
    DOM.sentenceDisplayText.classList.remove("highlight-pulse");
    void DOM.sentenceDisplayText.offsetWidth; // trigger reflow
    DOM.sentenceDisplayText.classList.add("highlight-pulse");

    if (data.english_translation || data.explanation) {
        DOM.sentenceDetails.style.display = "flex";
        DOM.englishTranslationText.textContent = data.english_translation ? `🇬🇧 ${data.english_translation}` : "";
        DOM.explanationText.textContent = data.explanation ? `💡 ${data.explanation}` : "";
    } else {
        DOM.sentenceDetails.style.display = "none";
    }
}

// -------------------------------------------------------------
// TEXT TO SPEECH (TTS) & COPY
// -------------------------------------------------------------
function speakVietnamese(text) {
    if (!text || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    const cleanText = text.replace(/["“”]/g, "").trim();
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = "vi-VN";
    utterance.rate = 1.0;

    const voices = window.speechSynthesis.getVoices();
    const viVoice = voices.find(v => v.lang.startsWith("vi") || v.name.toLowerCase().includes("vietnam"));
    if (viVoice) {
        utterance.voice = viVoice;
    }

    window.speechSynthesis.speak(utterance);
}

function copySentence() {
    const text = DOM.sentenceDisplayText.textContent.replace(/["“”]/g, "").trim();
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        const orig = DOM.btnCopySentence.textContent;
        DOM.btnCopySentence.textContent = "✅ Đã sao chép!";
        setTimeout(() => {
            DOM.btnCopySentence.textContent = orig;
        }, 1500);
    });
}

// -------------------------------------------------------------
// TRANSLATION HISTORY
// -------------------------------------------------------------
function addToHistory(sentence, words, english = "") {
    const timeStr = new Date().toLocaleTimeString("vi-VN", { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const item = {
        time: timeStr,
        sentence: sentence,
        words: words.join(" → "),
        english: english
    };

    state.history.unshift(item);
    renderHistory();
}

function renderHistory() {
    if (state.history.length === 0) {
        DOM.emptyHistoryNotice.style.display = "block";
        DOM.historyList.innerHTML = "";
        return;
    }

    DOM.emptyHistoryNotice.style.display = "none";
    DOM.historyList.innerHTML = state.history.map(item => `
        <div class="history-item">
            <span class="history-time">🕒 ${item.time}</span>
            <div class="history-sentence">"${escapeHtml(item.sentence)}"</div>
            ${item.english ? `<small style="color:#93c5fd;">🇬🇧 ${escapeHtml(item.english)}</small>` : ""}
            <div class="history-words">Từ gốc: [ ${escapeHtml(item.words)} ]</div>
        </div>
    `).join("");
}

function clearHistory() {
    state.history = [];
    renderHistory();
}

// -------------------------------------------------------------
// UI CONTROLS & EVENT LISTENERS
// -------------------------------------------------------------
function initEventListeners() {
    // Chuyển đổi giữa Browser Webcam và Server Webcam
    DOM.tabBrowserCam.addEventListener("click", () => {
        if (state.cameraSource === "browser") return;
        DOM.tabBrowserCam.classList.add("active");
        DOM.tabServerCam.classList.remove("active");
        state.cameraSource = "browser";

        if (state.cameraRunning) {
            stopServerCamera();
            startBrowserCamera();
        }
    });

    DOM.tabServerCam.addEventListener("click", () => {
        if (state.cameraSource === "server") return;
        DOM.tabServerCam.classList.add("active");
        DOM.tabBrowserCam.classList.remove("active");
        state.cameraSource = "server";

        if (state.cameraRunning) {
            stopBrowserCamera();
            startServerCamera();
        }
    });

    // Nút Bật/Tắt Camera
    const toggleCameraHandler = () => {
        if (state.cameraRunning) {
            if (state.cameraSource === "browser") stopBrowserCamera();
            else stopServerCamera();
        } else {
            if (state.cameraSource === "browser") startBrowserCamera();
            else startServerCamera();
        }
    };

    DOM.btnToggleCam.addEventListener("click", toggleCameraHandler);
    DOM.btnStartCamBig.addEventListener("click", () => {
        state.cameraSource = "browser";
        DOM.tabBrowserCam.classList.add("active");
        DOM.tabServerCam.classList.remove("active");
        startBrowserCamera();
    });

    // Segmented Branch buttons
    [DOM.btnBranchAuto, DOM.btnBranchOne, DOM.btnBranchTwo].forEach(btn => {
        if (!btn) return;
        btn.addEventListener("click", () => {
            [DOM.btnBranchAuto, DOM.btnBranchOne, DOM.btnBranchTwo].forEach(b => b && b.classList.remove("active"));
            btn.classList.add("active");
            const mode = btn.dataset.branch;
            state.activeBranch = mode;
            sendWs({ action: "set_branch", mode: mode });
        });
    });

    // Modal Cài đặt & Tùy chọn (Popup Settings Modal)
    const settingsModal = document.getElementById("settingsModal");
    const btnOpenSettings = document.getElementById("btnOpenSettings");
    const btnCloseModal = document.getElementById("btnCloseModal");
    const btnDoneSettings = document.getElementById("btnDoneSettings");

    function openSettingsModal(defaultTab = null) {
        if (!settingsModal) return;
        settingsModal.classList.remove("hidden");
        if (defaultTab) {
            switchModalTab(defaultTab);
        }
    }

    function closeSettingsModal() {
        if (!settingsModal) return;
        settingsModal.classList.add("hidden");
    }

    function switchModalTab(tabId) {
        const tabBtns = document.querySelectorAll(".modal-tab-btn");
        const tabContents = document.querySelectorAll(".modal-tab-content");

        tabBtns.forEach(btn => {
            if (btn.dataset.tab === tabId) btn.classList.add("active");
            else btn.classList.remove("active");
        });

        tabContents.forEach(content => {
            if (content.id === tabId) content.classList.add("active");
            else content.classList.remove("active");
        });
    }

    if (btnOpenSettings) btnOpenSettings.addEventListener("click", () => openSettingsModal());
    if (btnCloseModal) btnCloseModal.addEventListener("click", closeSettingsModal);
    if (btnDoneSettings) btnDoneSettings.addEventListener("click", closeSettingsModal);

    // Đóng modal khi bấm ra ngoài backdrop
    if (settingsModal) {
        settingsModal.addEventListener("click", (e) => {
            if (e.target === settingsModal) closeSettingsModal();
        });
    }

    // Đóng modal khi nhấn phím Escape
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && settingsModal && !settingsModal.classList.contains("hidden")) {
            closeSettingsModal();
        }
    });

    // Chuyển tab khi nhấp vào nút tab
    document.querySelectorAll(".modal-tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            switchModalTab(btn.dataset.tab);
        });
    });

    // Mở nhanh tab tương ứng khi click vào các badge trên Navbar
    if (DOM.geminiStatusBadge) {
        DOM.geminiStatusBadge.addEventListener("click", () => openSettingsModal("tabGemini"));
    }
    if (DOM.branchStatusBadge) {
        DOM.branchStatusBadge.addEventListener("click", () => openSettingsModal("tabVision"));
    }
    if (DOM.cameraStatusBadge) {
        DOM.cameraStatusBadge.addEventListener("click", () => openSettingsModal("tabVision"));
    }

    // Sliders
    if (DOM.sliderHoldDuration) {
        DOM.sliderHoldDuration.value = state.holdDuration;
        if (DOM.valHoldDuration) DOM.valHoldDuration.textContent = `${state.holdDuration.toFixed(1)}s`;
        DOM.sliderHoldDuration.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            DOM.valHoldDuration.textContent = `${val.toFixed(1)}s`;
            state.holdDuration = val;
            localStorage.setItem("vsl_hold_duration", val);
            sendWs({
                action: "set_settings",
                confidence_threshold: parseInt(DOM.sliderConfThreshold.value, 10) / 100.0,
                cooldown: parseFloat(DOM.sliderCooldown.value),
                hold_duration: val,
                draw_landmarks: DOM.toggleDrawLandmarks.checked
            });
        });
    }

    DOM.sliderConfThreshold.addEventListener("input", (e) => {
        const val = parseInt(e.target.value, 10);
        DOM.valConfThreshold.textContent = `${val}%`;
        state.confidenceThreshold = val / 100.0;
        sendWs({
            action: "set_settings",
            confidence_threshold: state.confidenceThreshold,
            cooldown: parseFloat(DOM.sliderCooldown.value),
            hold_duration: state.holdDuration,
            draw_landmarks: DOM.toggleDrawLandmarks.checked
        });
    });

    DOM.sliderCooldown.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        DOM.valCooldown.textContent = `${val.toFixed(1)}s`;
        state.cooldown = val;
        sendWs({
            action: "set_settings",
            confidence_threshold: parseInt(DOM.sliderConfThreshold.value, 10) / 100.0,
            cooldown: val,
            hold_duration: state.holdDuration,
            draw_landmarks: DOM.toggleDrawLandmarks.checked
        });
    });

    DOM.toggleDrawLandmarks.addEventListener("change", (e) => {
        state.drawLandmarks = e.target.checked;
        if (!state.drawLandmarks) clearOverlayCanvas();
        sendWs({
            action: "set_settings",
            confidence_threshold: parseInt(DOM.sliderConfThreshold.value, 10) / 100.0,
            cooldown: parseFloat(DOM.sliderCooldown.value),
            hold_duration: state.holdDuration,
            draw_landmarks: e.target.checked
        });
    });

    // Dictionary toggle (nếu có phần tử)
    if (DOM.dictToggleHeader) {
        DOM.dictToggleHeader.addEventListener("click", () => {
            if (DOM.dictBody) DOM.dictBody.classList.toggle("collapsed");
            if (DOM.dictArrow) DOM.dictArrow.textContent = DOM.dictBody && DOM.dictBody.classList.contains("collapsed") ? "►" : "▼";
        });
    }

    // Word queue buttons
    DOM.btnUndoWord.addEventListener("click", undoLastWord);
    DOM.btnClearQueue.addEventListener("click", clearQueue);

    DOM.btnAddManualWord.addEventListener("click", () => {
        const val = DOM.manualWordInput.value.trim();
        if (val) {
            addWordToQueue(val, "MANUAL", 1.0);
            DOM.manualWordInput.value = "";
        }
    });

    DOM.manualWordInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            DOM.btnAddManualWord.click();
        }
    });

    // Gemini Settings
    if (DOM.btnToggleSettings) {
        DOM.btnToggleSettings.addEventListener("click", () => openSettingsModal("tabGemini"));
    }

    DOM.btnToggleKeyMask.addEventListener("click", () => {
        DOM.geminiApiKeyInput.type = DOM.geminiApiKeyInput.type === "password" ? "text" : "password";
    });

    DOM.btnSaveKey.addEventListener("click", () => {
        const key = DOM.geminiApiKeyInput.value.trim();
        state.geminiApiKey = key;
        localStorage.setItem("vsl_gemini_api_key", key);
        updateGeminiBadge(!!key);
        showNotice(DOM.apiKeyNotice, "Đã lưu Gemini API Key thành công!", "success");
    });

    DOM.btnTestKey.addEventListener("click", async () => {
        const key = DOM.geminiApiKeyInput.value.trim() || state.geminiApiKey;
        const model = DOM.geminiModelSelect.value;
        if (!key) {
            showNotice(DOM.apiKeyNotice, "Vui lòng nhập API Key để kiểm tra!", "error");
            return;
        }

        DOM.btnTestKey.disabled = true;
        DOM.btnTestKey.textContent = "Đang kiểm tra...";

        try {
            const res = await fetch("/api/llm/test_key", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: key, model: model })
            });
            const data = await res.json();
            if (data.valid) {
                showNotice(DOM.apiKeyNotice, `✅ ${data.message}`, "success");
                state.geminiApiKey = key;
                localStorage.setItem("vsl_gemini_api_key", key);
                updateGeminiBadge(true);
            } else {
                showNotice(DOM.apiKeyNotice, `❌ ${data.message}`, "error");
            }
        } catch (e) {
            showNotice(DOM.apiKeyNotice, `❌ Không thể kết nối tới server: ${e.message}`, "error");
        } finally {
            DOM.btnTestKey.disabled = false;
            DOM.btnTestKey.textContent = "Kiểm tra";
        }
    });

    DOM.geminiModelSelect.addEventListener("change", (e) => {
        state.geminiModel = e.target.value;
        localStorage.setItem("vsl_gemini_model", state.geminiModel);
    });

    DOM.toggleAutoSentence.addEventListener("change", (e) => {
        state.autoSentence = e.target.checked;
        if (state.autoSentence && state.wordQueue.length > 0) {
            triggerAutoSentence(100);
        }
    });

    DOM.btnGenerateSentence.addEventListener("click", () => generateSentence(false));
    DOM.btnSpeakSentence.addEventListener("click", () => {
        const text = DOM.sentenceDisplayText.textContent;
        speakVietnamese(text);
    });
    DOM.btnCopySentence.addEventListener("click", copySentence);
    DOM.btnClearHistory.addEventListener("click", clearHistory);
}

// -------------------------------------------------------------
// HELPER FUNCTIONS
// -------------------------------------------------------------
function updateCameraBadge(isRunning) {
    if (isRunning) {
        DOM.cameraStatusBadge.className = "status-badge status-online";
        DOM.cameraStatusText.textContent = "Camera: Đang chạy";
        DOM.iconToggleCam.textContent = "⏸️";
        DOM.textToggleCam.textContent = "Tắt Camera";
        DOM.btnToggleCam.className = "btn btn-sm btn-outline";
    } else {
        DOM.cameraStatusBadge.className = "status-badge status-offline";
        DOM.cameraStatusText.textContent = "Camera: Đang tắt";
        DOM.iconToggleCam.textContent = "▶️";
        DOM.textToggleCam.textContent = "Bật Camera";
        DOM.btnToggleCam.className = "btn btn-sm btn-primary";
    }
}

function updateCameraBadgeLoading() {
    DOM.cameraStatusBadge.className = "status-badge status-offline";
    DOM.cameraStatusText.textContent = "Camera: Đang xin quyền...";
}

function updateBranchBadge(branch, isManual) {
    if (branch === "ONE_HAND") {
        DOM.branchStatusText.textContent = isManual ? "Ép: 1 Tay" : "Nhánh: 1 Tay";
    } else if (branch === "TWO_HANDS") {
        DOM.branchStatusText.textContent = isManual ? "Ép: 2 Tay" : "Nhánh: 2 Tay";
    } else {
        DOM.branchStatusText.textContent = "Chế độ: Tự động";
    }
}

function updateGeminiBadge(hasKey) {
    if (hasKey) {
        DOM.geminiStatusBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
        DOM.geminiStatusBadge.style.color = "#6ee7b7";
        DOM.geminiStatusText.textContent = "Gemini: Sẵn sàng";
    } else {
        DOM.geminiStatusBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
        DOM.geminiStatusBadge.style.color = "#fde68a";
        DOM.geminiStatusText.textContent = "Gemini: Chưa có Key";
    }
}

function showNotice(el, msg, type = "info") {
    el.textContent = msg;
    el.className = `notice-msg ${type}`;
    setTimeout(() => {
        el.textContent = "";
        el.className = "notice-msg";
    }, 5000);
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// -------------------------------------------------------------
// INITIALIZATION
// -------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
    if (state.geminiApiKey) {
        DOM.geminiApiKeyInput.value = state.geminiApiKey;
        updateGeminiBadge(true);
    }
    if (state.geminiModel) {
        DOM.geminiModelSelect.value = state.geminiModel;
    }

    fetch("/api/labels")
        .then(r => r.json())
        .then(data => {
            if (data.one_hand) state.labels.one_hand = data.one_hand;
            if (data.two_hands) state.labels.two_hands = data.two_hands;
            renderVocabularyDictionary();
        })
        .catch(e => console.warn("Labels fetch failed:", e));

    initEventListeners();
    initWebSocket();

    // Tự động thử bật Browser Camera sau khi tải xong trang web
    setTimeout(() => {
        startBrowserCamera();
    }, 500);
});
