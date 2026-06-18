(() => {
    const videoCanvas = document.getElementById('videoCanvas');
    const ctx = videoCanvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const connectionEl = document.getElementById('connectionStatus');
    const resetBtn = document.getElementById('resetBtn');
    const necklaceBtns = document.querySelectorAll('.necklace-btn');

    let ws = null;
    let video = null;
    let sending = false;
    let currentNecklace = 1;
    let frameInterval = null;
    const TARGET_FPS = 15;

    // --- WebSocket ---
    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/ws`);

        ws.onopen = () => {
            connectionEl.textContent = 'Connected';
            connectionEl.className = 'connected';
        };

        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === 'frame') {
                drawFrame(msg.data);
                if (msg.status) {
                    statusEl.textContent = msg.status;
                    statusEl.classList.add('visible');
                } else {
                    statusEl.classList.remove('visible');
                }
                sending = false;
            }
        };

        ws.onclose = () => {
            connectionEl.textContent = 'Disconnected - reconnecting...';
            connectionEl.className = 'error';
            setTimeout(connect, 2000);
        };

        ws.onerror = () => {
            connectionEl.textContent = 'Connection error';
            connectionEl.className = 'error';
        };
    }

    function drawFrame(b64) {
        const img = new Image();
        img.onload = () => {
            videoCanvas.width = img.width;
            videoCanvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/jpeg;base64,' + b64;
    }

    // --- Camera ---
    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: false,
            });
            video = document.createElement('video');
            video.srcObject = stream;
            video.playsInline = true;
            await video.play();

            // Draw initial mirror
            videoCanvas.width = video.videoWidth;
            videoCanvas.height = video.videoHeight;

            frameInterval = setInterval(captureAndSend, 1000 / TARGET_FPS);
        } catch (err) {
            console.error('Camera error:', err);
            statusEl.textContent = 'Camera access denied';
            statusEl.classList.add('visible');
        }
    }

    function captureAndSend() {
        if (!ws || ws.readyState !== WebSocket.OPEN || sending || !video) return;
        sending = true;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');

        // Mirror horizontally
        tempCtx.translate(tempCanvas.width, 0);
        tempCtx.scale(-1, 1);
        tempCtx.drawImage(video, 0, 0);
        tempCtx.setTransform(1, 0, 0, 1, 0, 0);

        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.8);
        const b64 = dataUrl.split(',')[1];

        ws.send(JSON.stringify({ type: 'frame', data: b64 }));
    }

    // --- Controls ---
    necklaceBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            necklaceBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentNecklace = parseInt(btn.dataset.id);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'select_necklace', id: currentNecklace }));
            }
        });
    });

    resetBtn.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'reset_calibration' }));
        }
    });

    // --- Init ---
    connect();
    startCamera();
})();
