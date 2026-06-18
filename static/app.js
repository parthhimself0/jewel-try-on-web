(() => {
    const videoCanvas = document.getElementById('videoCanvas');
    const ctx = videoCanvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const connectionEl = document.getElementById('connectionStatus');
    const resetBtn = document.getElementById('resetBtn');
    const clipCheckbox = document.getElementById('clipCheckbox');
    const debugMaskCheckbox = document.getElementById('debugMaskCheckbox');
    const debugSilhouetteCheckbox = document.getElementById('debugSilhouetteCheckbox');
    const jewelryOpenBtn = document.getElementById('jewelryOpenBtn');
    const sheetBackdrop = document.getElementById('sheetBackdrop');
    const sheet = document.getElementById('sheet');
    const sheetGrid = document.getElementById('sheetGrid');

    let ws = null;
    let video = null;
    let sending = false;
    let currentNecklace = null;
    let currentEarring = null;
    let activeCategory = 'necklaces';
    let frameInterval = null;
    const TARGET_FPS = 15;

    const jewelryData = { necklaces: [], earrings: [] };

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

        tempCtx.translate(tempCanvas.width, 0);
        tempCtx.scale(-1, 1);
        tempCtx.drawImage(video, 0, 0);
        tempCtx.setTransform(1, 0, 0, 1, 0, 0);

        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.8);
        const b64 = dataUrl.split(',')[1];

        ws.send(JSON.stringify({ type: 'frame', data: b64 }));
    }

    // --- Sheet Controls ---
    function openSheet() {
        sheetBackdrop.classList.add('open');
        sheet.classList.add('open');
    }

    function closeSheet() {
        sheetBackdrop.classList.remove('open');
        sheet.classList.remove('open');
    }

    function renderGrid(category) {
        activeCategory = category;
        sheetGrid.innerHTML = '';

        const items = jewelryData[category] || [];
        const currentId = category === 'necklaces' ? currentNecklace : currentEarring;

        // Add "None" option
        const noneDiv = document.createElement('div');
        noneDiv.className = 'sheet-item none-item' + (currentId === null ? ' active' : '');
        noneDiv.dataset.id = '';
        noneDiv.innerHTML = `<span class="none-label">None</span>`;
        noneDiv.addEventListener('click', () => selectItem(category, null));
        sheetGrid.appendChild(noneDiv);

        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'sheet-item' + (item.id === currentId ? ' active' : '');
            div.dataset.id = item.id;
            div.innerHTML = `<img src="${item.image}" alt="${item.name}" loading="lazy">`;
            div.addEventListener('click', () => selectItem(category, item.id));
            sheetGrid.appendChild(div);
        });
    }

    function selectItem(category, id) {
        if (category === 'necklaces') {
            currentNecklace = id;
        } else {
            currentEarring = id;
        }

        sheetGrid.querySelectorAll('.sheet-item').forEach(el => {
            const isNone = el.classList.contains('none-item');
            const elId = isNone ? null : el.dataset.id;
            el.classList.toggle('active', elId === id);
        });

        if (ws && ws.readyState === WebSocket.OPEN) {
            if (category === 'necklaces') {
                ws.send(JSON.stringify({ type: 'select_necklace', id: currentNecklace }));
            } else {
                ws.send(JSON.stringify({ type: 'select_earring', id: currentEarring }));
            }
        }

        closeSheet();
    }

    // Tab switching
    document.querySelectorAll('.sheet-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.sheet-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            renderGrid(tab.dataset.category);
        });
    });

    jewelryOpenBtn.addEventListener('click', openSheet);
    sheetBackdrop.addEventListener('click', closeSheet);

    async function loadJewelry() {
        try {
            const [neckRes, earRes] = await Promise.all([
                fetch('/necklaces'),
                fetch('/earrings'),
            ]);
            const neckData = await neckRes.json();
            const earData = await earRes.json();

            jewelryData.necklaces = neckData.necklaces || [];
            jewelryData.earrings = earData.earrings || [];

            if (jewelryData.necklaces.length > 0) {
                currentNecklace = jewelryData.necklaces[0].id;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'select_necklace', id: currentNecklace }));
                }
            }

            renderGrid('necklaces');
        } catch (e) {
            console.error('Failed to load jewelry:', e);
        }
    }

    loadJewelry();

    resetBtn.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'reset_calibration' }));
        }
    });

    clipCheckbox.addEventListener('change', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'toggle_face_clip', enabled: clipCheckbox.checked }));
        }
    });

    debugMaskCheckbox.addEventListener('change', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'toggle_debug_mask', enabled: debugMaskCheckbox.checked }));
        }
    });

    // --- Init ---
    connect();
    startCamera();
})();
