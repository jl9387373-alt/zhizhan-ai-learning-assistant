// 番茄钟变量
let pomodoroTimer = null;
let pomodoroTimeLeft = 25 * 60;
let pomodoroIsRunning = false;
let pomodoroMode = 'focus'; // focus, short-break, long-break
let pomodoroTotalTime = 25 * 60;

// Monaco Editor 实例
let noteEditorInstance = null;
let monacoLoaderPromise = null;
let mermaidLoaderPromise = null;
let isSendingMessage = false;
const pageLoadedState = {};

// 应用初始化
function initApp() {
    // alert('DEBUG: 前端代码 app_v2.js 已加载！如未看到此弹窗，请强制刷新。');
    if (!checkAuth()) return;
    trackAppOpened();
    setupEventListeners();
    setupAdminInterface();
    loadUserTheme();
    setupPomodoro(); // 初始化番茄钟
    updateChatActionButtons();
    
    // 检查用户画像是否已完善
    checkProfileStatus(getCurrentUserId());
}

async function checkProfileStatus(userId) {
    try {
        const data = await apiRequest(`/api/auth/profile?user_id=${userId}`);
        if (data.status === 'success' && data.data) {
            if (!data.data.profile_completed) {
                document.getElementById('profileSetupModal').style.display = 'flex';
            }
        }
    } catch (e) {
        console.error("检查画像状态失败", e);
    }
}

async function submitProfileSetup() {
    const userId = getCurrentUserId();
    const major = document.getElementById('profileMajor').value.trim();
    const grade = document.getElementById('profileGrade').value;
    const goal = document.getElementById('profileGoal').value.trim();
    const knownSkillsStr = document.getElementById('profileKnownSkills').value.trim();
    const weakAreasStr = document.getElementById('profileWeakAreas').value.trim();
    const dailyHours = parseFloat(document.getElementById('profileDailyHours').value);

    if (!major || !grade || !goal) {
        showError("请填写专业、年级和学习目标");
        return;
    }

    const knownSkills = knownSkillsStr.split(/[,，]/).map(s => s.trim()).filter(s => s);
    const weakAreas = weakAreasStr.split(/[,，]/).map(s => s.trim()).filter(s => s);

    try {
        const result = await apiRequest('/api/auth/profile', {
            method: 'POST',
            body: JSON.stringify({
                user_id: userId,
                major,
                grade,
                goal,
                known_skills: knownSkills,
                weak_areas: weakAreas,
                daily_hours: dailyHours,
                profile_completed: true
            })
        });

        if (result.status === 'success') {
            document.getElementById('profileSetupModal').style.display = 'none';
            showSuccess("学习画像已更新，将为您提供更精准的服务！");
        } else {
            showError(result.message || "保存失败");
        }
    } catch (error) {
        showError("网络错误: " + error.message);
    }
}

// 检查认证状态
function checkAuth() {
    const userId = localStorage.getItem('user_id');
    const account = localStorage.getItem('account');
    
    if (!userId) {
        window.location.href = 'login.html';
        return false;
    }
    
    const userSpan = document.getElementById('currentUser');
    if (userSpan && account) {
        userSpan.textContent = account;
    }
    return true;
}

// 退出登录
function logout() {
    localStorage.removeItem('user_id');
    localStorage.removeItem('account');
    localStorage.removeItem('role');
    window.location.href = 'login.html';
}

// 获取当前用户ID
function getCurrentUserId() {
    return localStorage.getItem('user_id') || 'default';
}

// 获取或创建当前浏览器会话ID（仅用于埋点，不落业务库）
function getAnalyticsSessionId() {
    let sessionId = sessionStorage.getItem('analytics_session_id');
    if (!sessionId) {
        sessionId = 'web_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10);
        sessionStorage.setItem('analytics_session_id', sessionId);
    }
    return sessionId;
}

// 前端统一埋点上报：后端只接受已登录用户的事件
function trackClientEvent(event, properties = {}) {
    try {
        const userId = getCurrentUserId();
        if (!userId || userId === 'default') return;
        fetch('/api/track', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                event,
                user_id: userId,
                session_id: getAnalyticsSessionId(),
                properties
            }),
            keepalive: true
        }).catch(() => {});
    } catch (e) {
        console.warn('埋点上报失败', e);
    }
}

// 每个用户每天只上报一次 app_opened，避免刷新造成事件膨胀
function trackAppOpened() {
    try {
        const userId = getCurrentUserId();
        if (!userId || userId === 'default') return;
        const day = new Date().toISOString().slice(0, 10);
        const key = `analytics_app_opened_${userId}_${day}`;
        if (sessionStorage.getItem(key)) return;
        sessionStorage.setItem(key, '1');
        trackClientEvent('app_opened', { path: window.location.pathname });
    } catch (e) {
        console.warn('app_opened 埋点失败', e);
    }
}

// HTML 转义工具（用于安全渲染账号名等外部文本）
function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

// 设置事件监听器
function setupEventListeners() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        const pageId = item.getAttribute('data-page');
        
        item.addEventListener('click', function(e) {
            e.preventDefault();
            switchPage(pageId);
            
            // 更新移动端菜单
            if (window.innerWidth <= 768) {
                toggleSidebar();
            }
        });
    });
    
    // 侧边栏切换按钮
    document.getElementById('menuToggle').addEventListener('click', toggleSidebar);
    
    // 主题切换按钮
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
    
    // 窗口大小改变时的响应式处理
    window.addEventListener('resize', handleResize);
    
    // ESC键关闭侧边栏
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && window.innerWidth <= 768) {
            const sidebar = document.querySelector('.sidebar');
            if (sidebar.classList.contains('open')) {
                toggleSidebar();
            }
        }
    });
}



// 切换侧边栏
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    sidebar.classList.toggle('open');
    mainContent.classList.toggle('expanded');
}

// 切换主题
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    
    // 更新主题切换按钮图标
    const themeToggle = document.getElementById('themeToggle');
    const icon = themeToggle.querySelector('i');
    icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    
    // 保存主题到本地存储
    localStorage.setItem('theme', newTheme);
}

// 加载用户主题偏好
function loadUserTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // 更新主题切换按钮图标
    const themeToggle = document.getElementById('themeToggle');
    const icon = themeToggle.querySelector('i');
    icon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

// 窗口大小改变时的处理
function handleResize() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    if (window.innerWidth > 768) {
        sidebar.classList.remove('open');
        mainContent.classList.remove('expanded');
    }
}

// 切换页面
function switchPage(pageId) {
    // 隐藏所有页面
    document.querySelectorAll('.page').forEach(page => {
        page.style.display = 'none';
        page.classList.remove('active');
    });
    
    // 移除所有导航项激活状态
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // 显示目标页面
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.style.display = 'block';
        setTimeout(() => targetPage.classList.add('active'), 10);
    }
    
    // 激活对应导航项
    const navItem = document.querySelector(`.nav-item[data-page="${pageId}"]`);
    if (navItem) {
        navItem.classList.add('active');
    }
    
    // 进入个人中心时刷新资料
    if (pageId === 'profile' && typeof loadProfilePage === 'function') {
        loadProfilePage();
    }

    // 移动端：关闭侧边栏
    if (window.innerWidth <= 768) {
        const sidebar = document.querySelector('.sidebar');
        const mainContent = document.querySelector('.main-content');
        sidebar.classList.remove('open');
        mainContent.classList.remove('expanded');
    }
    
    // 页面特定加载逻辑
    if (pageId === 'dashboard') updateDashboardData();
    if (pageId === 'plans') loadPlans();
    if (pageId === 'resources') loadResources();
    if (pageId === 'progress') loadProgress();
    if (pageId === 'quiz') generateNewQuiz();
    if (pageId === 'review') getNextReview();
    if (pageId === 'notes') loadNotes();
    if (pageId === 'admin') loadAdminStats();
    if (pageId === 'profile') loadUserProfile();
    if (pageId === 'flashcards') loadFlashcards();
}

async function loadMistakes() {
    // 移除
}

function reviewMistake(mistakeId) {
    // 移除
}

function deleteMistake(mistakeId) {
    // 移除
}

// ===== 新会话欢迎空状态：无消息时展示，发出第一条消息后隐藏 =====
function hideChatWelcome() {
    const welcome = document.getElementById('chatWelcome');
    if (welcome) welcome.hidden = true;
}

// 只清空消息，保留欢迎空状态节点
function clearChatMessages() {
    const box = document.getElementById('chatContainer');
    if (!box) return;
    box.querySelectorAll('.message').forEach(node => node.remove());
}

// 按当前消息数量同步空状态显隐
function syncChatEmptyState() {
    const box = document.getElementById('chatContainer');
    if (!box) return;
    const welcome = document.getElementById('chatWelcome');
    if (welcome) welcome.hidden = !!box.querySelector('.message');
}

function resetChatToWelcome() {
    clearChatMessages();
    syncChatEmptyState();
}

// 欢迎区快捷提问：只填入输入框，由用户确认后发送
function prefillChatInput(text) {
    const input = document.getElementById('userInput');
    if (!input) return;
    input.value = text;
    input.focus();
    try {
        input.setSelectionRange(text.length, text.length);
    } catch (e) {}
}

// 计划入口只给填写引导，由用户决定是否发送；主题和时间均可修改。
function showPlanInputGuide() {
    if (typeof switchPage === 'function') switchPage('chat');
    const guide = document.getElementById('planInputGuide');
    if (guide) guide.hidden = false;
    const input = document.getElementById('userInput');
    if (input) {
        input.placeholder = '例如：我想两周内入门 RAG，每天学 45 分钟，做一个问答应用';
        input.focus();
    }
}

function hidePlanInputGuide() {
    const guide = document.getElementById('planInputGuide');
    if (guide) guide.hidden = true;
    const input = document.getElementById('userInput');
    if (input) input.placeholder = '输入您的问题或需求...';
}

function prefillPlanExample() {
    prefillChatInput('我想在两周内入门 RAG，每天能学 45 分钟，目标是做出一个简单问答应用。');
}

// 快捷提示只插入可编辑的字段名，不预设用户的学习内容，也不自动发送。
function insertPlanField(field) {
    const labels = {
        topic: '学习主题：',
        goal: '目标：',
        duration: '周期：',
        daily: '每天时间：'
    };
    const label = labels[field];
    const input = document.getElementById('userInput');
    if (!label || !input) return;

    const value = input.value;
    const existing = value.indexOf(label);
    if (existing !== -1) {
        input.focus();
        input.setSelectionRange(existing + label.length, existing + label.length);
        return;
    }

    const start = input.selectionStart ?? value.length;
    const end = input.selectionEnd ?? value.length;
    const before = value.slice(0, start);
    const after = value.slice(end);
    const prefix = before && !/[\s，。；;]$/.test(before) ? '；' : '';
    const suffix = after && !/^[\s，。；;]/.test(after) ? '；' : '';
    input.value = before + prefix + label + suffix + after;
    const cursor = (before + prefix + label).length;
    input.focus();
    input.setSelectionRange(cursor, cursor);
    input.dispatchEvent(new Event('input', { bubbles: true }));
}
// 添加消息到聊天容器
function addMessage(content, isUser) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + (isUser ? 'user-message' : 'agent-message');

    const icon = isUser ? '<i class="fas fa-user-circle"></i>'
                        : '<i class="fas fa-robot"></i>';
    const speakBtn = isUser ? ''
        : '<i class="fas fa-volume-up speak-btn" onclick="speakText(this.closest(\'.message\').querySelector(\'.message-body\').innerText)" title="朗读"></i>';

    messageDiv.innerHTML =
        `<div class="message-head">${icon}<span>${isUser ? '您' : '启发式导师'}</span>${speakBtn}</div>` +
        `<div class="message-body">${content}</div>`;
    chatContainer.appendChild(messageDiv);
    hideChatWelcome();
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageDiv;
}

function formatText(text) {
    if (!text) return '';
    return text.replace(/\n/g, '<br>');
}

// 内部 mode/source 仅用于判断，不作为明文文案展示给用户。
// 只有真正用到资料时才展示来源标识与资料片段。
function sourceLabel(mode) {
    const labels = {
        'rag_hybrid': '资料结合模型知识',
        'rag_llm': '知识库问答',
        'rag_context': '知识要点提取',
        'knowledge_base': '直接知识解答',
        'knowledge_base_cross': '跨主题知识整合',
        'business_data': '学习记录数据',
        'default': '学习建议'
    };
    return labels[mode] || '知识库问答';
}

// 是否真正使用了资料：general_llm 是纯模型回答，不展示任何来源标识
function usesKnowledgeBase(mode, isGeneral) {
    return !(isGeneral || mode === 'general_llm');
}

function renderSourceBadge(mode, isGeneral) {
    if (!mode || !usesKnowledgeBase(mode, isGeneral)) return '';
    return `<div style="display:inline-flex;align-items:center;gap:5px;margin:8px 0 2px;padding:3px 10px;border-radius:12px;font-size:0.8em;color:#5B5CE2;background:rgba(91,92,226,0.08);">`
        + `<i class="fas fa-book-open" style="font-size:0.9em;"></i>${sourceLabel(mode)}</div>`;
}

function renderAnswerWithSources(data) {
    let html = '';
    const mode = data.source || data.mode || '';
    const isGeneralAnswer = !usesKnowledgeBase(mode, data.is_general);
    if (mode && !isGeneralAnswer) {
        html += renderSourceBadge(mode, data.is_general);
    }
    if (data.answer) {
        // 预处理答案：移除一些常见的冗余前缀
        let cleanAnswer = data.answer.trim();
        cleanAnswer = cleanAnswer.replace(/^我理解你想了解[^。]+。让我为你详细解答：\s*/gm, '');
        cleanAnswer = cleanAnswer.replace(/^关于[^，]+，我可以告诉你：\s*/gm, '');

        if (typeof marked !== 'undefined' && marked && typeof marked.parse === 'function') {
            if (typeof marked.setOptions === 'function') {
                marked.setOptions({
                    breaks: true,
                    gfm: true,
                    headerIds: false,
                    mangle: false
                });
            }
            try {
                const renderedMarkdown = marked.parse(cleanAnswer);
                html += `<div class="markdown-body">${renderedMarkdown}</div>`;
            } catch (e) {
                console.error('Markdown渲染失败:', e);
                html += `<div>${formatText(cleanAnswer)}</div>`;
            }
        } else {
            html += `<div>${formatText(cleanAnswer)}</div>`;
        }
    }
    
    // 参考资料片段保持折叠显示；纯模型回答不展示无关资料片段
    if (!isGeneralAnswer && data.context_snippets && data.context_snippets.length) {
        const filtered = data.context_snippets.slice(0, 3).filter(snippet => snippet.content);
        if (filtered.length) {
            html += '<div style="margin-top:15px;">';
            html += '<details style="border:1px solid var(--border-color);border-radius:8px;background-color:rgba(102,126,234,0.03);overflow:hidden;">';
            html += '<summary style="padding:10px 12px;cursor:pointer;font-weight:bold;color:var(--primary-color);user-select:none;display:flex;align-items:center;">';
            html += '<i class="fas fa-search-plus" style="margin-right:8px;font-size:0.9em;"></i>查看参考资料来源';
            html += '</summary>';
            html += '<div style="padding:0 12px 12px 12px;border-top:1px solid rgba(0,0,0,0.05);">';
            filtered.forEach((snippet, idx) => {
                html += `<div style="margin-top:10px;padding-top:10px;border-top:${idx === 0 ? 'none' : '1px dashed rgba(0,0,0,0.05)'}">`;
                const meta = snippet.metadata || {};
                const sourceLabel = meta.file_name || meta.title || meta.url || '';
                const titleText = snippet.title || sourceLabel || `资料片段 ${idx + 1}`;
                html += `<div style="font-size:0.85em;color:var(--text-light);margin-bottom:4px;font-weight:600;">${titleText}</div>`;
                if (sourceLabel && titleText !== sourceLabel) {
                    html += `<div style="font-size:0.8em;color:var(--text-light);margin-bottom:6px;opacity:0.9;">来源：${sourceLabel}</div>`;
                }
                html += `<div style="font-size:0.95em;line-height:1.5;color:var(--text-main);">${snippet.content}</div>`;
                html += `</div>`;
            });
            html += '</div></details></div>';
        }
    }
    return html || '暂无答案';
}

function openCreateFlashcardWithData(subject, front, back) {
    const modal = document.getElementById('createFlashcardModal');
    const subjEl = document.getElementById('cardSubject');
    const frontEl = document.getElementById('cardFront');
    const backEl = document.getElementById('cardBack');
    if (!modal || !subjEl || !frontEl || !backEl) return;
    subjEl.value = (subject || '通用').trim() || '通用';
    frontEl.value = (front || '').trim();
    backEl.value = (back || '').trim();
    modal.style.display = 'block';
}

function attachFlashcardQuickAction(messageDiv, payload) {
    if (!messageDiv || !payload || !payload.question || !payload.answer) return;
    const contentContainer = messageDiv.querySelector('div:last-child');
    if (!contentContainer) return;

    const wrap = document.createElement('div');
    wrap.className = 'message-tools';
    wrap.style.display = 'flex';
    wrap.style.alignItems = 'center';
    wrap.style.gap = '6px';
    wrap.style.flexWrap = 'wrap';

    const btn = document.createElement('button');
    btn.className = 'secondary-btn';
    btn.style.padding = '4px 8px';
    btn.style.fontSize = '0.85em';
    btn.innerHTML = '<i class="fas fa-clone"></i> 生成闪卡';
    btn.addEventListener('click', () => {
        openCreateFlashcardWithData(payload.subject || '通用', payload.question, payload.answer);
    });
    wrap.appendChild(btn);
    contentContainer.appendChild(wrap);
}

function attachFeedbackControls(messageDiv, payload) {
    if (!messageDiv || !payload || !payload.answer || !payload.question) return;
    if (localStorage.getItem('role') === 'admin' && payload.disableForAdmin) return;

    const contentContainer = messageDiv.querySelector('div:last-child');
    if (!contentContainer) return;

    const head = messageDiv.querySelector('.message-head');
    if (head && !head.querySelector('.message-more-btn')) {
        const moreBtn = document.createElement('i');
        moreBtn.className = 'fas fa-ellipsis-h message-more-btn';
        moreBtn.title = '更多操作';
        moreBtn.setAttribute('role', 'button');
        moreBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            messageDiv.classList.toggle('show-tools');
        });
        head.appendChild(moreBtn);
    }

    const controls = document.createElement('div');
    controls.className = 'message-tools';
    controls.style.display = 'flex';
    controls.style.alignItems = 'center';
    controls.style.gap = '6px';
    controls.style.fontSize = '0.85em';
    controls.style.flexWrap = 'wrap';

    const upBtn = document.createElement('button');
    upBtn.className = 'secondary-btn';
    upBtn.style.padding = '4px 8px';
    upBtn.innerHTML = '<i class="fas fa-thumbs-up"></i> 有用';

    const downBtn = document.createElement('button');
    downBtn.className = 'secondary-btn';
    downBtn.style.padding = '4px 8px';
    downBtn.innerHTML = '<i class="fas fa-thumbs-down"></i> 没用';

    const commentBtn = document.createElement('button');
    commentBtn.className = 'secondary-btn';
    commentBtn.style.padding = '4px 8px';
    commentBtn.innerHTML = '<i class="fas fa-comment-dots"></i> 备注';

    const textarea = document.createElement('textarea');
    textarea.placeholder = '可选：补充你期望的答案/指出哪里不对（用于后续优化）';
    textarea.style.width = '100%';
    textarea.style.marginTop = '10px';
    textarea.style.minHeight = '70px';
    textarea.style.display = 'none';

    const status = document.createElement('div');
    status.style.fontSize = '0.85em';
    status.style.color = 'var(--text-light)';
    status.style.marginTop = '10px';
    status.style.display = 'none';

    const disableAll = () => {
        upBtn.disabled = true;
        downBtn.disabled = true;
        commentBtn.disabled = true;
        upBtn.style.opacity = '0.7';
        downBtn.style.opacity = '0.7';
        commentBtn.style.opacity = '0.7';
    };

    const submit = async (rating) => {
        disableAll();
        status.style.display = 'block';
        status.textContent = '正在提交反馈...';
        try {
            await apiRequest('/api/feedback', {
                method: 'POST',
                body: JSON.stringify({
                    ...payload,
                    rating: rating,
                    comment: textarea.value.trim()
                })
            });
            status.textContent = '反馈已记录，感谢你的帮助！';
        } catch (e) {
            status.textContent = '提交失败：' + (e && e.message ? e.message : '未知错误');
        }
        setTimeout(() => messageDiv.classList.remove('tools-pinned'), 3000);
    };

    upBtn.addEventListener('click', () => submit(1));
    downBtn.addEventListener('click', () => submit(-1));
    commentBtn.addEventListener('click', () => {
        const willShow = textarea.style.display === 'none';
        textarea.style.display = willShow ? 'block' : 'none';
        messageDiv.classList.toggle('tools-pinned', willShow);
    });

    controls.appendChild(upBtn);
    controls.appendChild(downBtn);
    controls.appendChild(commentBtn);
    controls.appendChild(textarea);
    controls.appendChild(status);
    contentContainer.appendChild(controls);
}
// ========== 统一的工具函数 ==========

// 显示友好的错误提示
function showError(message, error = null) {
    console.error('错误:', message, error);
    showNotification(message, 'error');
}

// 显示成功的提示
function showSuccess(message) {
    showNotification(message, 'success');
}

function showNotification(message, type = 'success') {
    let container = document.getElementById('notificationContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notificationContainer';
        container.className = 'notification-container';
        document.body.appendChild(container);
    }

    const item = document.createElement('div');
    item.className = `notification ${type}`;
    item.textContent = message || '';
    container.appendChild(item);

    setTimeout(() => {
        item.classList.add('hide');
        setTimeout(() => safeRemove(item), 220);
    }, 2600);
}

function setChatInputState(disabled) {
    const input = document.getElementById('userInput');
    const sendBtn = document.getElementById('chatSendBtn');
    const micBtn = document.getElementById('micBtn');
    const imageBtn = document.querySelector('.input-container [onclick*="imageInput"]');
    if (input) input.disabled = disabled;
    if (sendBtn) {
        sendBtn.disabled = disabled;
        sendBtn.title = disabled ? '发送中' : '发送';
    }
    if (micBtn) micBtn.disabled = disabled;
    if (imageBtn) imageBtn.disabled = disabled;
    updateChatActionButtons();
}

function updateChatActionButtons() {
    const pauseBtn = document.getElementById('chatPauseBtn');
    const retryBtn = document.getElementById('chatRetryBtn');
    // 暂停：仅生成中/已暂停时显示；暂停时图标切换为继续
    if (pauseBtn) {
        pauseBtn.hidden = !isSendingMessage && !isStreamPaused;
        pauseBtn.title = isStreamPaused ? '继续生成' : '暂停生成';
        pauseBtn.setAttribute('aria-label', isStreamPaused ? '继续生成' : '暂停生成');
        const icon = pauseBtn.querySelector('i');
        if (icon) icon.className = isStreamPaused ? 'fas fa-play' : 'fas fa-pause';
    }
    // 重试：仅空闲且有上一条可重试消息时显示
    if (retryBtn) {
        retryBtn.hidden = isSendingMessage || !(lastPayload && lastPayload.message);
    }
}

function loadScriptOnce(url) {
    const existing = document.querySelector(`script[data-src="${url}"]`);
    if (existing) {
        return existing.dataset.loaded === 'true'
            ? Promise.resolve()
            : new Promise((resolve, reject) => {
                existing.addEventListener('load', resolve, { once: true });
                existing.addEventListener('error', reject, { once: true });
            });
    }
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = url;
        script.async = true;
        script.dataset.src = url;
        script.onload = () => {
            script.dataset.loaded = 'true';
            resolve();
        };
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

function loadStyleOnce(url) {
    const existing = document.querySelector(`link[data-href="${url}"]`);
    if (existing) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = url;
    link.dataset.href = url;
    document.head.appendChild(link);
}

function ensureMermaidLoaded() {
    if (typeof mermaid !== 'undefined') return Promise.resolve();
    if (!mermaidLoaderPromise) {
        mermaidLoaderPromise = loadScriptOnce('https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js');
    }
    return mermaidLoaderPromise;
}

async function ensureMonacoLoaded() {
    if (typeof monaco !== 'undefined' && typeof require !== 'undefined') return;
    if (!monacoLoaderPromise) {
        monacoLoaderPromise = (async () => {
            loadStyleOnce('https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/editor/editor.main.css');
            await loadScriptOnce('https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/loader.min.js');
            await new Promise((resolve, reject) => {
                try {
                    require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' } });
                    require(['vs/editor/editor.main'], () => resolve(), reject);
                } catch (e) {
                    reject(e);
                }
            });
        })();
    }
    return monacoLoaderPromise;
}

function showLoading(container, message = '加载中...') {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-spinner';
    loadingDiv.innerHTML = `<div class="spinner"></div><p>${message}</p>`;
    container.innerHTML = '';
    container.appendChild(loadingDiv);
    return loadingDiv;
}

// 安全的移除元素
function safeRemove(element) {
    if (element && element.parentNode) {
        element.parentNode.removeChild(element);
    }
}

    // 统一的API请求函数（带错误处理）
async function apiRequest(url, options = {}, timeout = 60000) {
    try {
        // 创建超时Promise
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('请求超时')), timeout);
        });
        
        // 使用Promise.race实现超时
        const response = await Promise.race([
            fetch(url, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            }),
            timeoutPromise
        ]);
        
        let data = null;
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
            try {
                data = await response.json();
            } catch (e) {
                console.warn('JSON解析失败:', e);
            }
        }

        // 检查响应状态
        if (!response.ok) {
            if (data && data.message) {
                throw new Error(data.message);
            }
            if (data && data.error) {
                throw new Error(data.error);
            }
            const text = data ? JSON.stringify(data) : await response.text();
            throw new Error(`HTTP ${response.status}: ${text.substring(0, 200)}`);
        }
        
        if (!data) {
             // 尝试读取文本并解析
             const text = await response.text();
             try {
                 data = JSON.parse(text);
             } catch (e) {
                 // 如果不是JSON且状态为200，可能就是空或者纯文本，视具体接口而定
                 // 但我们的API约定返回JSON
                 throw new Error(`服务器返回了非JSON格式: ${text.substring(0, 200)}`);
             }
        }
        
        // 检查业务状态
        if (data.status === 'error') {
            throw new Error(data.message || '请求处理失败');
        }
        
        return data;
    } catch (error) {
        // 友好的错误处理
        if (error.message.includes('Failed to fetch') || error.message.includes('Network')) {
            throw new Error('网络连接失败，请检查网络连接后重试');
        } else if (error.message.includes('timeout')) {
            throw new Error('请求超时，请稍后重试');
        } else {
            throw error;
        }
    }
}

// 当前会话ID（全局变量）
let currentConversationId = null;
let currentImageBase64 = null;
let currentStreamController = null;
let isStreamPaused = false;
let pendingPayload = null;
let lastPayload = null;

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            currentImageBase64 = e.target.result.split(',')[1];
            const preview = document.getElementById('imagePreview');
            if (preview) preview.src = e.target.result;
            const container = document.getElementById('imagePreviewContainer');
            if (container) container.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

function clearImage() {
    currentImageBase64 = null;
    const preview = document.getElementById('imagePreview');
    if (preview) preview.src = '';
    const container = document.getElementById('imagePreviewContainer');
    if (container) container.style.display = 'none';
    const input = document.getElementById('imageInput');
    if (input) input.value = '';
}

// 语音识别相关变量
let recognition = null;
let isListening = false;

// 切换语音输入
function toggleVoiceInput() {
    const micBtn = document.getElementById('micBtn');
    const micIcon = micBtn.querySelector('i');
    const input = document.getElementById('userInput');

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        showError('您的浏览器不支持语音输入，请使用 Chrome 或 Edge 浏览器。');
        return;
    }

    if (isListening) {
        if (recognition) recognition.stop();
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-CN';
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onstart = function() {
        isListening = true;
        micIcon.style.color = '#ff4d4f'; // 录音时显示红色
        micIcon.className = 'fas fa-microphone-slash'; // 显示静音图标表示点击停止
        micBtn.style.background = 'rgba(255, 77, 79, 0.1)';
        input.placeholder = '正在聆听...';
    };

    recognition.onend = function() {
        isListening = false;
        micIcon.style.color = '#666';
        micIcon.className = 'fas fa-microphone';
        micBtn.style.background = 'none';
        input.placeholder = '输入您的问题或需求...';
    };

    recognition.onresult = function(event) {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        
        if (finalTranscript) {
            input.value = finalTranscript;
            // 语音识别结束后稍微延迟自动发送，提升体验
            setTimeout(() => sendMessage(), 500);
        } else if (interimTranscript) {
             input.value = interimTranscript;
        }
    };

    recognition.onerror = function(event) {
        console.error('Speech recognition error', event.error);
        isListening = false;
        micIcon.style.color = '#666';
        micIcon.className = 'fas fa-microphone';
        micBtn.style.background = 'none';
        input.placeholder = '语音识别出错: ' + event.error;
    };

    try {
        recognition.start();
    } catch (e) {
        console.error(e);
    }
}

// 语音合成（朗读）
function speakText(text) {
    if (!('speechSynthesis' in window)) {
        showError('您的浏览器不支持语音合成');
        return;
    }
    
    // 如果正在朗读，则停止
    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        return;
    }
    
    // 简单的文本清理，移除Markdown符号
    const cleanText = text.replace(/[*#`]/g, '').substring(0, 200); // 限制长度防止读太久
    
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = 'zh-CN';
    utterance.rate = 1.0;
    
    window.speechSynthesis.speak(utterance);
}

// 发送消息
async function sendMessage(action = null, options = {}) {
    if (isSendingMessage) return;
    const input = document.getElementById('userInput');
    const query = (options.forceMessage ?? input.value).trim();
    const showUserMessage = options.showUserMessage !== false;
    
    if (!query && !action) {
        // showError('请输入问题');
        return;
    }

    hidePlanInputGuide();

    // 开始对话后不再展示欢迎空状态（动作类消息没有气泡）
    hideChatWelcome();
    
    // 1. 如果有输入内容，先显示用户消息
    if (query && showUserMessage) {
        addMessage(query, true);
        
        // 如果有图片，追加显示
        const previewImage = options.forceImage ?? currentImageBase64;
        if (previewImage) {
            const messages = document.getElementsByClassName('user-message');
            if (messages.length > 0) {
                const img = document.createElement('img');
                img.src = 'data:image/jpeg;base64,' + previewImage;
                img.style.maxWidth = '200px';
                img.style.borderRadius = '8px';
                img.style.marginTop = '8px';
                img.style.display = 'block';
                messages[messages.length - 1].appendChild(img);
            }
        }
    }
    
    // 本会话第一条消息：用于在拿到服务端 conversation_id 后刷新侧栏标题
    const isFirstChatMessage = document.querySelectorAll('#chatContainer .message').length <= 1;
    const requestMessage = query || (action ? action : '');
    const imageToSend = options.forceImage ?? currentImageBase64;
    pendingPayload = {
        message: requestMessage,
        action: action,
        image: imageToSend
    };
    lastPayload = { ...pendingPayload };
    isStreamPaused = false;
    isSendingMessage = true;
    setChatInputState(true);
    
    // 清空输入框和图片
    if (options.forceMessage === undefined) {
        input.value = '';
        clearImage();
    }
    
    // 2. 显示加载动画
    const chatContainer = document.getElementById('chatContainer');
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading-message-' + Date.now();
    loadingDiv.className = 'message agent-message loading';
    loadingDiv.innerHTML = '<div style="display:flex;align-items:center;margin-bottom:6px;font-weight:bold;font-size:0.85em;opacity:0.9;"><i class="fas fa-robot" style="margin-right:8px;font-size:0.9em;opacity:0.8;"></i>启发式导师</div>' + 
                          '<div><div class="loading-spinner-small"></div> 正在思考中...</div>';
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    try {
        // 3. 发送流式请求
        currentStreamController = new AbortController();
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: requestMessage,
                user_id: getCurrentUserId(),
                conversation_id: currentConversationId,
                action: action,
                image: imageToSend,
                skip_user_message: options.showUserMessage === false
            }),
            signal: currentStreamController.signal
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // 移除加载动画
        safeRemove(loadingDiv);

        // 创建消息容器
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message agent-message';
        // 添加头像和内容容器
        msgDiv.innerHTML = '<div class="message-head"><i class="fas fa-robot"></i><span>启发式导师</span><i class="fas fa-volume-up speak-btn" onclick="speakText(this.closest(\'.message\').querySelector(\'.content\').innerText)" title="朗读"></i></div>'
            + '<div class="think-line"><span class="think-spinner"></span><span class="think-text">正在分析请求</span><span class="think-elapsed"></span></div>'
            + '<div class="content message-body markdown-body"></div>';
        const streamStart = Date.now();
        chatContainer.appendChild(msgDiv);
        const contentDiv = msgDiv.querySelector('.content');

        // 4. 处理流式响应
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let metadata = {};
        let isCompleteResponse = false;
        let completeResponseData = null;
        let buffer = ''; // 用于处理分片到达的数据

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                // 处理最后剩余的 buffer
                if (buffer) {
                    processLine(buffer);
                }
                break;
            }
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // 最后一个可能是半截行，留给下次处理
            
            for (const line of lines) {
                processLine(line);
            }
        }

        function processLine(line) {
            if (line.startsWith('data: ')) {
                const dataStr = line.slice(6).trim();
                if (dataStr === '[DONE]') return;
                
                try {
                    const data = JSON.parse(dataStr);
                    
                    if (data.type === 'session') {
                        // 服务端已创建/确认会话并落库，前端同步会话ID
                        const sid = data.data && data.data.conversation_id;
                        if (sid) {
                            const conversationChanged = sid !== currentConversationId;
                            currentConversationId = sid;
                            if (conversationChanged || isFirstChatMessage) loadConversationList();
                        }
                    } else if (data.type === 'token') {
                        // 累积文本并渲染
                        const token = typeof data.data === 'string' ? data.data : JSON.stringify(data.data);
                        if (!fullText) {
                            // 首个 token 到达：思考行转为结果摘要（保留一行，不刷屏）
                            try {
                                const line = msgDiv.querySelector('.think-line');
                                if (line) {
                                    const n = Number((metadata && metadata.docs_count) || 0);
                                    const secs = ((Date.now() - streamStart) / 1000).toFixed(1);
                                    line.classList.add('done');
                                    line.innerHTML = `<span class="think-check">✓</span>`
                                        + `<span class="think-text">${n > 0 ? `参考 ${n} 篇资料` : '回答完成'} · 用时 ${secs} 秒</span>`;
                                }
                            } catch (e) { /* 忽略 */ }
                        }
                        fullText += token;
                        
                        // 使用 marked 解析 Markdown，增加错误兜底
                        try {
                            if (window.marked) {
                                contentDiv.innerHTML = marked.parse(fullText);
                                enhanceRenderedMarkdown(contentDiv);
                            } else {
                                contentDiv.innerHTML = renderLiteMarkdown(fullText);
                                enhanceRenderedMarkdown(contentDiv);
                            }
                        } catch (e) {
                            console.error('Marked error:', e);
                            contentDiv.innerText = fullText;
                        }

                        // 高亮代码块
                        if (window.hljs) {
                            contentDiv.querySelectorAll('pre code').forEach((block) => {
                                hljs.highlightElement(block);
                            });
                        }
                        // 自动滚动到底部
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    } else if (data.type === 'meta') {
                        metadata = data.data;
                        try {
                            const thinkText = msgDiv.querySelector('.think-text');
                            if (thinkText) {
                                const n = Number(metadata.docs_count || 0);
                                thinkText.textContent = n > 0
                                    ? `已检索到 ${n} 篇资料，正在组织答案`
                                    : '正在组织答案';
                            }
                        } catch (e) { /* 忽略 */ }
                        // 如果有上下文片段，保存到 dataset
                        if (metadata.context_snippets) {
                            msgDiv.dataset.snippets = JSON.stringify(metadata.context_snippets);
                        }
                    } else if (data.type === 'complete') {
                        // 非流式响应（如计划生成）
                        isCompleteResponse = true;
                        completeResponseData = data.data;
                        // 移除临时的消息容器
                        if (chatContainer.contains(msgDiv)) {
                            chatContainer.removeChild(msgDiv);
                        }
                        handleCompleteResponse(completeResponseData, query);
                    } else if (data.type === 'error') {
                        const errorText = String(data.message || data.data || '服务暂时不可用，请稍后重试。');
                        if (!fullText) {
                            contentDiv.textContent = errorText;
                        } else {
                            const errorNote = document.createElement('p');
                            errorNote.textContent = errorText;
                            contentDiv.appendChild(errorNote);
                        }
                    }
                } catch (e) {
                    console.error('Parse error for line:', line, e);
                }
            }
        }
        
        // 5. 流结束后添加交互控件（如果不是 complete 响应）
        if (!isCompleteResponse && fullText) {
            // 只有真正用到资料时才展示来源标识；纯模型回答不展示
            const streamMode = metadata.mode || metadata.source || '';
            if (streamMode && usesKnowledgeBase(streamMode, metadata.is_general)) {
                const badge = document.createElement('div');
                badge.innerHTML = renderSourceBadge(streamMode, metadata.is_general);
                if (badge.firstElementChild) {
                    contentDiv.insertBefore(badge.firstElementChild, contentDiv.firstChild);
                }
            }
            // 解析并分离追问问题
            let finalAnswer = fullText;
            let followUpQuestions = [];
            
            const followUpMatch = fullText.match(/\[FOLLOW_UP\]([\s\S]*?)\[\/FOLLOW_UP\]/);
            if (followUpMatch) {
                // 提取问题部分
                const questionsRaw = followUpMatch[1];
                finalAnswer = fullText.replace(followUpMatch[0], '').trim();
                
                // 重新渲染不带追问的正文
                contentDiv.innerHTML = window.marked ? marked.parse(finalAnswer) : finalAnswer;
                if (window.hljs) {
                    contentDiv.querySelectorAll('pre code').forEach((block) => hljs.highlightElement(block));
                }
                
                // 解析问题列表
                followUpQuestions = questionsRaw.split('\n')
                    .map(q => q.trim())
                    .filter(q => q.startsWith('- '))
                    .map(q => q.substring(2).trim());
            }

            // 添加追问按钮区域
            if (followUpQuestions.length > 0) {
                const followUpContainer = document.createElement('div');
                followUpContainer.className = 'follow-up-container';
                followUpContainer.style.marginTop = '10px';
                followUpContainer.style.display = 'flex';
                followUpContainer.style.flexWrap = 'wrap';
                followUpContainer.style.gap = '8px';
                
                const label = document.createElement('div');
                label.innerText = '💡 猜你想问：';
                label.style.width = '100%';
                label.style.fontSize = '0.85em';
                label.style.color = '#666';
                label.style.marginBottom = '4px';
                followUpContainer.appendChild(label);
                
                followUpQuestions.forEach(q => {
                    const btn = document.createElement('button');
                    btn.className = 'follow-up-btn';
                    btn.innerText = q;
                    btn.style.padding = '4px 12px';
                    btn.style.borderRadius = '16px';
                    btn.style.border = '1px solid #e0e0e0';
                    btn.style.background = '#f9f9f9';
                    btn.style.color = '#333';
                    btn.style.fontSize = '0.9em';
                    btn.style.cursor = 'pointer';
                    btn.style.transition = 'all 0.2s';
                    
                    btn.onmouseover = () => { btn.style.background = '#e3f2fd'; btn.style.borderColor = '#2196f3'; };
                    btn.onmouseout = () => { btn.style.background = '#f9f9f9'; btn.style.borderColor = '#e0e0e0'; };
                    btn.onclick = () => { sendQuickMessage(q); };
                    
                    followUpContainer.appendChild(btn);
                });
                
                msgDiv.appendChild(followUpContainer);
            }

            // 追加「查看参考资料来源」折叠面板（流式路径原来没有）
            appendSourcesPanel(msgDiv, metadata);

            // 添加闪卡快捷操作
            attachFlashcardQuickAction(msgDiv, {
                subject: metadata.subject || '通用',
                question: query,
                answer: finalAnswer
            });
            // 添加反馈控件
            attachFeedbackControls(msgDiv, {
                user_id: getCurrentUserId(),
                conversation_id: currentConversationId,
                subject: metadata.subject || '通用',
                question: query,
                answer: finalAnswer,
                source: metadata.source || 'rag_llm',
                context_snippets: metadata.context_snippets || []
            });
            
            // 刷新最近活动
            loadRecentActivities();
        }
        pendingPayload = null;
        
    } catch (error) {
        safeRemove(loadingDiv);
        if (error && error.name === 'AbortError') {
            if (isStreamPaused) {
                addMessage('已暂停本次回答。点击“继续”可重新生成。', false);
            } else {
                pendingPayload = null;
                addMessage('已取消本次回答。', false);
            }
        } else {
            pendingPayload = null;
            addMessage(`抱歉，处理请求时发生错误: ${error.message}`, false);
        }
    } finally {
        currentStreamController = null;
        isSendingMessage = false;
        setChatInputState(false);
        if (input) input.focus();
    }
}

// 处理完整的非流式响应
function handleCompleteResponse(responseData, query) {
    // 更新会话ID
    if (responseData && responseData.conversation_id) {
        currentConversationId = responseData.conversation_id;
        loadConversationList();
    }
    
    // 渲染不同类型的响应
    if (responseData?.type === 'plan_clarification') {
        addMessage(renderPlanClarification(responseData), false);
    } else if (responseData && responseData.plan && responseData.plan.weekly_plan) {
        const planHtml = renderPlan(responseData.plan, responseData.message);
        addMessage(planHtml, false);
    } else if (responseData && responseData.answer) {
        const renderedContent = renderAnswerWithSources(responseData);
        const msgDiv = addMessage(renderedContent, false);
        attachFlashcardQuickAction(msgDiv, {
            subject: responseData.subject || '通用',
            question: query,
            answer: responseData.answer
        });
        attachFeedbackControls(msgDiv, {
            user_id: getCurrentUserId(),
            conversation_id: currentConversationId,
            subject: responseData.subject || '通用',
            question: query,
            answer: responseData.answer,
            source: responseData.source || responseData.mode || '',
            mode: responseData.mode || responseData.source || '',
            context_snippets: responseData.context_snippets || []
        });
    } else if (responseData && Array.isArray(responseData.resources)) {
        const resourcesHtml = renderResources(responseData.resources);
        addMessage(resourcesHtml, false);
    } else if (responseData && responseData.message) {
        addMessage(responseData.message, false);
    } else {
        addMessage("抱歉，我没有理解您的请求，请稍后再试。", false);
    }
    
    // 处理意图相关的UI更新
    if (responseData && responseData.intent === 'plan_creation') {
        loadPlans();
    } else if (responseData && responseData.intent === 'quiz_generation') {
        switchPage('quiz');
        loadRecentQuizzes();
    }
    
    // 刷新最近活动
    loadRecentActivities();
}

// 发送快速消息
function sendQuickMessage(message) {
    const input = document.getElementById('userInput');
    input.value = message;
    sendMessage();
}

function togglePauseChat() {
    if (isSendingMessage && currentStreamController) {
        isStreamPaused = true;
        currentStreamController.abort();
        updateChatActionButtons();
        return;
    }
    if (isStreamPaused && pendingPayload && pendingPayload.message) {
        const payload = { ...pendingPayload };
        isStreamPaused = false;
        showNotification('继续生成中...', 'success');
        sendMessage(payload.action || null, {
            forceMessage: payload.message,
            forceImage: payload.image || null,
            showUserMessage: false
        });
    }
}

function retryLastMessage() {
    if (isSendingMessage) return;
    const payload = lastPayload && lastPayload.message ? { ...lastPayload } : null;
    if (!payload) {
        showNotification('暂无可重试的对话', 'error');
        return;
    }
    isStreamPaused = false;
    pendingPayload = { ...payload };
    showNotification('正在重试上一条问题...', 'success');
    sendMessage(payload.action || null, {
        forceMessage: payload.message,
        forceImage: payload.image || null,
        showUserMessage: false
    });
}

// 处理键盘事件
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (isSendingMessage) return;
        sendMessage();
    }
}

function escapePlanText(value) {
    return String(value ?? '').replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

function renderPlanClarification(data) {
    const esc = escapePlanText;
    const dialog = data.dialog;
    if (!dialog) return `<div class="plan-dialog">${esc(data.message).replace(/\n/g, '<br>')}</div>`;
    let html = `<div class="plan-dialog"><p class="plan-dialog-intro">${esc(dialog.intro)}</p>`;
    html += (dialog.errors || []).map(e => `<p class="plan-dialog-error">${esc(e)}</p>`).join('');
    (dialog.prompts || []).forEach(prompt => {
        html += `<div class="plan-dialog-question"><p>${esc(prompt.text)}</p><div class="plan-dialog-options">`;
        (prompt.options || []).forEach(option => {
            const message = option.message ? ` data-plan-message="${esc(option.message)}"` : '';
            const prefill = option.prefill ? ` data-plan-prefill="${esc(option.prefill)}"` : '';
            const after = option.prefill_after ? ` data-plan-after="${esc(option.prefill_after)}"` : '';
            html += `<button type="button" class="plan-choice-btn"${message}${prefill}${after}>${esc(option.label)}</button>`;
        });
        html += '</div></div>';
    });
    html += `<p class="plan-dialog-hint">${esc(dialog.hint)}</p>`;
    if (dialog.defaults) html += `<div class="plan-dialog-default"><button type="button" class="plan-choice-btn plan-choice-default" data-plan-message="${esc(dialog.defaults.message)}">${esc(dialog.defaults.label)}</button><span>${esc(dialog.default_note)}</span></div>`;
    return html + '</div>';
}

function planConditionMessage(plan) {
    return `请重新制定${plan.subject || ''}学习计划，目标：${plan.goal || ''}，周期：${plan.requested_duration || plan.duration || ''}，每天${plan.daily_hours ?? 2}小时`;
}

function planCurrentWeek(plan) {
    const total = Math.ceil((plan.duration_days || plan.daily_plan?.length || 7) / 7);
    const today = new Intl.DateTimeFormat('en-CA', {timeZone:'Asia/Shanghai', year:'numeric', month:'2-digit', day:'2-digit'}).format(new Date());
    const start = String(plan.planned_start_date || plan.start_date || today).slice(0,10);
    const elapsed = Math.floor((Date.parse(today) - Date.parse(start)) / 86400000);
    return Math.max(1, Math.min(total, Math.floor((Number.isFinite(elapsed) ? elapsed : 0) / 7) + 1));
}

// 只统计实际任务，超出的旧勾选和未来未安排的周不参与完成率。
function planTaskSummary(plan) {
    const weekly = Array.isArray(plan.weekly_plan) ? plan.weekly_plan : [];
    const completion = plan.task_completion || {};
    const current = planCurrentWeek(plan);
    const totalWeeks = Math.ceil((plan.duration_days || weekly.length * 7 || 7) / 7);
    let total = 0, completed = 0, detailedWeeks = 0, weekTotal = 0, weekCompleted = 0;
    weekly.forEach((week, index) => {
        const key = String(week.week || index + 1);
        const names = Array.isArray(week.tasks) ? week.tasks : [];
        const flags = Array.isArray(completion[key]) ? completion[key] : [];
        const done = names.reduce((sum, _, i) => sum + (flags[i] ? 1 : 0), 0);
        total += names.length;
        completed += done;
        if (names.length) detailedWeeks++;
        if (key === String(current)) { weekTotal = names.length; weekCompleted = done; }
    });
    return {total, completed, current, totalWeeks, detailedWeeks, weekTotal, weekCompleted,
        percentage: total ? Math.floor(completed / total * 100) : null,
        weekPercentage: weekTotal ? Math.floor(weekCompleted / weekTotal * 100) : null,
        hasUnplanned: detailedWeeks < totalWeeks};
}

function planTaskProgressLabel(plan) {
    const s = planTaskSummary(plan);
    if ((plan.duration_days || 0) > 14 || s.hasUnplanned) {
        return s.weekTotal ? `本周完成 ${s.weekCompleted}/${s.weekTotal} 项` : '本周任务尚未安排';
    }
    return s.total ? `任务完成 ${s.completed}/${s.total} 项` : '任务尚未安排';
}

function progressTaskScope(progress) {
    return progress.progress_scope === 'detailed_tasks' ? '已安排任务' : '已记录任务';
}

const planTaskSaveStates = new Map();

function planSaveKey(userId, subject, plan) {
    return JSON.stringify([userId, subject, plan.created_at || '']);
}

function setPlanSaveStatus(state, message) {
    const container = state.view?.container;
    if (!container || container._planSaveState !== state) return;
    const status = container.querySelector('[data-plan-save-status]');
    if (status) status.textContent = message;
    container.querySelectorAll('[data-plan-refine], .plan-sync-btn').forEach(button => {
        button.disabled = state.running;
    });
}

function queuePlanTaskSave(state, week, index, completed) {
    const key = `${week}_${index}`;
    state.pending.set(key, {week, index, completed});
    if (state.running) return state.done;
    state.running = true;
    // 同一份计划共享队列，快速连续勾选及取消只提交最终所需状态。
    state.done = drainPlanTaskSaves(state);
    return state.done;
}

async function drainPlanTaskSaves(state) {
    let warning = null;
    setPlanSaveStatus(state, '正在保存勾选…');
    while (state.pending.size) {
        const [key, task] = state.pending.entries().next().value;
        state.pending.delete(key);
        state.inFlight = task;
        try {
            const response = await apiRequest(`/api/plans/${encodeURIComponent(state.subject)}/tasks/toggle`, {
                method: 'POST', body: JSON.stringify({user_id: state.userId, week: task.week,
                    task_index: task.index, completed: task.completed, plan_created_at: state.createdAt})
            });
            if (response.status !== 'success' || !response.data) throw new Error(response.message || '保存失败');
            state.plan = response.data;
            if (response.sync_warning) warning = response.sync_warning;
        } catch (error) {
            // 响应丢失也可能已经保存，读取实际记录后再恢复页面。
            try {
                const response = await apiRequest(`/api/plans/${encodeURIComponent(state.subject)}?user_id=${encodeURIComponent(state.userId)}`);
                const latest = response?.data?.plan || response?.data;
                if (latest?.weekly_plan && latest.created_at === state.createdAt) state.plan = latest;
            } catch (_) { /* 保留最后一次确认成功的状态，提示用户重试。 */ }
            warning = `部分勾选未确认保存，请核对并重试：${error.message}`;
            showError(warning);
        }
        state.inFlight = null;
    }
    state.running = false;
    const view = state.view;
    if (view?.container._planSaveState === state && getCurrentUserId() === state.userId) {
        renderPlanDetail(view.container, state.subject, state.plan, view.options);
        setPlanSaveStatus(state, warning || '勾选已保存');
    }
    loadPlans();
    loadProgressPlanEntries();
    loadProgressOverview();
    updateDashboardData();
    if (document.getElementById('progressStats')) loadProgress();
}

function planPhases(plan) {
    if (Array.isArray(plan.stage_plan) && plan.stage_plan.length) return plan.stage_plan;
    // 旧计划也能折叠展示，保留原任务与完成记录。
    const days = plan.duration_days || plan.daily_plan?.length || 0;
    if (days <= 14) return [];
    const entries = plan.daily_plan || [];
    const phases = [];
    for (const entry of entries) {
        const title = (entry.topics || []).join('、') || '学习与实践';
        const last = phases[phases.length - 1];
        if (last && last.title === title) last.end_day = entry.day;
        else phases.push({title,start_day:entry.day,end_day:entry.day,objective:title,deliverable:(entry.deliverables||[]).join('、')});
    }
    if (phases.length <= 6) return phases;
    return ['基础与目标','跟做案例','独立实践','检验与总结'].map((title,idx)=>({
        title,start_day:Math.floor(idx*days/4)+1,end_day:Math.floor((idx+1)*days/4),objective:'按已有任务推进',deliverable:''
    }));
}

function renderPlanPhases(plan) {
    const phases = planPhases(plan), esc = escapePlanText;
    if (!phases.length) return '';
    return `<div class="plan-stage-overview"><h4>阶段概览</h4><ol>${phases.map(p=>`<li><strong>${esc(p.title)}</strong><span>第${esc(p.start_day)}～${esc(p.end_day)}天</span><p>${esc(p.objective)}${p.deliverable ? ` · 产出：${esc(p.deliverable)}` : ''}</p></li>`).join('')}</ol></div>`;
}

function renderPlanWeekGroups(plan, renderDay, focusWeek = null) {
    const daily = plan.daily_plan || [], days = plan.duration_days || daily.length, esc = escapePlanText;
    if (days <= 14) return daily.map(renderDay).join('');
    const current = planCurrentWeek(plan), detailed = plan.detailed_weeks || [...new Set(daily.map(d=>Math.floor((d.day-1)/7)+1))];
    const phases = planPhases(plan);
    let html = '<div class="plan-week-groups">';
    for (let week=1; week<=Math.ceil(days/7); week++) {
        const first=(week-1)*7+1,last=Math.min(week*7,days),entries=daily.filter(d=>d.day>=first&&d.day<=last);
        const phase=phases.find(p=>first>=p.start_day&&first<=p.end_day);
        const previousDone=plan.task_completion?.[String(week-1)] || [];
        const canGenerate=(week===1 || detailed.includes(week-1)) && (week<=current+1 || (previousDone.length>0 && previousDone.every(Boolean)));
        const action=(canGenerate && (!entries.length || !entries.some(d=>(d.task_refs||[]).some(r=>(plan.task_completion?.[r.week]||[])[r.task_index]))))
            ? `<button type="button" class="plan-choice-btn" data-plan-refine="${week}" data-plan-subject="${esc(plan.subject)}">${entries.length ? '根据进度调整本周' : '根据进度安排这周'}</button>` : '';
        html += `<details class="plan-week-group" data-plan-week="${week}"${week===(focusWeek || current) ? ' open' : ''}><summary>第${week}周 · 第${first}～${last}天${week===current ? ' · 本周' : ''}<span>${esc(phase?.title || (entries[0]?.topics||[]).join('、') || '后续安排')}</span></summary>`;
        if (entries.length) html += entries.map(renderDay).join('');
        else html += `<p class="plan-week-pending">${esc(phase?.objective || '按阶段方向推进')}。具体任务在临近时结合完成情况细化。</p>`;
        html += action + '</details>';
    }
    return html + '</div>';
}

document.addEventListener('click', async event => {
    const button = event.target.closest?.('button[data-plan-open], button[data-plan-refine]');
    if (!button || button.disabled) return;
    const subject=button.dataset.planSubject || button.dataset.planOpen;
    if (button.dataset.planOpen) {
        switchPage('plans');
        await viewPlanDetails({subject});
        return;
    }
    button.disabled=true;
    const text=button.textContent;
    button.textContent='正在结合进度安排…';
    try {
        const result=await apiRequest(`/api/plans/${encodeURIComponent(subject)}/weeks/${Number(button.dataset.planRefine)}/generate`,{
            method:'POST',body:JSON.stringify({user_id:getCurrentUserId()})
        }, 190000);
        if (result.status!=='success' || !result.data) throw new Error(result.message || '生成失败');
        const card=button.closest('.chat-plan');
        if (card) card.outerHTML=renderPlan(result.data,'');
        else renderPlanDetail(document.getElementById('planDetailPanel'),subject,result.data,{focusWeek:Number(button.dataset.planRefine)});
        loadPlans();
        loadProgressPlanEntries();
    } catch(error) { showError(error.message); }
    finally { button.disabled=false; button.textContent=text; }
});

// 事件委托让当前卡片和重开的历史卡片都能使用；自定义和修改条件只填入输入框。
document.addEventListener('click', event => {
    const button = event.target.closest?.('button[data-plan-message], button[data-plan-prefill]');
    if (!button || isSendingMessage) return;
    if (button.dataset.planPrefill) {
        switchPage('chat');
        prefillChatInput(button.dataset.planPrefill);
        return;
    }
    if (button.dataset.planMessage) {
        button.disabled = true;
        switchPage('chat');
        sendQuickMessage(button.dataset.planMessage);
        if (button.dataset.planAfter) prefillChatInput(button.dataset.planAfter);
    }
});

// 渲染学习计划
function renderPlan(plan, message) {
    if (!plan || !plan.weekly_plan) {
        return message || '学习计划为空';
    }

    const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));

    let html = '';
    if (message) html += `<div class="plan-intro">${message}</div>`;

    const subject = esc(plan.subject || '通用');
    const duration = esc(plan.requested_duration || plan.duration || '');
    const meta = [];
    if (plan.start_date) {
        const d = new Date(plan.start_date);
        if (!isNaN(d.getTime())) meta.push(`开始日期：${d.toLocaleDateString('zh-CN')}`);
    }
    const end = plan.planned_end_date || (plan.duration_days && plan.start_date ? (() => {
        const d = new Date(plan.start_date);
        d.setDate(d.getDate() + plan.duration_days - 1);
        return d.toLocaleDateString('zh-CN');
    })() : '');
    if (end) meta.push(`结束日期：${esc(end)}`);
    if (plan.goal) meta.push(`目标：${esc(plan.goal)}`);
    if (plan.daily_hours != null) meta.push(`每天：${esc(Math.round(plan.daily_hours * 60))} 分钟`);
    if (plan.requested_duration) meta.push(`周期：${esc(plan.requested_duration)}`);

    html += '<div class="chat-plan">';
    html += `<div class="plan-header"><div class="plan-title"><i class="fas fa-calendar-check"></i> ${subject} 学习计划${duration ? ` · ${duration}` : ''}</div>`;
    if (meta.length) html += `<div class="plan-meta">${meta.join('　·　')}</div>`;
    html += `<button type="button" class="plan-choice-btn plan-edit-conditions" data-plan-prefill="${esc(planConditionMessage(plan))}">修改条件</button>`;
    html += '</div>';

    if (plan.schedule_note) html += `<p class="plan-meta">${esc(plan.schedule_note)}</p>`;
    if (plan.outcome) html += `<p class="plan-meta">本期成果：${esc(plan.outcome)}</p>`;
    if (Array.isArray(plan.assumptions) && plan.assumptions.length) html += `<p class="plan-meta">${plan.assumptions.map(esc).join(' ')}</p>`;
    if (Array.isArray(plan.deferred_tasks) && plan.deferred_tasks.length) {
        html += `<details><summary>后续任务（${plan.deferred_tasks.length} 项）</summary><ul>${plan.deferred_tasks.map(t => `<li>${esc(t.task)}</li>`).join('')}</ul></details>`;
    }

    html += renderPlanPhases(plan);
    html += '<div class="plan-weeks">';
    const isDaily = Array.isArray(plan.daily_plan) && plan.daily_plan.length > 0;
    const entries = isDaily ? plan.daily_plan : plan.weekly_plan;
    const renderDay = item => {
        let html = '';
        const topics = (item.topics || []).map(esc);
        const tasks = (item.tasks || []).map(esc);
        const goals = (item.goals || []).map(esc);
        html += '<div class="plan-week">';
        html += `<div class="plan-week-label">第 ${esc(isDaily ? item.day : item.week)} ${isDaily ? '天' : '周'}${isDaily && item.date ? `<br><small>${esc(item.date)}</small>` : ''}</div>`;
        html += '<div class="plan-week-detail">';
        if (item.available_hours != null) html += `<div class="plan-row"><span class="plan-key">时长</span><span class="plan-val">预计 ${esc(isDaily ? Math.round(item.estimated_hours * 60) : item.estimated_hours)} / 可用 ${esc(isDaily ? Math.round(item.available_hours * 60) : item.available_hours)} ${isDaily ? '分钟' : '小时'}</span></div>`;
        if (topics.length) html += `<div class="plan-row"><span class="plan-key">主题</span><span class="plan-val">${topics.join('、')}</span></div>`;
        if (tasks.length) html += `<div class="plan-row"><span class="plan-key">任务</span><ul class="plan-task-list">${tasks.map(t => `<li>${t}</li>`).join('')}</ul></div>`;
        if ((item.deliverables || []).length) html += `<div class="plan-row"><span class="plan-key">产出</span><span class="plan-val">${item.deliverables.map(esc).join('、')}</span></div>`;
        if (goals.length && !isDaily) html += `<div class="plan-row"><span class="plan-key">目标</span><span class="plan-val">${goals.join('、')}</span></div>`;
        html += '</div></div>';
        return html;
    };
    html += isDaily ? renderPlanWeekGroups(plan,renderDay) : entries.map(renderDay).join('');
    html += '</div>';
    if ((plan.duration_days || entries.length)>14) html += `<div class="plan-summary-actions"><button type="button" class="plan-choice-btn" data-plan-open="${esc(plan.subject)}">查看完整计划</button><span>每周可根据已完成任务调整下一周。</span></div>`;
    html += '</div>';
    return html;
}

// 渲染学习资源
function renderResources(resources) {
    if (!resources || !resources.length) {
        return '未找到合适的资源';
    }
    
    let html = '<div class="chat-resource-header"><strong>为您推荐以下资源：</strong></div>';
    html += '<div class="chat-resource-grid">';
    
    resources.forEach(r => {
        const name = r.name || '资源';
        const type = r.type || '';
        const difficulty = r.difficulty || '';
        const description = (r.description || '').trim();
        const url = r.url;
        const sourceText = r.source === 'web_search' ? '网页推荐' : '本地精选';
        
        html += '<div class="chat-resource-card">';
        
        if (url) {
            html += `<div class="chat-resource-title"><strong><a href="${url}" target="_blank" rel="noopener noreferrer">${name}<span class="chat-resource-external">↗</span></a></strong></div>`;
        } else {
            html += `<div class="chat-resource-title"><strong>${name}</strong></div>`;
        }
        
        html += '<div class="chat-resource-meta">';
        if (type) html += `<span class="chat-resource-badge chat-resource-badge-type">${type}</span>`;
        if (difficulty && difficulty !== '通用') html += `<span class="chat-resource-badge chat-resource-badge-level">${difficulty}</span>`;
        html += `</div>`;
        
        if (description) {
            const briefDescription = description.length > 72 ? `${description.slice(0, 72)}...` : description;
            html += `<div class="chat-resource-desc">${briefDescription}</div>`;
        }

        html += `<div class="chat-resource-footer"><span class="chat-resource-source">${sourceText}</span>${url ? `<a class="chat-resource-link" href="${url}" target="_blank" rel="noopener noreferrer">查看资源</a>` : ''}</div>`;
        
        html += '</div>';
    });
    
    html += '</div>';
    return html;
}

// 渲染学习进度
function renderProgress(progress) {
    if (!progress || Object.keys(progress).length === 0) {
        return '<div>暂无学习进度记录</div>';
    }
    
    const percentage = progress.progress_percentage || 0;
    const completed = progress.completed_tasks || 0;
    const total = progress.total_tasks || 0;
    const studyTime = progress.study_time || 0;
    const subject = progress.subject || '未知主题';
    
    let html = '<div><strong>学习进度详情：</strong></div><div style="margin-top:12px">';
    html += `<div><strong>主题：</strong>${subject}</div>`;
    
    if (progress.message) {
        html += `<div style="margin:15px 0;padding:15px;background-color:rgba(255, 193, 7, 0.1);border:1px solid rgba(255, 193, 7, 0.3);border-radius:8px;color:var(--warning-color);">${progress.message}</div>`;
    }
    
    if (total > 0) {
        html += `<div style="margin:15px 0;">`;
        html += `<div style="margin-bottom:8px"><strong>${progressTaskScope(progress)}完成：</strong>${completed} / ${total} 已完成</div>`;
        html += `<div style="background-color:var(--border-color);border-radius:10px;overflow:hidden;height:24px">`;
        html += `<div style="background:linear-gradient(90deg, var(--primary-color), var(--secondary-color));height:100%;width:${percentage}%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:0.9em">${percentage}%</div>`;
        html += `</div>`;
        html += `</div>`;
    }
    
    if (studyTime > 0) {
        const hours = Math.floor(studyTime / 60);
        const minutes = studyTime % 60;
        html += `<div style="margin:15px 0;"><strong>学习时间：</strong>${hours > 0 ? hours + '小时' : ''}${minutes}分钟</div>`;
    }
    
    if (progress.notes && progress.notes.length > 0) {
        html += `<div style="margin-top:20px"><strong>学习笔记：</strong></div>`;
        html += `<ul style="margin-top:10px;padding-left:20px">`;
        progress.notes.slice(-5).forEach(note => {
            html += `<li style="margin-bottom:8px;color:var(--text-light)">${note.note || ''}</li>`;
        });
        html += `</ul>`;
    }
    
    if (completed > 0 && total > 0) {
        html += `<div style="margin-top:15px;padding:12px;background-color:rgba(40, 167, 69, 0.1);border:1px solid rgba(40, 167, 69, 0.3);border-radius:6px;color:var(--success-color);">✅ 学习进度已记录</div>`;
    }
    
    html += '</div>';
    return html;
}

// 更新仪表盘数据
function updateDashboardData() {
    const cards = document.querySelectorAll('.stat-item .card-value');
    cards.forEach(card => {
        card.innerHTML = '<span class="loading-spinner-small"></span>';
    });

    Promise.all([
        getActivePlansCount(),
        getResourcesCount(),
        getProgressData()
    ]).then(([plans, resources, {tasks, studyTime}]) => {
        const statCards = document.querySelectorAll('.stat-item .card-value');
        const setNum = (el, n) => {
            if (!el) return;
            el.textContent = n;
            animateValue(el, 0, n, 800);
        };
        setNum(statCards[0], plans);
        setNum(statCards[1], resources);
        setNum(statCards[2], tasks);
        const timeCard = statCards[3];
        if (timeCard) {
            const hours = Math.floor(studyTime / 60);
            const num = hours > 0 ? hours : studyTime;
            const unit = hours > 0 ? '小时' : '分钟';
            timeCard.innerHTML = `<span class="val-num">0</span><span class="val-unit">${unit}</span>`;
            animateValue(timeCard.querySelector('.val-num'), 0, num, 800);
        }

        // 欢迎区：用真实计划数决定主按钮，副标题用真实待复习数（由 loadPendingReviews 回填）
        updateWelcome(plans);
        updateGreeting();
        loadWeekChart();
        bindDashboardClickable();
        loadDailyConcept();
        loadRecentActivities();
        loadPendingReviews();
        loadProgressOverview();
        loadRecentQuizzes();
        loadRecommendations();
    }).catch(error => {
        console.error('加载仪表盘数据失败:', error);
        document.querySelectorAll('.stat-item .card-value').forEach((card, index) => {
            if (index === 3) {
                card.innerHTML = '<span class="val-num">0</span><span class="val-unit">分钟</span>';
            } else {
                card.textContent = '0';
            }
        });
        updateWelcome(0);
    });
}

// 欢迎区：仅使用真实数据（计划数、待复习数），无数据用普通文案
function updateWelcome(plansCount) {
    const primary = document.getElementById('welcomePrimary');
    const subtitle = document.getElementById('welcomeSubtitle');
    if (primary) {
        const hasPlan = Number(plansCount) > 0;
        primary.innerHTML = hasPlan
            ? '<i class="fas fa-arrow-right"></i> 继续学习'
            : '<i class="fas fa-plus"></i> 创建学习计划';
        primary.dataset.hasPlan = hasPlan ? '1' : '0';
    }
    if (subtitle) {
        const due = Number(window.__dueReviewCount || 0);
        subtitle.textContent = due > 0
            ? `今天有 ${due} 项内容等待复习，完成后再继续新课程。`
            : '先从制定一个学习计划开始吧。';
    }
    if (typeof updateWelcomeTitle === 'function') updateWelcomeTitle();
}

// 主按钮：有计划进入第一个计划详情，无计划进入计划页创建
function goContinueLearning() {
    const btn = document.getElementById('welcomePrimary');
    if (btn && btn.dataset.hasPlan === '1') {
        switchPage('plans');
        if (typeof loadPlans === 'function') loadPlans();
        return;
    }
    switchPage('plans');
    setTimeout(() => {
        if (typeof createNewPlan === 'function') createNewPlan();
    }, 200);
}

// 推荐折叠：默认 3 条，其余展开
let _recExpanded = false;
function toggleRecommendations() {
    _recExpanded = !_recExpanded;
    const list = document.getElementById('recommendations');
    const btn = document.getElementById('recToggle');
    if (!list || !btn) return;
    list.querySelectorAll('.rec-item[data-extra="1"]').forEach(el => {
        el.hidden = !_recExpanded;
        el.style.display = _recExpanded ? 'flex' : 'none';
    });
    btn.innerHTML = _recExpanded
        ? '收起 <i class="fas fa-chevron-up"></i>'
        : '展开更多 <i class="fas fa-chevron-down"></i>';
}

// 右侧动态面板 Tab 切换：待复习 / 最近活动 / 最近测验
function switchSideTab(name) {
    document.querySelectorAll('.side-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === name);
        btn.setAttribute('aria-selected', btn.dataset.tab === name ? 'true' : 'false');
    });
    document.querySelectorAll('.side-pane').forEach(pane => {
        pane.hidden = pane.dataset.pane !== name;
        pane.classList.toggle('active', pane.dataset.pane === name);
    });
    syncSideFooter(name);
}

// 底部链接跟随当前 Tab，避免"最近活动"下还写着"查看全部复习"
function syncSideFooter(name) {
    const link = document.getElementById('sideFooterLink');
    if (!link) return;
    const map = {
        review: { text: '查看全部复习', page: 'review' },
        activity: { text: '查看学习进度', page: 'progress' },
        quiz: { text: '查看全部测验', page: 'quiz' },
    };
    const cfg = map[name] || map.review;
    link.innerHTML = `${cfg.text} <i class="fas fa-chevron-right"></i>`;
    link.setAttribute('onclick', `switchPage('${cfg.page}')`);
}

async function loadRecommendations() {
    const container = document.getElementById('recommendations');
    if (!container) return;
    container.innerHTML = '<div class="loading-spinner"></div>';
    _recExpanded = false;
    try {
        const resp = await apiRequest(`/api/recommendations?user_id=${encodeURIComponent(getCurrentUserId())}`);
        if (!resp || resp.status !== 'success' || !resp.data) {
            container.innerHTML = '<p class="empty-state">暂无推荐</p>';
            return;
        }
        const actions = Array.isArray(resp.data.actions) ? resp.data.actions : [];
        const subjects = Array.isArray(resp.data.top_subjects) ? resp.data.top_subjects : [];
        if (!actions.length) {
            container.innerHTML = '<p class="empty-state">暂无推荐</p>';
            return;
        }

        // 合并规则：仅当主题相同且任务目标相同（同 type + 同 subject + 同标题）才合并
        const merged = [];
        const seen = new Set();
        actions.forEach(a => {
            const key = `${a.type || ''}|${a.subject || ''}|${a.title || ''}`;
            if (seen.has(key)) return;
            seen.add(key);
            merged.push(a);
        });

        const metaOf = a => {
            const s = subjects.find(x => x.subject === (a.subject || '')) || {};
            if (a.type === 'flashcards') {
                const n = s.due_flashcards || 0;
                return {
                    cls: 'rec-flashcards', icon: 'fa-clone', tag: '闪卡', cta: '复习闪卡',
                    reason: n > 0 ? `${a.subject || '当前'}有 ${n} 张到期闪卡` : ''
                };
            }
            if (a.type === 'quiz') {
                const avg = s.quiz_average || 0;
                return {
                    cls: 'rec-quiz', icon: 'fa-question-circle', tag: '测验', cta: '开始测验',
                    reason: avg > 0 ? `${a.subject || '当前'}最近测验平均 ${avg.toFixed(0)} 分` : ''
                };
            }
            const pct = s.progress_percentage || 0;
            return {
                cls: 'rec-plan', icon: 'fa-calendar-check', tag: '计划', cta: '查看任务',
                reason: s.has_plan ? `${a.subject || '当前'}计划完成 ${pct}%` : ''
            };
        };

        const html = merged.map((a, idx) => {
            const m = metaOf(a);
            // 标题统一成「主题 · 动作」格式：同类型不同科目的卡不再看起来像重复
            const rawTitle = (a.title || '推荐任务').replace(/\s+/g, ' ').trim();
            const subjectLabel = (a.subject || '通用').trim();
            const title = a.type === 'flashcards' ? `${subjectLabel} · 复习闪卡`
                : a.type === 'quiz' ? `${subjectLabel} · 知识测验`
                : rawTitle;
            const reason = m.reason ? `<div class="rec-reason"><i class="fas fa-clock"></i> ${m.reason}</div>` : '';
            const extra = idx >= 3 ? ' data-extra="1" hidden style="display:none"' : '';
            return `
                <div class="rec-item ${m.cls}"${extra}>
                    <div class="rec-body">
                        <div class="rec-top">
                            <span class="rec-icon"><i class="fas ${m.icon}"></i></span>
                            <span class="rec-title" title="${title}">${title}</span>
                        </div>
                        <div class="rec-meta"><i class="fas fa-layer-group"></i> ${subjectLabel} · ${m.tag}</div>
                        ${reason}
                        <div class="rec-foot"><i class="fas fa-clock"></i> 待处理</div>
                    </div>
                    <div class="rec-actions">
                        <button class="rec-cta" data-rec-index="${idx}">${m.cta}</button>
                        <button class="rec-close" onclick="removeRecommendation(this)" title="移除此推荐" aria-label="移除此推荐"><i class="fas fa-times"></i></button>
                    </div>
                </div>
            `;
        }).join('');
        container.innerHTML = html;

        const toggle = document.getElementById('recToggle');
        if (toggle) {
            const needToggle = merged.length > 3;
            toggle.hidden = !needToggle;
            toggle.style.display = needToggle ? '' : 'none';
            _recExpanded = false;
            toggle.innerHTML = '展开更多 <i class="fas fa-chevron-down"></i>';
        }

        container.querySelectorAll('button[data-rec-index]').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = Number(btn.getAttribute('data-rec-index'));
                runRecommendationAction(merged[idx]);
            });
        });
    } catch (e) {
        container.innerHTML = '<p class="error-state">加载推荐失败</p>';
    }
}

function removeRecommendation(btn) {
    const item = btn.closest('.rec-item');
    if (item) item.remove();
    const container = document.getElementById('recommendations');
    if (container && !container.querySelector('.rec-item')) {
        container.innerHTML = '<p class="empty-state">暂无推荐</p>';
    }
}

function runRecommendationAction(action) {
    if (!action) return;
    const type = action.type;
    const subject = action.subject || '';
    if (type === 'flashcards') {
        switchPage('flashcards');
        setTimeout(() => loadFlashcards(), 50);
        return;
    }
    if (type === 'quiz') {
        startQuizForSubject(encodeURIComponent(subject || '通用'));
        return;
    }
    if (type === 'plan') {
        switchPage('plans');
        loadPlans();
        setTimeout(() => {
            viewPlanDetails({ subject: subject });
        }, 300);
        return;
    }
    switchPage('dashboard');
}

// 加载最近活动
async function loadRecentActivities() {
    const container = document.getElementById('recentActivities');
    if (!container) return;
    
    try {
        // 获取最近的问答历史
        const qaData = await apiRequest(`/api/qa/history?user_id=${getCurrentUserId()}&limit=5`);
        const activities = [];
        
        if (qaData.history && Array.isArray(qaData.history)) {
            qaData.history.slice(0, 5).forEach(item => {
                activities.push({
                    type: 'qa',
                    icon: 'fa-comment',
                    title: item.question || '提问',
                    time: item.timestamp || new Date().toISOString(),
                    subject: item.subject || '通用',
                    originalQuestion: item.question || '提问',  // 保存原始问题用于删除
                    originalTimestamp: item.timestamp || new Date().toISOString()  // 保存原始时间戳用于删除
                });
            });
        }
        
        // 获取最近的学习进度更新
        const progressData = await apiRequest('/api/progress', {
            method: 'POST',
            body: JSON.stringify({
                action: 'get_all',
                user_id: getCurrentUserId()
            })
        });
        
        if (progressData.progress && typeof progressData.progress === 'object') {
            for (const subject in progressData.progress) {
                const progress = progressData.progress[subject];
                if (progress && progress.last_updated) {
                    activities.push({
                        type: 'progress',
                        icon: 'fa-chart-line',
                        title: `更新了${subject}的学习进度`,
                        time: progress.last_updated,
                        subject: subject
                    });
                }
            }
        }
        
        // 按时间排序
        activities.sort((a, b) => new Date(b.time) - new Date(a.time));
        
        if (activities.length === 0) {
            container.innerHTML = '<p class="empty-state">暂无最近活动</p>';
            return;
        }
        
        let html = '';
        activities.slice(0, 5).forEach((activity, index) => {
            // 统一日期格式：MM-DD HH:mm
            const d = new Date(activity.time);
            const time = `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
            // 删除按钮沿用 .activity-delete-btn 样式（hover/focus 由 CSS 控制）
            const deleteBtn = activity.type === 'qa' ? `
                <button class="activity-delete-btn" 
                        data-timestamp="${activity.originalTimestamp || activity.time}"
                        data-question="${(activity.originalQuestion || activity.title).replace(/"/g, '&quot;').replace(/'/g, '&#39;')}"
                        data-type="qa"
                        onclick="deleteActivity(this)" 
                        title="删除此活动" aria-label="删除此活动">
                    <i class="fas fa-times"></i>
                </button>
            ` : activity.type === 'progress' ? `
                <button class="activity-delete-btn" 
                        data-timestamp="${activity.time}"
                        data-subject="${activity.subject.replace(/"/g, '&quot;').replace(/'/g, '&#39;')}"
                        data-type="progress"
                        onclick="deleteActivity(this)" 
                        title="删除此活动" aria-label="删除此活动">
                    <i class="fas fa-times"></i>
                </button>
            ` : '';
            html += `
                <div class="activity-item">
                    <div class="activity-icon">
                        <i class="fas ${activity.icon}"></i>
                    </div>
                    <div class="activity-content">
                        <div class="activity-title">${activity.title}</div>
                        <div class="activity-meta">
                            <span class="activity-subject">${activity.subject}</span>
                            <span class="activity-time">${time}</span>
                        </div>
                    </div>
                    ${deleteBtn}
                </div>
            `;
        });
        container.innerHTML = html;
    } catch (error) {
        console.error('加载最近活动失败:', error);
        container.innerHTML = '<p class="empty-state">加载失败</p>';
    }
}

// 删除最近活动
async function deleteActivity(button) {
    const timestamp = button.getAttribute('data-timestamp');
    const question = button.getAttribute('data-question');
    const subject = button.getAttribute('data-subject');
    const type = button.getAttribute('data-type');
    
    if (!timestamp || !type) {
        alert('缺少必要信息，无法删除');
        return;
    }
    
    if (!confirm('确定要删除这条活动记录吗？')) {
        return;
    }
    
    // 禁用按钮，防止重复点击
    button.disabled = true;
    button.style.opacity = '0.5';
    
    try {
        let result;
        if (type === 'qa') {
            // 删除问答记录
            result = await apiRequest('/api/qa/history', {
                method: 'DELETE',
                body: JSON.stringify({
                    user_id: getCurrentUserId(),
                    timestamp: timestamp,
                    question: question || null
                })
            });
        } else if (type === 'progress') {
            // 删除学习进度记录
            if (!subject) {
                alert('缺少学科信息，无法删除学习进度记录');
                button.disabled = false;
                button.style.opacity = '1';
                return;
            }
            result = await apiRequest('/api/progress/' + subject, {
                method: 'DELETE',
                body: JSON.stringify({
                    user_id: getCurrentUserId(),
                    timestamp: timestamp
                })
            });
        }
        
        if (result && result.status === 'success') {
            // 重新加载最近活动
            loadRecentActivities();
        } else {
            alert('删除失败: ' + (result?.message || '未知错误'));
            button.disabled = false;
            button.style.opacity = '1';
        }
    } catch (error) {
        console.error('删除活动失败:', error);
        alert('删除失败，请稍后重试');
        button.disabled = false;
        button.style.opacity = '1';
    }
}


// 闪卡功能实现
let currentFlashcard = null;
let flashcardsMode = localStorage.getItem('flashcards_mode') || 'due';
let flashcardsAllowReview = true;

function setFlashcardsMode(mode) {
    const m = (mode || 'due').toLowerCase() === 'all' ? 'all' : 'due';
    flashcardsMode = m;
    localStorage.setItem('flashcards_mode', m);
    updateFlashcardsModeButtons();
    loadFlashcards();
}

function updateFlashcardsModeButtons() {
    const dueBtn = document.getElementById('flashcardsModeDue');
    const allBtn = document.getElementById('flashcardsModeAll');
    if (dueBtn && allBtn) {
        if (flashcardsMode === 'all') {
            dueBtn.className = 'secondary-btn';
            allBtn.className = 'primary-btn';
        } else {
            dueBtn.className = 'primary-btn';
            allBtn.className = 'secondary-btn';
        }
        dueBtn.style.padding = '8px 14px';
        allBtn.style.padding = '8px 14px';
    }
}

function formatNextReview(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '—';
    return d.toLocaleString('zh-CN');
}

// 加载闪卡页面
async function loadFlashcards() {
    const reviewArea = document.getElementById('flashcardReviewArea');
    const noCardsState = document.getElementById('noCardsState');
    const listContainer = document.getElementById('flashcardsList');
    
    reviewArea.style.display = 'none';
    noCardsState.style.display = 'none';
    if (listContainer) listContainer.style.display = 'none';
    updateFlashcardsModeButtons();
    
    try {
        const data = await apiRequest(`/api/flashcards?user_id=${getCurrentUserId()}&mode=${encodeURIComponent(flashcardsMode)}`);
        const cards = (data && data.status === 'success' && Array.isArray(data.cards)) ? data.cards : [];

        if (flashcardsMode === 'all') {
            const p = noCardsState ? noCardsState.querySelector('p') : null;
            const h = noCardsState ? noCardsState.querySelector('h3') : null;
            if (p) p.textContent = '当前暂无闪卡，请先创建';
            if (h) h.textContent = '还没有闪卡';

            if (!cards.length) {
                noCardsState.style.display = 'block';
                return;
            }

            flashcardsAllowReview = false;
            currentFlashcard = cards[0];
            showFlashcard(currentFlashcard);
            reviewArea.style.display = 'block';

            if (listContainer) {
                listContainer.style.display = 'block';
                listContainer.innerHTML = `
                    <div style="margin:10px 0 12px;color:var(--text-light);font-size:0.9em;">共 ${cards.length} 张（切换到“待复习”可进行正式复习打分）</div>
                    ${cards.map((c, idx) => {
                        const due = (() => {
                            const d = new Date(c.next_review || '');
                            return !Number.isNaN(d.getTime()) && d <= new Date();
                        })();
                        const badge = due ? `<span style="background:rgba(220,53,69,0.12);color:var(--danger-color);padding:2px 8px;border-radius:999px;font-size:0.8em;">待复习</span>` : `<span style="background:rgba(40,167,69,0.12);color:var(--success-color);padding:2px 8px;border-radius:999px;font-size:0.8em;">未到期</span>`;
                        const title = (c.front || '').slice(0, 28) + ((c.front || '').length > 28 ? '…' : '');
                        return `
                            <div data-card-index="${idx}" style="background:var(--card-bg);border-radius:12px;padding:12px 14px;margin:10px 0;box-shadow:0 4px 6px rgba(0,0,0,0.05);cursor:pointer;">
                                <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start;">
                                    <div style="font-weight:700;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${title || '（空）'}</div>
                                    <div style="display:flex;align-items:center;gap:8px;flex:0 0 auto;">
                                        ${badge}
                                        <button class="link-danger" data-card-id="${c.id}" onclick="event.stopPropagation(); deleteFlashcard('${c.id}')">删除</button>
                                    </div>
                                </div>
                                <div style="margin-top:6px;color:var(--text-light);font-size:0.85em;">主题：${c.subject || '通用'} · 下次复习：${formatNextReview(c.next_review)}</div>
                            </div>
                        `;
                    }).join('')}
                `;
                listContainer.querySelectorAll('[data-card-index]').forEach(el => {
                    el.addEventListener('click', () => {
                        const idx = Number(el.getAttribute('data-card-index'));
                        const c = cards[idx];
                        if (!c) return;
                        currentFlashcard = c;
                        showFlashcard(c);
                        reviewArea.scrollIntoView({ behavior: 'smooth' });
                    });
                });
            }
            return;
        }

        const p = noCardsState ? noCardsState.querySelector('p') : null;
        const h = noCardsState ? noCardsState.querySelector('h3') : null;
        if (p) p.textContent = '当前没有需要复习的闪卡';
        if (h) h.textContent = '太棒了！';

        flashcardsAllowReview = true;
        if (cards.length > 0) {
            currentFlashcard = cards[0];
            showFlashcard(currentFlashcard);
            reviewArea.style.display = 'block';
        } else {
            noCardsState.style.display = 'block';
        }
    } catch (error) {
        showError('加载闪卡失败');
    }
}

// 显示闪卡
function showFlashcard(card) {
    const cardElement = document.querySelector('.flashcard');
    const contentElement = document.getElementById('cardContent');
    const controls = document.getElementById('cardControls');
    const meta = document.getElementById('cardMeta');
    
    // 重置状态
    cardElement.classList.remove('flipped');
    cardElement.style.transform = 'rotateY(0deg)';
    contentElement.style.transform = 'rotateY(0deg)';
    contentElement.textContent = card.front;
    controls.style.display = 'none';
    if (meta) {
        meta.innerHTML = `<span>主题：${card.subject || '通用'}</span><span>下次复习：${formatNextReview(card.next_review)}</span>`;
    }
    
    // 点击翻转逻辑
    cardElement.onclick = function() {
        if (!cardElement.classList.contains('flipped')) {
            flipCard();
        }
    };
}

function flipCard() {
    const cardElement = document.querySelector('.flashcard');
    const contentElement = document.getElementById('cardContent');
    const controls = document.getElementById('cardControls');
    if (!cardElement || !contentElement || !controls) return;
    if (!currentFlashcard) return;
    if (cardElement.classList.contains('flipped')) return;

    cardElement.classList.add('flipped');
    contentElement.textContent = currentFlashcard.back || '';
    if (flashcardsAllowReview) {
        controls.style.display = 'flex';
    } else {
        controls.style.display = 'none';
    }
}

// 提交复习结果
async function reviewCardResult(success) {
    if (!currentFlashcard) return;
    
    try {
        await apiRequest(`/api/flashcards/${currentFlashcard.id}/review`, {
            method: 'POST',
            body: JSON.stringify({
                user_id: getCurrentUserId(),
                success: success
            })
        });
        
        // 加载下一张
        loadFlashcards();
    } catch (error) {
        showError('操作失败');
    }
}

// 显示创建闪卡模态框
function showCreateFlashcardModal() {
    document.getElementById('createFlashcardModal').style.display = 'block';
}

// 关闭创建闪卡模态框
function closeCreateFlashcardModal() {
    document.getElementById('createFlashcardModal').style.display = 'none';
    document.getElementById('cardSubject').value = '';
    document.getElementById('cardFront').value = '';
    document.getElementById('cardBack').value = '';
}

// 提交创建闪卡
async function submitCreateFlashcard() {
    const subject = document.getElementById('cardSubject').value.trim();
    const front = document.getElementById('cardFront').value.trim();
    const back = document.getElementById('cardBack').value.trim();
    
    if (front && back) {
        try {
            await apiRequest('/api/flashcards', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: getCurrentUserId(),
                    subject: subject || '通用',
                    front: front,
                    back: back
                })
            });
            
            showSuccess('闪卡创建成功');
            closeCreateFlashcardModal();
            loadFlashcards();
        } catch (error) {
            showError('创建失败');
        }
    } else {
        showError('请填写完整信息');
    }
}

// 思维导图生成
async function generateMindMap() {
    const input = document.getElementById('mindmapInput').value.trim();
    const container = document.getElementById('mindmapContainer');
    
    if (!input) {
        showError('请输入主题');
        return;
    }
    
    container.innerHTML = '<div class="loading-spinner"></div> 正在生成思维导图...';
    
    try {
        await ensureMermaidLoaded();
        const data = await apiRequest('/api/mindmap', {
            method: 'POST',
            body: JSON.stringify({
                user_id: getCurrentUserId(),
                topic: input
            })
        });
        
        if (data.status === 'success') {
            container.innerHTML = `<div class="mermaid">${data.mermaid_code}</div>`;
            if (typeof mermaid.run === 'function') {
                mermaid.run({ nodes: document.querySelectorAll('.mermaid') });
            } else {
                mermaid.init(undefined, document.querySelectorAll('.mermaid'));
            }
        } else {
            container.innerHTML = '<div class="error-state">生成失败</div>';
        }
    } catch (error) {
        container.innerHTML = '<div class="error-state">生成失败: ' + error.message + '</div>';
    }
}

// 加载待复习内容
async function loadPendingReviews() {
    const container = document.getElementById('pendingReviews');
    if (!container) return;
    
    try {
        const data = await apiRequest(`/api/review/due?user_id=${getCurrentUserId()}`);
        if (data.status === 'success') {
            // 统一日期格式：MM-DD HH:mm
            const fmt = v => {
                const d = new Date(v);
                return `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
            };
            const overdue = (data.overdue || []).filter(r => r && r.status !== 'mastered');
            // 用真实待复习数回填欢迎区副标题与右侧计数
            window.__dueReviewCount = overdue.length;
            const countEl = document.getElementById('reviewCount');
            if (countEl) countEl.textContent = overdue.length;
            const subEl = document.getElementById('welcomeSubtitle');
            if (subEl && overdue.length > 0) {
                subEl.textContent = `今天有 ${overdue.length} 项内容等待复习，完成后再继续新课程。`;
            }
            if (overdue.length > 0) {
                const now = Date.now();
                const cards = overdue.map(review => {
                    const t = review.next_review_at ? new Date(review.next_review_at) : null;
                    const isOverdue = t && t.getTime() < now;
                    const timeStr = t ? fmt(t) : '待安排';
                    // 缺少描述时省略描述区，不突出显示"无内容描述"
                    const desc = review.content
                        ? `<p class="review-preview-content">${review.content}</p>` : '';
                    return `
                        <div class="review-preview-card">
                            <div class="review-preview-header">
                                <span class="review-icon"><i class="fas fa-redo-alt"></i></span>
                                <h4>${review.subject || '复习内容'}</h4>
                                <span class="review-badge${isOverdue ? ' overdue' : ''}">${isOverdue ? '已逾期' : '待复习'}</span>
                            </div>
                            ${desc}
                            <div class="review-preview-footer">
                                <span><i class="fas fa-clock"></i> ${timeStr}</span>
                                <button class="primary-btn" onclick="switchPage('review')">开始复习</button>
                            </div>
                        </div>`;
                }).join('');
                container.innerHTML = `<div class="review-due-title"><i class="fas fa-bell"></i> 你有 ${overdue.length} 项复习到期</div>${cards}`;
            } else if (data.next) {
                const nextTime = fmt(data.next.next_review_at);
                container.innerHTML = `
                    <div class="review-preview-card">
                        <div class="review-preview-header">
                            <h4>暂无到期复习</h4>
                            <span class="review-badge">等待中</span>
                        </div>
                        <div class="review-preview-footer">
                            <span><i class="fas fa-clock"></i> 下次时间：${nextTime}</span>
                        </div>
                    </div>`;
            } else {
                container.innerHTML = '<p class="empty-state">暂无待复习内容</p>';
            }
        } else {
            container.innerHTML = '<p class="empty-state">暂无待复习内容</p>';
        }
    } catch (error) {
        console.error('加载待复习内容失败:', error);
        container.innerHTML = '<p class="empty-state">暂无待复习内容</p>';
    }
}

// 加载学习进度概览
async function loadProgressOverview() {
    const container = document.getElementById('progressOverview');
    if (!container) return;
    
    try {
        const data = await apiRequest('/api/progress', {
            method: 'POST',
            body: JSON.stringify({
                action: 'get_all',
                user_id: getCurrentUserId()
            })
        });
        
        if (data.progress && typeof data.progress === 'object') {
            const subjects = Object.keys(data.progress);
            
            if (subjects.length === 0) {
                container.innerHTML = '<p class="empty-state">暂无学习进度</p>';
                return;
            }
            
            let html = '<div class="progress-list">';
            subjects.slice(0, 5).forEach(subject => {
                const progress = data.progress[subject];
                const displaySubject = progress.subject || subject.replace(`${getCurrentUserId()}_`, '');
                const percentage = progress.progress_percentage || 0;
                const completed = progress.completed_tasks || 0;
                const total = progress.total_tasks || 0;
                
                // 任务点阵：任务数不多时，一格代表一个任务，比百分比直观
                const MAX_SEGMENTS = 24;
                let segsHtml;
                if (total > 0 && total <= MAX_SEGMENTS) {
                    segsHtml = '<div class="progress-segs">' +
                        Array.from({ length: total }, (_, i) =>
                            `<span class="seg${i < completed ? ' on' : ''}"></span>`).join('') +
                        '</div>';
                } else {
                    segsHtml = `<div class="progress-segs is-bar"><span class="seg-bar"><i style="width:${percentage}%"></i></span></div>`;
                }
                const remain = Math.max(0, total - completed);
                html += `
                    <div class="progress-item-card">
                        <div class="progress-row">
                            <span class="progress-icon"><i class="fas fa-book-open"></i></span>
                            <span class="progress-subject">${displaySubject}</span>
                            ${segsHtml}
                            <span class="progress-frac"><b>${completed}</b>/${total}<em>${percentage}%</em></span>
                        </div>
                        <div class="progress-item-footer">
                            <span>${progressTaskScope(progress)}：${remain > 0 ? `还剩 ${remain} 项` : total ? '当前任务已完成' : '暂无任务'}${progress.plan_progress_summary?.has_unplanned_weeks ? ' · 后续周尚未安排' : ''}</span>
                        </div>
                    </div>
                `;
            });
            // 总体汇总：填满卡片底部，同时给出全局视角
            let sumTotal = 0, sumDone = 0, sumMinutes = 0;
            subjects.forEach(k => {
                const p = data.progress[k] || {};
                sumTotal += Number(p.total_tasks || 0);
                sumDone += Number(p.completed_tasks || 0);
                sumMinutes += Number(p.study_time || 0);
            });
            const sumPct = sumTotal > 0 ? Math.round((sumDone / sumTotal) * 100) : 0;
            const sumHours = (sumMinutes / 60).toFixed(1);
            html += `
                <div class="progress-overall">
                    <div class="po-metric"><span>已记录任务完成率</span><b>${sumPct}%</b></div>
                    <div class="po-metric"><span>已完成任务</span><b>${sumDone}/${sumTotal}</b></div>
                    <div class="po-metric"><span>累计学习</span><b>${sumHours} 小时</b></div>
                </div>
            `;
            html += '</div>';

            if (subjects.length > 5) {
                html += `<button class="secondary-btn" onclick="switchPage('progress')" style="margin-top: 10px; width: 100%;">查看全部进度</button>`;
            }
            
            container.innerHTML = html;
        } else {
            container.innerHTML = '<p class="empty-state">暂无学习进度</p>';
        }
    } catch (error) {
        console.error('加载进度概览失败:', error);
        container.innerHTML = '<p class="empty-state">加载失败</p>';
    }
}

// 加载最近测验
async function loadRecentQuizzes() {
    const container = document.getElementById('recentQuizzes');
    if (!container) return;
    
    try {
        const data = await apiRequest(`/api/quiz/history?user_id=${getCurrentUserId()}&limit=5`);
        
        if (data.status === 'success' && data.history && data.history.length > 0) {
            let html = '<div class="quiz-list">';
            data.history.forEach(quiz => {
                const ts = quiz.timestamp || quiz.created_at || quiz.completed_at || new Date().toISOString();
                const d = new Date(ts);
                const date = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
                const scoreVal = quiz.score;
                const score = scoreVal !== undefined && scoreVal !== null ? `${scoreVal}%` : '未评分';
                const scoreClass = scoreVal >= 80 ? 'success' : scoreVal >= 60 ? 'warning' : 'danger';
                let title = (quiz.title || (quiz.subject ? `测验：${quiz.subject}` : '测验')).replace(/\s+/g, ' ').trim();
                // 难度标签转中文，避免 easy/normal/hard 直接暴露
                const diffMap = { easy: '简单', normal: '普通', medium: '中等', hard: '困难' };
                const diff = diffMap[String(quiz.difficulty || '').toLowerCase()];
                if (diff && !title.includes(diff)) title = `${title} · ${diff}`;
                
                html += `
                    <div class="quiz-item-card" data-quiz-id="${quiz.quiz_id}">
                        <div class="quiz-item-icon">
                            <i class="fas fa-file-alt"></i>
                        </div>
                        <div class="quiz-item-content" onclick="viewQuizDetail('${quiz.quiz_id}')">
                            <div class="quiz-item-title">${title}</div>
                            <div class="quiz-item-date">${date}</div>
                        </div>
                        <div class="quiz-item-score ${scoreClass}">${score}</div>
                        <button class="quiz-item-delete" onclick="deleteQuiz('${quiz.quiz_id}')" title="删除">
                            <i class="fas fa-trash-alt"></i>
                        </button>
                    </div>
                `;
            });
            html += '</div>';
            container.innerHTML = html;
        } else {
            container.innerHTML = '<p class="empty-state">暂无测验记录</p>';
        }
    } catch (error) {
        console.error('加载最近测验失败:', error);
        container.innerHTML = '<p class="empty-state">加载失败</p>';
    }
}

// 删除测验
async function deleteQuiz(quizId) {
    if (!confirm('确定要删除这个测验记录吗？')) return;
    
    try {
        const data = await apiRequest('/api/quiz/delete', {
            method: 'POST',
            body: JSON.stringify({
                user_id: getCurrentUserId(),
                quiz_id: quizId
            })
        });
        
        if (data.status === 'success') {
            const card = document.querySelector(`[data-quiz-id="${quizId}"]`);
            if (card) card.remove();

            const historyPage = document.getElementById('quizHistory');
            const inHistory = historyPage && historyPage.style.display !== 'none';
            if (inHistory && typeof viewQuizHistory === 'function') {
                viewQuizHistory();
            } else if (typeof loadRecentQuizzes === 'function') {
                loadRecentQuizzes();
            }
        } else {
            alert(data.message || '删除失败');
        }
    } catch (error) {
        console.error('删除测验失败:', error);
        alert('删除失败，请重试');
    }
}

// 查看测验详情
function viewQuizDetail(quizId) {
    // 跳转到测验页面并加载详情
    switchPage('quiz');
    setTimeout(() => {
        loadQuizDetail(quizId);
    }, 100);
}

// 获取活跃计划数量
async function getActivePlansCount() {
    try {
        const data = await apiRequest('/api/plan', {
            method: 'POST',
            body: JSON.stringify({
                action: 'get_all',
                user_id: getCurrentUserId()
            })
        });
        
        if (data.plans && Array.isArray(data.plans)) {
            const activePlans = data.plans.filter(plan => plan.status === 'active');
            return activePlans.length;
        }
        return 0;
    } catch (error) {
        console.error('获取计划数量失败:', error);
        return 0;
    }
}

// 获取学习资源数量
async function getResourcesCount() {
    try {
        const data = await apiRequest('/api/resources', {
            method: 'POST',
            body: JSON.stringify({
                user_id: getCurrentUserId(),
                subject: 'all',
                count: 100
            })
        });
        
        if (data.resources && Array.isArray(data.resources)) {
            return data.resources.length;
        }
        return 0;
    } catch (error) {
        console.error('获取资源数量失败:', error);
        return 0;
    }
}

// 获取完成任务数量
async function getCompletedTasksCount() {
    try {
        const data = await apiRequest('/api/progress', {
            method: 'POST',
            body: JSON.stringify({
                action: 'get_all',
                user_id: getCurrentUserId()
            })
        });
        
        let totalCompleted = 0;
        if (data.status === 'success' && data.progress) {
            for (const subject in data.progress) {
                const progress = data.progress[subject];
                if (progress) {
                    totalCompleted += progress.completed_tasks || 0;
                }
            }
        }
        return totalCompleted;
    } catch (error) {
        console.error('获取任务数量失败:', error);
        return 0;
    }
}

// 获取学习时长（分钟）
async function getStudyTime() {
    try {
        const data = await apiRequest('/api/progress', {
            method: 'POST',
            body: JSON.stringify({
                action: 'get_all',
                user_id: getCurrentUserId()
            })
        });
        
        let totalTime = 0;
        
        if (data.status === 'success' && data.progress) {
            for (const subject in data.progress) {
                const progress = data.progress[subject];
                if (progress) {
                    totalTime += progress.study_time || 0;
                }
            }
        }
        return totalTime;
    } catch (error) {
        console.error('获取学习时长失败:', error);
        return 0;
    }
}

// 合并获取完成任务和学习时长的数据
async function getProgressData() {
    try {
        const data = await apiRequest('/api/progress', {
            method: 'POST',
            body: JSON.stringify({
                action: 'get_all',
                user_id: getCurrentUserId()
            })
        });
        
        let totalTasks = 0;
        let totalTime = 0;
        
        if (data.status === 'success' && data.progress) {
            for (const subject in data.progress) {
                const progress = data.progress[subject];
                if (progress) {
                    totalTasks += progress.completed_tasks || 0;
                    totalTime += progress.study_time || 0;
                }
            }
        }
        
        return { tasks: totalTasks, studyTime: totalTime };
    } catch (error) {
        console.error('获取进度数据失败:', error);
        return { tasks: 0, studyTime: 0 };
    }
}

// 数值动画
function animateValue(element, start, end, duration) {
    if (!element) return;
    
    const startTime = performance.now();
    const isNumber = typeof end === 'number';
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        if (isNumber) {
            const current = Math.floor(start + (end - start) * progress);
            element.textContent = current;
        }
        
        if (progress < 1) {
            requestAnimationFrame(update);
        } else if (isNumber) {
            element.textContent = end;
        }
    }
    
    requestAnimationFrame(update);
}

// 加载学习计划
function loadPlans() {
    // 从API获取学习计划
    apiRequest('/api/plan', {
        method: 'POST',
        body: JSON.stringify({ action: 'get', user_id: getCurrentUserId() })
    })
        .then(data => {
            const planList = document.getElementById('planList');
            
            if (!data.plans || data.plans.length === 0) {
                planList.innerHTML = '<p class="empty-state">暂无学习计划，点击上方按钮创建</p>';
                return;
            }
            
            planList.innerHTML = '';
            data.plans.forEach(plan => {
                const planItem = document.createElement('div');
                planItem.className = 'resource-card';
                planItem.innerHTML = `
                    <h3>${plan.subject || '未命名计划'}</h3>
                    <p><strong>时长：</strong>${plan.duration || '未知'}</p>
                    <p><strong>开始日期：</strong>${plan.start_date ? new Date(plan.start_date).toLocaleDateString('zh-CN') : '未设置'}</p>
                    <div class="resource-tags">
                        <span class="resource-tag">${escapePlanText(planTaskProgressLabel(plan))}</span>
                    </div>
                `;
                // 添加操作按钮容器
                const buttonContainer = document.createElement('div');
                buttonContainer.style.display = 'flex';
                buttonContainer.style.gap = '10px';
                buttonContainer.style.marginTop = '15px';
                
                // 查看详情按钮
                const detailButton = document.createElement('button');
                detailButton.className = 'primary-btn';
                detailButton.textContent = '查看详情';
                detailButton.onclick = function() {
                    viewPlanDetails(plan);
                };
                buttonContainer.appendChild(detailButton);
                
                // 删除按钮
                const deleteButton = document.createElement('button');
                deleteButton.className = 'secondary-btn';
                deleteButton.textContent = '删除';
                deleteButton.onclick = function() {
                    handleDeletePlan(plan);
                };
                buttonContainer.appendChild(deleteButton);
                
                planItem.appendChild(buttonContainer);
                planList.appendChild(planItem);
            });
        })
        .catch(error => {
            console.error('加载学习计划失败:', error);
        });
}

// 加载学习资源
const RESOURCE_PREVIEW_COUNT = 8;
let _resourcesExpanded = false;

function toggleResources() {
    _resourcesExpanded = !_resourcesExpanded;
    document.querySelectorAll('#resourceList .resource-card[data-extra="1"]').forEach(el => {
        el.hidden = !_resourcesExpanded;
    });
    const btn = document.getElementById('resourceToggle');
    if (btn) {
        const total = document.querySelectorAll('#resourceList .resource-card').length;
        btn.textContent = _resourcesExpanded ? '收起' : `展开更多（还有 ${total - RESOURCE_PREVIEW_COUNT} 个）`;
    }
}

function loadResources(subject = '') {
    // 从API获取学习资源
    apiRequest('/api/resources', {
        method: 'POST',
        body: JSON.stringify({ user_id: getCurrentUserId(), subject: subject || '' })
    })
        .then(data => {
            const resourceList = document.getElementById('resourceList');
            
            if (!data.resources || data.resources.length === 0) {
                resourceList.innerHTML = '<p class="empty-state">暂无学习资源</p>';
                return;
            }
            
            resourceList.innerHTML = '';
            _resourcesExpanded = false;
            data.resources.forEach((resource, idx) => {
                const resourceItem = document.createElement('div');
                resourceItem.className = 'resource-card';
                if (idx >= RESOURCE_PREVIEW_COUNT) {
                    resourceItem.dataset.extra = '1';
                    resourceItem.hidden = true;
                }
                resourceItem.innerHTML = `
                    <h3>${resource.name || '未命名资源'}</h3>
                    <div class="resource-tags">
                        <span class="resource-tag">${resource.type || '其他'}</span>
                        ${resource.difficulty && resource.difficulty !== '通用' ? `<span class="resource-tag">${resource.difficulty}</span>` : ''}
                    </div>
                    <p>${resource.description || ''}</p>
                    ${resource.url ? `<a href="${resource.url}" target="_blank" class="primary-btn" style="display:inline-block;margin-top:15px">访问资源</a>` : ''}
                `;
                resourceList.appendChild(resourceItem);
            });

            const resToggle = document.getElementById('resourceToggle');
            if (resToggle) {
                const rest = data.resources.length - RESOURCE_PREVIEW_COUNT;
                resToggle.hidden = rest <= 0;
                resToggle.textContent = `展开更多（还有 ${rest} 个）`;
            }
        })
        .catch(error => {
            console.error('加载学习资源失败:', error);
        });
}

// 加载学习进度
function renderProgressPlanEntries(plans) {
    const esc=escapePlanText;
    if (!plans.length) return '<p class="empty-state">暂无学习计划，创建后可在这里查看本周任务和安排下一周。</p>';
    return plans.map(plan=>{
        const current=planCurrentWeek(plan), total=Math.ceil((plan.duration_days || 7)/7),next=current+1;
        const isLong=(plan.duration_days || 0)>14;
        const nextDetailed=(plan.detailed_weeks || []).includes(next) || (plan.daily_plan || []).some(d=>Math.floor((d.day-1)/7)+1===next);
        const summary=planTaskSummary(plan);
        const detailed=(plan.daily_plan || []).filter(d=>Math.floor((d.day-1)/7)+1===current);
        return `<div class="progress-plan-card"><div class="progress-item-header"><strong>${esc(plan.subject)}</strong><span>第${current}周 / 共${total}周</span></div>
            ${plan.goal ? `<p>${esc(plan.goal)}</p>` : ''}
            <p class="plan-meta">${summary.weekTotal ? `本周已完成 ${summary.weekCompleted}/${summary.weekTotal} 项任务` : '本周任务尚未细化'}${nextDetailed ? ' · 下一周已有安排' : ''}</p>
            <div class="progress-plan-actions"><button type="button" class="plan-choice-btn" data-progress-plan-view="${esc(plan.subject)}">查看本周计划</button>
            <button type="button" class="plan-choice-btn" data-progress-plan-next="${esc(plan.subject)}"${!isLong || next>total || !detailed.length ? ' disabled' : ''}>安排下一周</button></div>
            ${isLong && !detailed.length ? '<p class="plan-meta">请先查看并安排本周，再继续下一周。</p>' : ''}
            ${!isLong ? '<p class="plan-meta">短计划已逐日安排，可在本周计划中查看。</p>' : next>total ? '<p class="plan-meta">已到计划最后一周。</p>' : ''}</div>`;
    }).join('');
}

async function loadProgressPlanEntries() {
    const container=document.getElementById('progressPlanEntries');
    if (!container) return;
    try {
        const response=await apiRequest(`/api/plans?user_id=${encodeURIComponent(getCurrentUserId())}`);
        if (response.status!=='success' || !Array.isArray(response.data)) throw new Error(response.message || '未获得计划列表');
        container.innerHTML=renderProgressPlanEntries(response.data.filter(p=>p.status!=='deleted'));
    } catch(error) {
        container.innerHTML=`<p class="error-state">计划入口加载失败：${escapePlanText(error.message)}</p>`;
    }
}

async function handleProgressPlanAction(button) {
    const subject=button.dataset.progressPlanView || button.dataset.progressPlanNext;
    if (!subject || button.disabled) return;
    button.disabled=true;
    const label=button.textContent;
    try {
        const response=await apiRequest(`/api/plans/${encodeURIComponent(subject)}?user_id=${encodeURIComponent(getCurrentUserId())}`);
        let plan=response?.data?.plan || response?.data;
        if (!plan?.weekly_plan) throw new Error('未找到学习计划，请刷新后重试。');
        const current=planCurrentWeek(plan), next=current+1;
        let focusWeek=current;
        if (button.dataset.progressPlanNext) {
            if (next>Math.ceil(plan.duration_days/7)) throw new Error('已到计划最后一周。');
            if (!(plan.daily_plan || []).some(d=>Math.floor((d.day-1)/7)+1===current)) throw new Error('请先查看并安排本周，再继续下一周。');
            focusWeek=next;
            const exists=(plan.daily_plan || []).some(d=>Math.floor((d.day-1)/7)+1===next);
            // 已有下一周安排直接查看，避免重按入口把已生成任务替换掉。
            if (!exists) {
                button.textContent='正在结合进度安排…';
                const result=await apiRequest(`/api/plans/${encodeURIComponent(subject)}/weeks/${next}/generate`,{
                    method:'POST',body:JSON.stringify({user_id:getCurrentUserId()})
                }, 190000);
                if (result.status!=='success' || !result.data) throw new Error(result.message || '安排失败');
                plan=result.data;
            }
        }
        switchPage('plans');
        const panel=document.getElementById('planDetailPanel');
        if (panel) {
            panel.style.display='block';
            renderPlanDetail(panel,subject,plan,{focusWeek});
            (panel.querySelector(`[data-plan-week="${focusWeek}"]`) || panel).scrollIntoView({behavior:'smooth',block:'start'});
        }
        loadProgressPlanEntries();
    } catch(error) { showError(error.message); }
    finally { button.disabled=false; button.textContent=label; }
}

document.addEventListener('click',event=>{
    const button=event.target.closest?.('button[data-progress-plan-view], button[data-progress-plan-next]');
    if (button) handleProgressPlanAction(button);
});

function loadProgress() {
    apiRequest('/api/progress', {
        method: 'POST',
        body: JSON.stringify({ action: 'get_all', user_id: getCurrentUserId() })
    })
    .then(data => {
        const progressStats = document.getElementById('progressStats');
        if (!progressStats) return;
        const all = data.progress && typeof data.progress === 'object' ? data.progress : {};
        const subjects = Object.keys(all);

        let totalTasks = 0;
        let completedTasks = 0;
        let totalStudyMinutes = 0;
        subjects.forEach(key => {
            const p = all[key] || {};
            totalTasks += Number(p.total_tasks || 0);
            completedTasks += Number(p.completed_tasks || 0);
            totalStudyMinutes += Number(p.study_time || 0);
        });
        const overallPercentage = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;
        const totalStudyHours = (totalStudyMinutes / 60).toFixed(1);

        let html = `
            <section class="progress-plans-section"><h3>学习计划</h3><p class="plan-meta">查看本周任务，完成后结合进度安排下一周。</p><div id="progressPlanEntries"><div class="loading-spinner-small"></div> 正在加载计划…</div></section>
            <div class="card" style="padding:16px;margin-bottom:16px;">
                <h3 style="margin-bottom:12px;">手动更新学习进度</h3>
                <div class="form-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;">
                    <div class="form-group">
                        <label for="manualProgressSubject">学习主题</label>
                        <input id="manualProgressSubject" type="text" placeholder="如：机器学习" />
                    </div>
                    <div class="form-group">
                        <label for="manualProgressCompleted">已完成任务</label>
                        <input id="manualProgressCompleted" type="number" min="0" value="0" />
                    </div>
                    <div class="form-group">
                        <label for="manualProgressTotal">总任务数</label>
                        <input id="manualProgressTotal" type="number" min="0" value="0" />
                    </div>
                    <div class="form-group">
                        <label for="manualProgressHours">本次学习时长(小时)</label>
                        <input id="manualProgressHours" type="number" min="0" step="0.5" value="0" />
                    </div>
                </div>
                <div style="margin-top:10px;">
                    <button class="primary-btn" onclick="submitManualProgressUpdate()">保存进度</button>
                </div>
            </div>
            <div class="progress-summary">
                <div class="progress-item">
                    <span class="progress-label">已记录任务完成率</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${overallPercentage}%"></div>
                    </div>
                    <span class="progress-percentage">${overallPercentage}%</span>
                </div>
                <div style="margin-top:10px;color:var(--text-light);font-size:0.9em;">
                    已完成 ${completedTasks}/${totalTasks} 任务 · 累计学习 ${totalStudyHours} 小时
                </div>
            </div>
            <div class="progress-list" style="margin-top:16px;">
        `;

        if (subjects.length === 0) {
            html += '<p class="empty-state">暂无学习进度记录，请先使用上方表单新增一条。</p>';
        } else {
            subjects.sort((a, b) => (all[b].last_updated || '').localeCompare(all[a].last_updated || ''));
            subjects.forEach(subjectKey => {
                const p = all[subjectKey] || {};
                const subject = p.subject || subjectKey.replace(`${getCurrentUserId()}_`, '');
                const percentage = Number(p.progress_percentage || 0);
                const completed = Number(p.completed_tasks || 0);
                const total = Number(p.total_tasks || 0);
                const studyTime = (Number(p.study_time || 0) / 60).toFixed(1);
                const suggestions = Array.isArray(p.suggestions) ? p.suggestions : [];
                const suggestionText = suggestions.length > 0 ? suggestions[0] : '继续保持当前学习节奏';

                html += `
                    <div class="progress-item-card">
                        <div class="progress-item-header">
                            <span class="progress-subject">${subject}</span>
                            <span class="progress-percentage">${percentage}%</span>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: ${percentage}%"></div>
                        </div>
                        <div class="progress-item-footer">
                            <span>${progressTaskScope(p)} ${completed}/${total}</span>
                            <span>学习 ${studyTime}h</span>${p.plan_progress_summary?.has_unplanned_weeks ? '<span>后续周尚未安排</span>' : ''}
                        </div>
                        <div style="font-size:0.85em;color:var(--text-light);margin-top:6px;">建议：${suggestionText}</div>
                        <div class="progress-item-actions">
                            <button class="ghost-btn compact" onclick="editProgress('${subject}', ${completed}, ${total}, ${studyTime})">编辑</button>
                            <button class="link-danger" onclick="deleteProgress('${subject}')">删除</button>
                        </div>
                    </div>
                `;
            });
        }
        html += '</div>';
        progressStats.innerHTML = html;
        loadProgressPlanEntries();
    })
    .catch(error => {
        const progressStats = document.getElementById('progressStats');
        if (progressStats) {
            progressStats.innerHTML = '<section class="progress-plans-section"><h3>学习计划</h3><div id="progressPlanEntries"></div></section><p class="error-state">加载学习进度失败: ' + escapePlanText(error.message) + '</p>';
            loadProgressPlanEntries();
        }
    });
}

async function submitManualProgressUpdate() {
    const subject = (document.getElementById('manualProgressSubject')?.value || '').trim();
    const completedTasks = Number(document.getElementById('manualProgressCompleted')?.value || 0);
    const totalTasks = Number(document.getElementById('manualProgressTotal')?.value || 0);
    const studyHours = Number(document.getElementById('manualProgressHours')?.value || 0);

    if (!subject) {
        showError('请先填写学习主题');
        return;
    }
    if (completedTasks < 0 || totalTasks < 0 || studyHours < 0) {
        showError('请输入合法的非负数');
        return;
    }

    try {
        await apiRequest('/api/progress', {
            method: 'POST',
            body: JSON.stringify({
                action: 'update',
                user_id: getCurrentUserId(),
                subject: subject,
                completed_tasks: completedTasks,
                total_tasks: totalTasks,
                study_hours: studyHours
            })
        });
        showSuccess('学习进度已更新');
        loadProgress();
        loadProgressOverview();
        updateDashboardData();
    } catch (error) {
        showError('更新失败: ' + error.message);
    }
}

let currentNoteId = null;

// 加载学习笔记
function loadNotes() {
    const noteList = document.getElementById('noteList');
    if (!noteList) return;
    
    noteList.innerHTML = '<div class="loading-spinner-small"></div> 正在加载...';
    
    apiRequest(`/api/notes?user_id=${getCurrentUserId()}`)
        .then(data => {
            if (data.status === 'success') {
                const notes = data.data; // 注意：后端返回的是 data: notes
                if (!notes || notes.length === 0) {
                    noteList.innerHTML = '<p class="empty-state">暂无笔记</p>';
                    return;
                }
                
                noteList.innerHTML = '';
                notes.forEach(note => {
                    const safeId = encodeURIComponent(String(note.id || ''));
                    const noteItem = document.createElement('div');
                    noteItem.className = 'note-item';
                    noteItem.style.cursor = 'pointer';
                    noteItem.innerHTML = `
                        <div style="display:flex;justify-content:space-between;align-items:start">
                            <h4 style="margin:0 0 5px 0">${escapeHtml(note.title || '无标题笔记')}</h4>
                            <button class="danger-btn" style="padding:2px 6px;font-size:0.8em;margin-left:10px" data-note-delete="${safeId}">删除</button>
                        </div>
                        <p style="margin:0 0 5px 0;color:var(--text-main)">${escapeHtml(String(note.content || '').substring(0, 100))}${String(note.content || '').length > 100 ? '...' : ''}</p>
                        <div style="font-size:0.8em;color:var(--text-light);display:flex;justify-content:space-between">
                            <span>${note.subject ? `<span class="tag" style="font-size:0.9em;padding:1px 5px">${escapeHtml(note.subject)}</span> ` : ''}</span>
                            <span>${new Date(note.updated_at || note.created_at).toLocaleString('zh-CN')}</span>
                        </div>
                    `;
                    noteItem.querySelector('[data-note-delete]').addEventListener('click', event => {
                        event.stopPropagation();
                        deleteNote(note.id);
                    });
                    noteItem.addEventListener('click', () => editNote(note));
                    noteList.appendChild(noteItem);
                });
            } else {
                noteList.innerHTML = '<p class="error-state">加载失败: ' + (data.message || '未知错误') + '</p>';
            }
        })
        .catch(error => {
            console.error('加载笔记失败:', error);
            noteList.innerHTML = '<p class="error-state">加载失败</p>';
        });
}

// 创建新计划
function createNewPlan() {
    showPlanInputGuide();
}

// 查看计划详情
async function viewPlanDetails(plan) {
    const panel = document.getElementById('planDetailPanel');
    if (!panel) return;
    const subject = (plan && plan.subject) ? String(plan.subject) : '';
    if (!subject) return;
    panel.style.display = 'block';
    panel._planSaveState = null;
    panel.innerHTML = '<div class="loading-spinner"></div><p>正在加载计划详情...</p>';
    try {
        const resp = await apiRequest(`/api/plans/${encodeURIComponent(subject)}?user_id=${encodeURIComponent(getCurrentUserId())}`);
        const p = resp?.data?.plan || (resp?.data?.weekly_plan ? resp.data : null);
        if (!p) {
            panel.innerHTML = '<div class="error-state">未找到计划详情</div>';
            return;
        }
        renderPlanDetail(panel, subject, p);
        panel.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        panel.innerHTML = '<div class="error-state">加载失败：' + (e && e.message ? e.message : '未知错误') + '</div>';
    }
}

// 删除学习计划
async function handleDeletePlan(plan) {
    const subject = (plan && plan.subject) ? String(plan.subject) : '';
    if (!subject) return;
    if (!confirm(`确定要删除学习计划"${subject}"吗？`)) return;
    try {
        const resp = await apiRequest(`/api/plans/${encodeURIComponent(subject)}?user_id=${encodeURIComponent(getCurrentUserId())}`, {
            method: 'DELETE'
        });
        if (resp.status === 'success') {
            showSuccess('学习计划已删除');
            const panel = document.getElementById('planDetailPanel');
            if (panel) {
                panel._planSaveState = null;
                panel.style.display = 'none';
                panel.innerHTML = '';
            }
            loadPlans();
            loadProgressOverview();
        } else {
            showError('删除失败: ' + (resp.message || '未知错误'));
        }
    } catch (e) {
        showError('删除失败: ' + (e && e.message ? e.message : '未知错误'));
    }
}

function renderPlanDetail(container, subject, plan, options = {}) {
    const userId = getCurrentUserId();
    const key = planSaveKey(userId, subject, plan);
    let saveState = planTaskSaveStates.get(key);
    if (!saveState) {
        saveState = {userId, subject, createdAt: plan.created_at, plan, pending: new Map(), running: false};
        planTaskSaveStates.set(key, saveState);
    }
    if (saveState.running) {
        plan = JSON.parse(JSON.stringify(saveState.plan));
        const desired = [saveState.inFlight, ...saveState.pending.values()].filter(Boolean);
        desired.forEach(task => {
            plan.task_completion ||= {};
            plan.task_completion[task.week] ||= [];
            plan.task_completion[task.week][task.index] = task.completed;
        });
    } else saveState.plan = plan;
    saveState.view = {container, options};
    container._planSaveState = saveState;
    const summary = planTaskSummary(plan);
    const showWeek = (plan.duration_days || 0) > 14 || summary.hasUnplanned;
    const count = showWeek ? summary.weekCompleted : summary.completed;
    const total = showWeek ? summary.weekTotal : summary.total;
    const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
    const duration = esc(plan.requested_duration || plan.duration || '');
    const start = plan.start_date ? new Date(plan.start_date).toLocaleDateString('zh-CN') : '';
    const progress = (showWeek ? summary.weekPercentage : summary.percentage) ?? 0;
    const isDaily = Array.isArray(plan.daily_plan) && plan.daily_plan.length > 0;
    const weekly = isDaily ? plan.daily_plan : (Array.isArray(plan.weekly_plan) ? plan.weekly_plan : []);
    const tc = plan.task_completion && typeof plan.task_completion === 'object' ? plan.task_completion : {};

    const header = `
        <div style="background:var(--card-bg);border-radius:12px;padding:18px;box-shadow:0 4px 6px rgba(0,0,0,0.05);">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
                <div style="font-weight:800;font-size:1.1em;">${esc(subject)} · 计划详情</div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;">
                    <button class="secondary-btn" onclick="collapsePlanDetail()"><i class="fas fa-chevron-up"></i> 收起</button>
                    <button class="secondary-btn plan-sync-btn" onclick="syncPlanTasks('${encodeURIComponent(subject)}')"><i class="fas fa-sync"></i> 同步进度</button>
                    <button class="secondary-btn" onclick="startQuizForSubject('${encodeURIComponent(subject)}')"><i class="fas fa-question-circle"></i> 生成测验</button>
                    <button class="secondary-btn" onclick="openNoteForSubject('${encodeURIComponent(subject)}')"><i class="fas fa-sticky-note"></i> 新建笔记</button>
                    <button class="secondary-btn" onclick="openResourcesForSubject('${encodeURIComponent(subject)}')"><i class="fas fa-book-open"></i> 查看资源</button>
                </div>
            </div>
            <div style="margin-top:10px;color:var(--text-light);font-size:0.9em;">时长：${duration || '—'} · 开始：${start || '—'}</div>
            ${plan.planned_end_date ? `<div class="plan-meta">结束：${esc(plan.planned_end_date)}</div>` : ''}
            <button type="button" class="plan-choice-btn plan-edit-conditions" data-plan-prefill="${esc(planConditionMessage(plan))}">修改条件</button>
            ${plan.goal ? `<div style="margin-top:8px;">目标：${esc(plan.goal)}</div>` : ''}
            ${plan.daily_hours != null ? `<div style="margin-top:8px;">每天可用：${esc(Math.round(plan.daily_hours * 60))} 分钟</div>` : ''}
            ${plan.schedule_note ? `<p>${esc(plan.schedule_note)}</p>` : ''}
            ${plan.outcome ? `<p>本期成果：${esc(plan.outcome)}</p>` : ''}
            ${(plan.assumptions || []).length ? `<p>${plan.assumptions.map(esc).join(' ')}</p>` : ''}
            ${Array.isArray(plan.deferred_tasks) && plan.deferred_tasks.length ? `<details><summary>后续任务（${plan.deferred_tasks.length} 项）</summary><ul>${plan.deferred_tasks.map(t => `<li>${esc(t.task)}</li>`).join('')}</ul></details>` : ''}
            <div style="margin-top:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="color:var(--text-light);font-size:0.9em;">${showWeek ? '本周任务完成' : '任务完成'}</span>
                    <span style="font-weight:700;">${total ? `${count}/${total} 项 · ${progress}%` : '尚未安排'}</span>
                </div>
                <div style="height:10px;background:rgba(0,0,0,0.06);border-radius:999px;overflow:hidden;">
                    <div style="height:10px;width:${Math.max(0, Math.min(100, progress))}%;background:var(--primary-color);"></div>
                </div>
                <p class="plan-meta">已安排任务累计完成 ${summary.completed}/${summary.total} 项${summary.hasUnplanned ? ` · 已细化 ${summary.detailedWeeks}/${summary.totalWeeks} 周，后续任务尚未安排` : ""}。任务完成率用于查看执行情况。</p>
                <p class="plan-save-status" data-plan-save-status role="status" aria-live="polite"></p>
            </div>
        </div>
    `;

    let body = renderPlanPhases(plan) + '<div style="margin-top:15px;display:grid;gap:12px;">';
    const renderDay = (w, idx=0) => {
        const weekKey = String(w.week || (idx + 1));
        const topics = Array.isArray(w.topics) ? w.topics : [];
        const tasks = Array.isArray(w.tasks) ? w.tasks : [];
        const doneArr = Array.isArray(tc[weekKey]) ? tc[weekKey] : [];
        const list = tasks.map((t, i) => {
            const ref = isDaily ? (w.task_refs || [])[i] : {week: weekKey, task_index: i};
            const checked = isDaily ? !!((tc[ref?.week] || [])[ref?.task_index]) : !!doneArr[i];
            return `
                <label style="display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-top:1px dashed rgba(0,0,0,0.06);">
                    <input type="checkbox" data-week="${esc(ref?.week ?? weekKey)}" data-idx="${esc(ref?.task_index ?? i)}" ${checked ? 'checked' : ''}>
                    <span style="line-height:1.4;">${esc(t)}</span>
                </label>
            `;
        }).join('');

        const topicsHtml = topics.length ? `<div style="color:var(--text-light);font-size:0.9em;margin-top:6px;">主题：${topics.map(esc).join(' / ')}</div>` : '';
        return `
            <div style="background:var(--card-bg);border-radius:12px;padding:16px;box-shadow:0 4px 6px rgba(0,0,0,0.05);">
                <div style="font-weight:800;">第${esc(isDaily ? w.day : weekKey)}${isDaily ? '天' : '周'}${isDaily && w.date ? ` · ${esc(w.date)}` : ''}</div>
                ${w.available_hours != null ? `<div>预计 ${esc(isDaily ? Math.round(w.estimated_hours * 60) : w.estimated_hours)} / 可用 ${esc(isDaily ? Math.round(w.available_hours * 60) : w.available_hours)} ${isDaily ? '分钟' : '小时'}</div>` : ''}
                ${topicsHtml}
                <div style="margin-top:10px;">
                    ${tasks.length ? list : `<div class="empty-state">${isDaily ? '当天' : '本周'}暂无任务</div>`}
                </div>
                ${(w.deliverables || []).length ? `<div>产出：${w.deliverables.map(esc).join('、')}</div>` : ''}
            </div>
        `;
    };
    body += isDaily ? renderPlanWeekGroups(plan, renderDay, options.focusWeek) : weekly.map(renderDay).join('');
    body += '</div>';
    container.innerHTML = header + body;
    container.querySelectorAll('input[type="checkbox"][data-week]').forEach(cb => {
        cb.addEventListener('change', () => {
            queuePlanTaskSave(saveState, cb.getAttribute('data-week'), Number(cb.getAttribute('data-idx')), cb.checked);
        });
    });
    setPlanSaveStatus(saveState, saveState.running ? '正在保存勾选…' : '');
}

function collapsePlanDetail() {
    const panel = document.getElementById('planDetailPanel');
    if (!panel) return;
    panel._planSaveState = null;
    panel.style.display = 'none';
    panel.innerHTML = '';
}

async function syncPlanTasks(encodedSubject) {
    const subject = decodeURIComponent(encodedSubject || '');
    const panel = document.getElementById('planDetailPanel');
    if (!subject || !panel) return;
    if (panel._planSaveState?.running) await panel._planSaveState.done;
    try {
        const resp = await apiRequest(`/api/plans/${encodeURIComponent(subject)}/tasks/sync`, {
            method: 'POST',
            body: JSON.stringify({ user_id: getCurrentUserId() })
        });
        if (resp && resp.status === 'success' && resp.data) {
            renderPlanDetail(panel, subject, resp.data);
            loadProgressOverview();
            updateDashboardData();
            loadProgressPlanEntries();
            loadPlans();
            if (resp.sync_warning) showError(resp.sync_warning);
            else showSuccess('已同步到进度');
        } else {
            showError('同步失败');
        }
    } catch (e) {
        showError('同步失败: ' + (e && e.message ? e.message : '未知错误'));
    }
}

function startQuizForSubject(encodedSubject) {
    const subject = decodeURIComponent(encodedSubject || '');
    if (!subject) return;
    switchPage('quiz');
    setTimeout(() => {
        const input = document.getElementById('quizSubject');
        if (input) input.value = subject;
        submitQuizGeneration();
    }, 50);
}

let currentNoteSubject = '通用';
function openNoteForSubject(encodedSubject) {
    const subject = decodeURIComponent(encodedSubject || '') || '通用';
    currentNoteSubject = subject;
    switchPage('notes');
    setTimeout(() => {
        const titleInput = document.getElementById('noteTitle');
        const contentInput = document.getElementById('noteContent');
        if (titleInput && !titleInput.value) titleInput.value = subject + ' - ';
        if (contentInput && !contentInput.value) contentInput.value = '';
        const editor = document.getElementById('noteEditor');
        if (editor) editor.scrollIntoView({ behavior: 'smooth' });
    }, 50);
}

function openResourcesForSubject(encodedSubject) {
    const subject = decodeURIComponent(encodedSubject || '') || '';
    switchPage('resources');
    setTimeout(() => loadResources(subject), 50);
}

// 创建新笔记
function createNewNote() {
    currentNoteId = null;
    currentNoteSubject = '通用';
    const titleInput = document.getElementById('noteTitle');
    const contentInput = document.getElementById('noteContent');
    if (titleInput) titleInput.value = '';
    
    // 如果使用了Monaco Editor，清空它
    if (noteEditorInstance) {
        noteEditorInstance.setValue('');
    } else if (contentInput) {
        contentInput.value = '';
    }
    
    switchPage('notes');
}

// 编辑笔记
function editNote(note) {
    currentNoteId = note.id;
    const titleInput = document.getElementById('noteTitle');
    const contentInput = document.getElementById('noteContent');
    if (titleInput) titleInput.value = note.title || '';
    
    // 如果使用了Monaco Editor，设置值
    if (noteEditorInstance) {
        noteEditorInstance.setValue(note.content || '');
    } else if (contentInput) {
        contentInput.value = note.content || '';
    }
    
    // 滚动到编辑器
    const editor = document.getElementById('noteEditor');
    if (editor) editor.scrollIntoView({ behavior: 'smooth' });
}

// 保存笔记
function saveNote() {
    const titleInput = document.getElementById('noteTitle');
    const contentInput = document.getElementById('noteContent');
    const title = titleInput.value.trim();
    
    let content = '';
    // 如果使用了Monaco Editor，从它获取值
    if (noteEditorInstance) {
        content = noteEditorInstance.getValue().trim();
    } else {
        content = contentInput.value.trim();
    }
    
    if (!title || !content) {
        showError('请输入笔记标题和内容');
        return;
    }
    
    const noteData = {
        user_id: getCurrentUserId(),
        title,
        content,
        subject: currentNoteSubject || '通用'
    };
    
    let promise;
    if (currentNoteId) {
        promise = apiRequest(`/api/notes/${encodeURIComponent(currentNoteId)}?user_id=${encodeURIComponent(getCurrentUserId())}`, {
            method: 'PUT',
            body: JSON.stringify(noteData)
        });
    } else {
        promise = apiRequest('/api/notes', {
            method: 'POST',
            body: JSON.stringify(noteData)
        });
    }
    
    promise.then(data => {
        if (data.status === 'success') {
            showSuccess('笔记保存成功');
            // 清空编辑器状态但保留在当前页面
            currentNoteId = null;
            titleInput.value = '';
            
            if (noteEditorInstance) {
                noteEditorInstance.setValue('');
            } else if (contentInput) {
                contentInput.value = '';
            }

            loadNotes();
        } else {
            showError('保存失败: ' + (data.message || '未知错误'));
        }
    }).catch(error => {
        showError('保存失败: ' + error.message);
    });
}

// 删除笔记
function deleteNote(noteId) {
    if (!confirm('确定要删除这条笔记吗？')) return;
    
    apiRequest(`/api/notes/${encodeURIComponent(noteId)}?user_id=${encodeURIComponent(getCurrentUserId())}`, {
        method: 'DELETE'
    }).then(data => {
        if (data.status === 'success') {
            showSuccess('笔记已删除');
            if (currentNoteId === noteId) {
                createNewNote();
            }
            loadNotes();
        } else {
            showError('删除失败: ' + (data.message || '未知错误'));
        }
    }).catch(error => {
        showError('删除失败: ' + error.message);
    });
}

// 测验功能实现
let currentQuiz = null;
let currentQuestionIndex = 0;
let lastQuizResult = null;

// 生成新测验
function generateNewQuiz() {
    document.getElementById('quizGenerator').style.display = 'block';
    document.getElementById('quizTaking').style.display = 'none';
    document.getElementById('quizResult').style.display = 'none';
    document.getElementById('quizHistory').style.display = 'none';
}

// 提交测验生成请求
function submitQuizGeneration() {
    const subject = document.getElementById('quizSubject').value.trim();
    const difficulty = document.getElementById('quizDifficulty').value;
    const questionsCount = parseInt(document.getElementById('quizQuestions').value);
    
    if (!subject) {
        showError('请输入测验主题');
        return;
    }
    
    // 显示加载状态
    const quizContainer = document.getElementById('quizContainer');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-state';
    loadingDiv.innerHTML = '<div class="loading-spinner"></div><p>正在生成测验...</p>';
    quizContainer.appendChild(loadingDiv);
    
    // 准备请求体
    const requestBody = {
        subject: subject,
        difficulty: difficulty,
        questions_count: questionsCount,
        user_id: getCurrentUserId()
    };
    
    // 发送请求到后端
    apiRequest('/api/quiz/generate', {
        method: 'POST',
        body: JSON.stringify(requestBody)
    })
    .then(data => {
        // 移除加载状态
        quizContainer.removeChild(loadingDiv);
        
        if (data.status === 'success' && data.quiz) {
            currentQuiz = data.quiz;
            currentQuestionIndex = 0;
            
            // 显示测验
            document.getElementById('quizGenerator').style.display = 'none';
            document.getElementById('quizTaking').style.display = 'block';
            
            // 设置测验标题
            document.getElementById('quizTitle').textContent = currentQuiz.title || `测验：${subject}`;
            
            // 显示第一个问题
            displayQuestion();
        } else {
            showError('测验生成失败: ' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        quizContainer.removeChild(loadingDiv);
        showError('发生错误: ' + error.message, error);
    });
}

// 显示问题
function displayQuestion() {
    if (!currentQuiz || !currentQuiz.questions || currentQuestionIndex >= currentQuiz.questions.length) {
        return;
    }
    
    const question = currentQuiz.questions[currentQuestionIndex];
    const questionElement = document.getElementById('quizQuestion');
    
    // 更新进度
    document.getElementById('quizProgress').textContent = 
        `问题 ${currentQuestionIndex + 1}/${currentQuiz.questions.length}`;
    
    // 构建问题HTML
    let html = `<div class="question-text"><strong>${question.text}</strong></div>`;
    
    // 根据问题类型显示不同的选项
    if (question.type === 'multiple_choice' && question.options) {
        html += '<div class="question-options">';
        question.options.forEach((option, index) => {
            const optionId = `option_${currentQuestionIndex}_${index}`;
            const isChecked = question.selected === index;
            html += `
                <div class="option-item">
                    <input type="radio" id="${optionId}" name="question_${currentQuestionIndex}" 
                           value="${index}" ${isChecked ? 'checked' : ''} 
                           onchange="selectQuizOption(${currentQuestionIndex}, ${index})">
                    <label for="${optionId}">${option}</label>
                </div>
            `;
        });
        html += '</div>';
    } else if (question.type === 'true_false') {
        html += '<div class="question-options">';
        const trueChecked = question.selected === 'true';
        const falseChecked = question.selected === 'false';
        html += `
            <div class="option-item">
                <input type="radio" id="option_${currentQuestionIndex}_true" name="question_${currentQuestionIndex}" 
                       value="true" ${trueChecked ? 'checked' : ''} 
                       onchange="selectQuizOption(${currentQuestionIndex}, 'true')">
                <label for="option_${currentQuestionIndex}_true">正确</label>
            </div>
            <div class="option-item">
                <input type="radio" id="option_${currentQuestionIndex}_false" name="question_${currentQuestionIndex}" 
                       value="false" ${falseChecked ? 'checked' : ''} 
                       onchange="selectQuizOption(${currentQuestionIndex}, 'false')">
                <label for="option_${currentQuestionIndex}_false">错误</label>
            </div>
        `;
        html += '</div>';
    } else if (question.type === 'short_answer') {
        const userAnswer = question.user_answer || '';
        html += `
            <div class="question-answer">
                <textarea id="answer_${currentQuestionIndex}" class="short-answer-input" 
                          placeholder="请输入您的答案..." 
                          onchange="saveShortAnswer(${currentQuestionIndex}, this.value)">
                    ${userAnswer}
                </textarea>
            </div>
        `;
    }
    
    questionElement.innerHTML = html;
    
    // 更新导航按钮
    updateQuizNavigationButtons();
}

// 选择测验选项
function selectQuizOption(questionIndex, optionIndex) {
    if (currentQuiz && currentQuiz.questions && currentQuiz.questions[questionIndex]) {
        currentQuiz.questions[questionIndex].selected = optionIndex;
    }
}

// 保存简答题答案
function saveShortAnswer(questionIndex, answer) {
    if (currentQuiz && currentQuiz.questions && currentQuiz.questions[questionIndex]) {
        currentQuiz.questions[questionIndex].user_answer = answer;
    }
}

// 更新测验导航按钮
function updateQuizNavigationButtons() {
    const prevButton = document.querySelector('.quiz-navigation .secondary-btn');
    const nextButton = document.querySelector('.quiz-navigation .primary-btn');
    const submitButton = document.querySelector('.quiz-navigation .success-btn');
    
    // 第一题时隐藏上一题按钮
    prevButton.style.display = currentQuestionIndex === 0 ? 'none' : 'inline-block';
    
    // 最后一题时显示提交按钮，隐藏下一题按钮
    if (currentQuestionIndex === currentQuiz.questions.length - 1) {
        nextButton.style.display = 'none';
        submitButton.style.display = 'inline-block';
    } else {
        nextButton.style.display = 'inline-block';
        submitButton.style.display = 'none';
    }
}

// 上一题
function previousQuestion() {
    if (currentQuestionIndex > 0) {
        currentQuestionIndex--;
        displayQuestion();
    }
}

// 下一题
function nextQuestion() {
    if (currentQuiz && currentQuestionIndex < currentQuiz.questions.length - 1) {
        currentQuestionIndex++;
        displayQuestion();
    }
}

// 提交测验
function submitQuiz() {
    // 检查是否所有问题都已回答
    let allAnswered = true;
    currentQuiz.questions.forEach(question => {
        if (question.type === 'multiple_choice' || question.type === 'true_false') {
            if (question.selected === undefined) {
                allAnswered = false;
            }
        } else if (question.type === 'short_answer') {
            if (!question.user_answer || question.user_answer.trim() === '') {
                allAnswered = false;
            }
        }
    });
    
    if (!allAnswered && !confirm('还有未回答的问题，确定要提交吗？')) {
        return;
    }
    
    // 显示加载状态
    const quizContainer = document.getElementById('quizContainer');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-state';
    loadingDiv.innerHTML = '<div class="loading-spinner"></div><p>正在评分...</p>';
    quizContainer.appendChild(loadingDiv);
    
    // 发送请求到后端
    apiRequest('/api/quiz/submit', {
        method: 'POST',
        body: JSON.stringify({
            quiz_id: currentQuiz.id,
            answers: currentQuiz.questions.map(q => ({
                question_id: q.id,
                selected: q.selected,
                user_answer: q.user_answer
            })),
            user_id: getCurrentUserId()
        })
    })
    .then(data => {
        // 移除加载状态
        quizContainer.removeChild(loadingDiv);
        
        if (data.status === 'success' && data.result) {
            // 增加经验值
            if (data.result && data.result.questions) {
                apiRequest('/api/profile', {
                    method: 'PUT',
                    body: JSON.stringify({ xp_gain: data.result.score })
                }).then(() => loadUserProfile());
            }

            // 显示结果
            document.getElementById('quizTaking').style.display = 'none';
            document.getElementById('quizResult').style.display = 'block';
            
            // 渲染结果
            renderQuizResult(data.result);
        } else {
            alert('提交失败: ' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        quizContainer.removeChild(loadingDiv);
        showError('发生错误: ' + error.message, error);
    });
}

// 渲染测验结果
function renderQuizResult(result) {
    const summaryElement = document.getElementById('resultSummary');
    const detailsElement = document.getElementById('resultDetails');

    lastQuizResult = result || null;
    if (lastQuizResult && currentQuiz) {
        if (!lastQuizResult.subject) lastQuizResult.subject = currentQuiz.subject;
        if (!lastQuizResult.difficulty) lastQuizResult.difficulty = currentQuiz.difficulty;
        if (!lastQuizResult.quiz_id && currentQuiz.id) lastQuizResult.quiz_id = currentQuiz.id;
    }
    
    // 计算得分
    const correctCount = result.questions.filter(q => q.correct).length;
    const totalCount = result.questions.length;
    const scorePercentage = Math.round((correctCount / totalCount) * 100);
    
    // 渲染摘要
    summaryElement.innerHTML = `
        <div class="result-score">
            <div class="score-circle">${scorePercentage}%</div>
            <div class="score-details">
                <p><strong>得分：</strong>${correctCount}/${totalCount}</p>
                <p><strong>评价：</strong>${getScoreEvaluation(scorePercentage)}</p>
            </div>
        </div>
    `;
    
    // 渲染详情
    let detailsHtml = '<div class="question-results">';
    result.questions.forEach((question, index) => {
        const isCorrect = question.correct;
        let flashcardBtn = '';
        if (!isCorrect) {
            const payload = {
                subject: result.subject || '通用',
                front: question.text || '',
                back: formatCorrectAnswer(question) + (question.explanation ? ('\n\n解释：' + question.explanation) : '')
            };
            flashcardBtn = `<button class="secondary-btn quiz-flashcard-btn" style="margin-top:10px;padding:6px 10px;font-size:0.9em;" data-card="${encodeURIComponent(JSON.stringify(payload))}"><i class="fas fa-clone"></i> 生成闪卡</button>`;
        }
        detailsHtml += `
            <div class="question-result ${isCorrect ? 'correct' : 'incorrect'}">
                <div class="question-header">
                    <span class="question-number">问题 ${index + 1}</span>
                    <span class="question-status">${isCorrect ? '✓ 正确' : '✗ 错误'}</span>
                </div>
                <div class="question-text">${question.text}</div>
                <div class="answer-info">
                    <p><strong>您的答案：</strong>${formatUserAnswer(question)}</p>
                    <p><strong>正确答案：</strong>${formatCorrectAnswer(question)}</p>
                    ${question.explanation ? `<p><strong>解释：</strong>${question.explanation}</p>` : ''}
                    ${flashcardBtn}
                </div>
            </div>
        `;
    });
    detailsHtml += '</div>';
    detailsElement.innerHTML = detailsHtml;
    detailsElement.querySelectorAll('.quiz-flashcard-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            try {
                const raw = btn.getAttribute('data-card') || '';
                const data = JSON.parse(decodeURIComponent(raw));
                openCreateFlashcardWithData(data.subject, data.front, data.back);
            } catch (e) {
                showError('生成闪卡失败');
            }
        });
    });
}

// 获取得分评价
function getScoreEvaluation(percentage) {
    if (percentage >= 90) return '优秀！';
    if (percentage >= 80) return '良好！';
    if (percentage >= 70) return '不错！';
    if (percentage >= 60) return '及格';
    return '需要努力';
}

// 格式化正确答案
function formatCorrectAnswer(question) {
    if (question.type === 'multiple_choice' && question.options) {
        const correctIndex = typeof question.correct_answer === 'number' ? question.correct_answer : parseInt(question.correct_answer);
        if (correctIndex >= 0 && correctIndex < question.options.length) {
            return question.options[correctIndex];
        }
        return String(question.correct_answer);
    } else if (question.type === 'true_false') {
        return question.correct_answer === 'true' || question.correct_answer === true ? '正确' : '错误';
    }
    return String(question.correct_answer || '');
}

// 格式化用户答案
function formatUserAnswer(question) {
    // 优先使用selected，如果没有则使用user_answer
    const userAnswer = question.selected !== undefined ? question.selected : question.user_answer;
    
    if (question.type === 'multiple_choice' && question.options) {
        if (userAnswer !== undefined && userAnswer !== null) {
            const answerIndex = typeof userAnswer === 'number' ? userAnswer : parseInt(userAnswer);
            if (answerIndex >= 0 && answerIndex < question.options.length) {
                return question.options[answerIndex];
            }
        }
        return '未回答';
    } else if (question.type === 'true_false') {
        if (userAnswer !== undefined && userAnswer !== null) {
            return userAnswer === 'true' || userAnswer === true ? '正确' : '错误';
        }
        return '未回答';
    } else if (question.type === 'short_answer') {
        return question.user_answer || '未回答';
    }
    
    // 如果都没有，尝试从user_answer获取
    if (question.user_answer !== undefined && question.user_answer !== null) {
        if (question.type === 'multiple_choice' && question.options) {
            const answerIndex = typeof question.user_answer === 'number' ? question.user_answer : parseInt(question.user_answer);
            if (answerIndex >= 0 && answerIndex < question.options.length) {
                return question.options[answerIndex];
            }
        }
        return String(question.user_answer);
    }
    
    return '未回答';
}

function addQuizMistake(questionIndex, button) {
    // 移除
}

// 查看测验历史
function viewQuizHistory() {
    document.getElementById('quizGenerator').style.display = 'none';
    document.getElementById('quizTaking').style.display = 'none';
    document.getElementById('quizResult').style.display = 'none';
    document.getElementById('quizHistory').style.display = 'block';
    
    // 加载历史记录
    const historyList = document.getElementById('quizHistoryList');
    historyList.innerHTML = '<div class="loading-spinner"></div>';
    
    apiRequest(`/api/quiz/history?user_id=${getCurrentUserId()}`, { method: 'GET' })
    .then(data => {
        if (data.status === 'success' && data.history) {
            if (data.history.length === 0) {
                historyList.innerHTML = '<p class="empty-state">暂无测验历史</p>';
                return;
            }
            
            let historyHtml = '';
            data.history.forEach(quiz => {
                const ts = quiz.timestamp || quiz.created_at || quiz.completed_at || new Date().toISOString();
                const date = new Date(ts).toLocaleString('zh-CN');
                const score = quiz.score !== undefined && quiz.score !== null ? `${quiz.score}%` : '未评分';
                const title = quiz.title || (quiz.subject ? `测验：${quiz.subject}` : '测验');
                const qcount = quiz.questions_count || quiz.num_questions || 0;
                historyHtml += `
                    <div class="history-item" data-quiz-id="${quiz.quiz_id || ''}">
                        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
                            <h4 style="margin:0;">${title}</h4>
                            <div class="history-actions">
                                ${quiz.subject ? `<button class="ghost-btn compact" onclick="startQuizForSubject('${encodeURIComponent(quiz.subject)}')">重做</button>` : ''}
                                <button class="secondary-btn" style="padding:6px 10px;font-size:0.9em;" onclick="viewQuizDetail('${quiz.quiz_id || ''}')">查看详情</button>
                                <button class="link-danger" onclick="deleteQuiz('${quiz.quiz_id || ''}')">删除</button>
                            </div>
                        </div>
                        <div class="history-details">
                            <span>日期：${date}</span>
                            <span>得分：${score}</span>
                            <span>问题数：${qcount}</span>
                        </div>
                    </div>
                `;
            });
            historyList.innerHTML = historyHtml;
        } else {
            historyList.innerHTML = '<p class="error-state">加载历史记录失败</p>';
        }
    })
    .catch(error => {
        historyList.innerHTML = '<p class="error-state">加载失败</p>';
    });
}

function viewQuizDetail(quizId) {
    if (!quizId) {
        showError('缺少测验ID，无法查看详情');
        return;
    }
    
    const historyList = document.getElementById('quizHistoryList');
    if (historyList) {
        historyList.innerHTML = '<div class="loading-spinner"></div><p>正在加载测验详情...</p>';
    }
    
    apiRequest(`/api/quiz/detail?user_id=${getCurrentUserId()}&quiz_id=${encodeURIComponent(quizId)}`, { method: 'GET' })
    .then(data => {
        if (data.status === 'success' && data.result) {
            document.getElementById('quizHistory').style.display = 'none';
            document.getElementById('quizTaking').style.display = 'none';
            document.getElementById('quizGenerator').style.display = 'none';
            document.getElementById('quizResult').style.display = 'block';
            renderQuizResult(data.result);
        } else {
            showError('加载测验详情失败: ' + (data.message || '未知错误'));
            viewQuizHistory();
        }
    })
    .catch(error => {
        showError('加载测验详情失败: ' + error.message);
        viewQuizHistory();
    });
}

// 复习功能实现
let currentReview = null;

function normalizeReviewApiPayload(data) {
    if (!data || typeof data !== 'object') {
        return { status: 'error', message: '响应格式错误', review: null };
    }
    let normalized = { ...data };
    if (normalized.review && typeof normalized.review === 'object' && normalized.review.status) {
        const nested = normalized.review;
        normalized = {
            ...nested,
            review: nested.review || null,
            next_review_at: nested.next_review_at || normalized.next_review_at || null,
            message: nested.message || normalized.message || ''
        };
    }
    if (normalized.review && normalized.next_review_at && !normalized.review.next_review_at) {
        normalized.review.next_review_at = normalized.next_review_at;
    }
    return normalized;
}

// 获取下一次复习
function getNextReview() {
    const contentElement = document.getElementById('nextReviewContent');
    contentElement.innerHTML = '<div class="loading-spinner"></div><p>正在获取复习内容...</p>';
    
    apiRequest(`/api/review/next?user_id=${getCurrentUserId()}`, {
        method: 'GET'
    })
    .then(data => {
        const payload = normalizeReviewApiPayload(data);
        if (payload.status === 'success' && payload.review) {
            currentReview = payload.review;
            
            // 显示当前复习内容
            document.getElementById('nextReviewSection').style.display = 'none';
            document.getElementById('currentReviewSection').style.display = 'block';
            
            const reviewContentElement = document.getElementById('currentReviewContent');
            const scheduledText = currentReview.scheduled_at ? new Date(currentReview.scheduled_at).toLocaleString('zh-CN') : '未设置';
            const bodyText = (currentReview.content == null ? '' : String(currentReview.content)).trim();
            reviewContentElement.innerHTML = `
                <h4>${currentReview.subject || '复习内容'}</h4>
                <div class="review-content-text">${bodyText || '<span class="muted">（未填写复习内容）</span>'}</div>
                <div class="review-meta">
                    <span>重要性：${getImportanceText(currentReview.importance)}</span>
                    <span>安排时间：${scheduledText}</span>
                </div>
            `;
        } else if (payload.status === 'success' && payload.next_review_at) {
            const nextTime = new Date(payload.next_review_at).toLocaleString('zh-CN');
            contentElement.innerHTML = `<p class="empty-state">当前没有到期任务，下次复习时间：${nextTime}</p>`;
        } else {
            contentElement.innerHTML = '<p class="empty-state">暂无待复习内容</p>';
        }
    })
    .catch(error => {
        contentElement.innerHTML = '<p class="error-state">获取复习内容失败</p>';
    });
}

// 获取重要性文本
function getImportanceText(importance) {
    switch(importance) {
        case 'high': return '高';
        case 'medium': return '中';
        case 'low': return '低';
        default: return '未知';
    }
}

// 标记复习难度
function markReviewDifficulty(difficulty) {
    if (!currentReview) {
        showError('当前没有复习内容');
        return;
    }
    
    // 获取review_id，支持id或review_id字段
    const reviewId = currentReview.review_id || currentReview.id;
    if (!reviewId) {
        showError('复习ID缺失，无法提交');
        console.error('currentReview对象:', currentReview);
        return;
    }
    
    // 根据难度计算分数和信心值
    let score = 50; // 默认50分
    let confidence = 0.5; // 默认0.5
    if (difficulty === 'easy') {
        score = 90;
        confidence = 0.9;
    } else if (difficulty === 'medium') {
        score = 70;
        confidence = 0.7;
    } else if (difficulty === 'hard') {
        score = 40;
        confidence = 0.4;
    }
    
    const performance = {
        difficulty: difficulty,
        score: score,
        confidence: confidence,
        reviewed_at: new Date().toISOString()
    };
    
    apiRequest('/api/review/complete', {
        method: 'POST',
        body: JSON.stringify({
            user_id: getCurrentUserId(),
            review_id: reviewId,
            performance: performance
        })
    })
    .then(data => {
        if (data.status === 'success') {
            // 显示完成消息
            showSuccess('复习已完成！系统将根据您的表现安排下次复习时间。');
            
            // 重置状态
            currentReview = null;
            document.getElementById('currentReviewSection').style.display = 'none';
            document.getElementById('nextReviewSection').style.display = 'block';
            
            // 重新获取下一次复习
            getNextReview();
        } else {
            alert('提交失败: ' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        showError('发生错误: ' + error.message, error);
    });
}

// 加载复习历史（带热力图）
function loadReviewHistory() {
    // 显示历史列表，隐藏其他部分
    document.getElementById('reviewHistorySection').style.display = 'block';
    document.getElementById('nextReviewSection').style.display = 'none';
    document.getElementById('currentReviewSection').style.display = 'none';
    document.getElementById('reviewScheduleSection').style.display = 'none';
    
    const historyList = document.getElementById('reviewHistoryList');
    historyList.innerHTML = '<div class="loading-spinner"></div><p>正在加载复习计划...</p>';
    
    apiRequest(`/api/review/history?user_id=${getCurrentUserId()}&limit=100`)
    .then(data => {
        const history = Array.isArray(data.history) ? data.history : (data.history && Array.isArray(data.history.history) ? data.history.history : []);
        if (history.length > 0) {
            let historyHtml = '';
            
            // 1. 生成热力图
            historyHtml += generateReviewHeatmap(history);
            
            // 2. 生成列表
            historyHtml += '<div class="review-history-grid">';
            history.forEach(review => {
                const scheduledAt = review.scheduled_at ? new Date(review.scheduled_at).toLocaleString('zh-CN') : '未设置';
                const nextReviewAt = review.next_review_at ? new Date(review.next_review_at).toLocaleString('zh-CN') : '未设置';
                const lastReviewedAt = review.last_reviewed_at ? new Date(review.last_reviewed_at).toLocaleString('zh-CN') : '未复习';
                const status = review.status === 'active' ? '活跃' : review.status === 'mastered' ? '已掌握' : '未知';
                const importance = getImportanceText(review.importance);
                const mastery = Math.round((review.mastery_level || 0) * 100);
                const reviewCount = review.review_count || 0;
                const isOverdue = review.status === 'active' && review.next_review_at
                    && new Date(review.next_review_at) < new Date();
                
                // 渲染 Markdown 内容预览
                const rawContent = review.content || '无内容描述';
                const contentPreview = rawContent.length > 100 ? rawContent.substring(0, 100) + '...' : rawContent;
                const renderedContent = window.marked ? marked.parse(contentPreview) : contentPreview;
                
                historyHtml += `
                    <div class="review-card">
                        <div class="review-card-header">
                            <h4>${review.subject || '未命名主题'}</h4>
                            <span class="review-badges">
                                ${isOverdue ? '<span class="status-badge overdue">已逾期</span>' : ''}
                                <span class="status-badge ${review.status}">${status}</span>
                            </span>
                        </div>
                        <div class="review-card-content">
                            <div class="markdown-body" style="font-size:0.9em;margin-bottom:10px;">${renderedContent}</div>
                            <div class="review-meta">
                                <span><strong>重要性：</strong>${importance}</span>
                                <span><strong>复习次数：</strong>${reviewCount}</span>
                                <span><strong>掌握程度：</strong>${mastery}%</span>
                            </div>
                            <div class="review-dates">
                                <p><strong>安排时间：</strong>${scheduledAt}</p>
                                <p><strong>下次复习：</strong>${nextReviewAt}</p>
                                <p><strong>最后复习：</strong>${lastReviewedAt}</p>
                            </div>
                        </div>
                        <div class="review-card-actions">
                            ${review.status === 'active' ? `
                                <button class="primary-btn" onclick="startReview('${review.review_id}')">开始复习</button>
                            ` : ''}
                            <button class="ghost-btn compact" onclick="toggleReschedule('${review.review_id}')">
                                <i class="fas fa-clock"></i> 调整时间
                            </button>
                            <button class="link-danger" onclick="confirmDeleteReview('${review.review_id}', '${review.subject || '该复习计划'}')">删除</button>
                        </div>
                        <div class="review-reschedule" id="rs-${review.review_id}" hidden>
                            <div class="rs-quick">
                                <button class="chip-btn" onclick="postponeReview('${review.review_id}', 1)">推迟 1 天</button>
                                <button class="chip-btn" onclick="postponeReview('${review.review_id}', 3)">推迟 3 天</button>
                                <button class="chip-btn" onclick="postponeReview('${review.review_id}', 7)">推迟 1 周</button>
                            </div>
                            <div class="rs-custom">
                                <input type="datetime-local" id="rs-input-${review.review_id}">
                                <button class="primary-btn compact" onclick="saveReschedule('${review.review_id}')">保存</button>
                            </div>
                        </div>
                    </div>
                `;
            });
            historyHtml += '</div>';
            historyList.innerHTML = historyHtml;
        } else {
            historyList.innerHTML = '<p class="empty-state">暂无复习计划，点击上方"安排复习"按钮创建</p>';
        }
    })
    .catch(error => {
        historyList.innerHTML = '<p class="error-state">加载复习计划失败: ' + error.message + '</p>';
        showError('加载复习计划失败: ' + error.message, error);
    });
}

// 生成复习热力图
function generateReviewHeatmap(history) {
    // 统计过去365天的复习数据
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setDate(today.getDate() - 365);
    
    const stats = {};
    history.forEach(review => {
        if (review.performance_history) {
            review.performance_history.forEach(p => {
                const date = p.timestamp.split('T')[0];
                stats[date] = (stats[date] || 0) + 1;
            });
        }
    });
    
    // 生成网格
    let html = '<div class="heatmap-container" style="overflow-x:auto;padding:20px;background:var(--card-bg);border-radius:12px;margin-bottom:20px;box-shadow:0 4px 6px rgba(0,0,0,0.05);">';
    html += '<h4 style="margin-bottom:15px;">复习热力图</h4><div style="display:flex;gap:4px;">';
    
    // 简化版：只显示最近30天
    for (let i = 29; i >= 0; i--) {
        const d = new Date();
        d.setDate(today.getDate() - i);
        const dateStr = d.toISOString().split('T')[0];
        const count = stats[dateStr] || 0;
        let color = '#ebedf0'; // 0
        if (count > 0) color = '#9be9a8'; // 1-2
        if (count > 2) color = '#40c463'; // 3-5
        if (count > 5) color = '#30a14e'; // 6-10
        if (count > 10) color = '#216e39'; // >10
        
        html += `
            <div title="${dateStr}: ${count}次复习" style="width:20px;height:20px;background:${color};border-radius:2px;"></div>
        `;
    }
    
    html += '</div></div>';
    return html;
}

// 开始复习（从历史列表）
function startReview(reviewId) {
    if (!reviewId) return;
    
    // 隐藏其他部分，显示当前复习部分
    document.getElementById('reviewHistorySection').style.display = 'none';
    document.getElementById('nextReviewSection').style.display = 'none';
    document.getElementById('reviewScheduleSection').style.display = 'none';
    document.getElementById('currentReviewSection').style.display = 'block';
    
    const contentElement = document.getElementById('currentReviewContent');
    contentElement.innerHTML = '<div class="loading-spinner"></div><p>正在加载复习内容...</p>';
    
    // 获取复习详情
    apiRequest(`/api/review/get?user_id=${getCurrentUserId()}&review_id=${encodeURIComponent(reviewId)}`, {
        method: 'GET'
    })
    .then(data => {
        if (data.status === 'success' && data.review) {
            currentReview = data.review;
            
            contentElement.innerHTML = `
                <h4>${currentReview.subject || '复习内容'}</h4>
                <div class="review-content-text">${currentReview.content}</div>
                <div class="review-meta">
                    <span>重要性：${getImportanceText(currentReview.importance)}</span>
                    <span>安排时间：${new Date(currentReview.scheduled_at).toLocaleString('zh-CN')}</span>
                </div>
            `;
        } else {
            contentElement.innerHTML = '<p class="error-state">获取复习内容失败</p>';
            showError('获取复习内容失败: ' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        contentElement.innerHTML = '<p class="error-state">加载失败</p>';
        showError('加载失败: ' + error.message);
    });
}

// 确认删除复习计划
function confirmDeleteReview(reviewId, subject) {
    if (confirm(`确定要删除复习计划"${subject}"吗？\n\n此操作无法撤销。`)) {
        deleteReview(reviewId);
    }
}

// 删除复习计划
function deleteReview(reviewId) {
    apiRequest('/api/review/delete', {
        method: 'POST',
        body: JSON.stringify({
            user_id: getCurrentUserId(),
            review_id: reviewId
        })
    })
    .then(data => {
        if (data.status === 'success') {
            showSuccess('复习计划已成功删除');
            // 重新加载复习历史
            loadReviewHistory();
        } else {
            showError('删除失败: ' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        showError('删除复习计划失败: ' + error.message, error);
    });
}

// 创建新对话
async function createNewConversation() {
    try {
        switchPage('chat');

        const result = await apiRequest('/api/conversations', {
            method: 'POST',
            body: JSON.stringify({
                user_id: getCurrentUserId()
            })
        });
        
        currentConversationId = result.conversation_id || null;
        
        // 清空聊天容器，回到新会话欢迎空状态
        resetChatToWelcome();
        hidePlanInputGuide();
        const input = document.getElementById('userInput');
        if (input) input.focus();
        
        // 移除所有会话的激活状态
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // 重新加载会话列表
        loadConversationList();
    } catch (error) {
        console.error('创建新对话失败:', error);
        showError('创建新对话失败');
    }
}

// ===== 会话批量管理（全选 / 批量删除）=====
let batchMode = false;

function toggleBatchMode(force) {
    batchMode = (typeof force === 'boolean') ? force : !batchMode;
    const toolbar = document.getElementById('batchToolbar');
    const manageBtn = document.getElementById('manageConversationsBtn');
    if (toolbar) toolbar.style.display = batchMode ? 'flex' : 'none';
    if (manageBtn) {
        manageBtn.classList.toggle('active', batchMode);
        manageBtn.title = batchMode ? '退出管理' : '批量管理';
    }
    const selAll = document.getElementById('selectAllConversations');
    if (selAll) selAll.checked = false;
    const delBtn = document.getElementById('batchDeleteBtn');
    if (delBtn) delBtn.disabled = true;
    // 重新渲染列表以切换复选框显示
    loadConversationList();
}

function getSelectedConversationIds() {
    return Array.from(document.querySelectorAll('.conversation-item input[type="checkbox"]:checked'))
        .map(cb => cb.value);
}

function updateBatchDeleteButton() {
    const btn = document.getElementById('batchDeleteBtn');
    const selAll = document.getElementById('selectAllConversations');
    const boxes = Array.from(document.querySelectorAll('.conversation-item input[type="checkbox"]'));
    const checked = boxes.filter(b => b.checked).length;
    if (btn) {
        btn.disabled = checked === 0;
        btn.innerHTML = `<i class="fas fa-trash-alt"></i> 删除选中${checked ? ' (' + checked + ')' : ''}`;
    }
    if (selAll) selAll.checked = boxes.length > 0 && checked === boxes.length;
}

function toggleSelectAll(checked) {
    document.querySelectorAll('.conversation-item input[type="checkbox"]').forEach(cb => {
        cb.checked = checked;
    });
    updateBatchDeleteButton();
}

async function deleteSelectedConversations() {
    const ids = getSelectedConversationIds();
    if (!ids.length) return;
    if (!confirm(`确定要删除选中的 ${ids.length} 个会话吗？删除后无法恢复。`)) return;
    try {
        const result = await apiRequest('/api/conversations', {
            method: 'DELETE',
            body: JSON.stringify({ conversation_ids: ids, all: 1 })
        });
        if (result.status === 'success') {
            // 若当前会话被删除，回到新对话状态
            if (ids.includes(currentConversationId)) currentConversationId = null;
            showSuccess(`已删除 ${result.deleted != null ? result.deleted : ids.length} 个会话`);
            toggleBatchMode(false);
            return;
        }
        showError('删除失败: ' + (result.message || '未知错误'));
    } catch (e) {
        showError('删除失败: ' + e.message);
    }
}

// 加载会话列表
async function loadConversationList() {
    const container = document.getElementById('conversationList');
    if (!container) {
        console.error('找不到conversationList元素');
        return;
    }
    
    container.innerHTML = '<div class="loading-spinner"></div><p>正在加载会话列表...</p>';
    
    try {
        const data = await apiRequest(`/api/conversations?user_id=${getCurrentUserId()}&limit=50&all=1`);
        console.log('会话列表API响应:', data); // 调试信息
        
        if (data && data.status === 'success') {
            // 确保conversations是数组
            const conversations = Array.isArray(data.conversations) ? data.conversations : [];
            if (conversations.length > 0) {
                // 按时间分桶：今天 / 昨天 / 更早；无有效时间则不分组（保持原顺序）
                const _tsOf = c => {
                    const raw = c.updated_at || c.created_at || c.timestamp;
                    const t = raw ? new Date(raw).getTime() : NaN;
                    return Number.isFinite(t) ? t : NaN;
                };
                const hasTime = conversations.some(c => Number.isFinite(_tsOf(c)));
                const ordered = hasTime
                    ? conversations.slice().sort((a, b) => (_tsOf(b) || 0) - (_tsOf(a) || 0))
                    : conversations;
                const bucketOf = c => {
                    const t = _tsOf(c);
                    if (!Number.isFinite(t)) return '更早';
                    const now = new Date();
                    const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
                    if (t >= startToday) return '今天';
                    if (t >= startToday - 86400000) return '昨天';
                    return '更早';
                };
                // 只有一个时间桶时不显示分组标题（否则会出现孤零零的「更早」）
                const distinctBuckets = new Set(ordered.map(bucketOf));
                const showGroups = hasTime && distinctBuckets.size > 1;
                let html = '';
                let lastBucket = null;
                ordered.forEach(conv => {
                    const bucket = bucketOf(conv);
                    if (showGroups && bucket !== lastBucket) {
                        html += `<div class="conv-group-title">${bucket}</div>`;
                        lastBucket = bucket;
                    }
                    const isActive = conv.id === currentConversationId;
                    const safeId = escapeHtml(conv.id);
                    const safeTitle = escapeHtml(conv.title || '新对话');
                    if (batchMode) {
                        // 批量模式：显示复选框，点击整项切换勾选
                        html += `
                            <label class="conversation-item batch-item ${isActive ? 'active' : ''}"
                                   data-conversation-id="${safeId}">
                                <input type="checkbox" value="${safeId}" onchange="updateBatchDeleteButton()">
                                <span class="conversation-item-title">${safeTitle}</span>
                            </label>
                        `;
                    } else {
                        // 普通模式：参考 Codex，只显示标题；批量管理请点右上角按钮
                        html += `
                            <div class="conversation-item ${isActive ? 'active' : ''}"
                                 onclick="loadConversation('${safeId}')"
                                 data-conversation-id="${safeId}"
                                 title="${safeTitle}">
                                <span class="conversation-item-title">${safeTitle}</span>
                            </div>
                        `;
                    }
                });
                container.innerHTML = html;
                if (batchMode) updateBatchDeleteButton();
            } else {
                container.innerHTML = '<p class="empty-state" style="text-align: center; color: var(--text-light); padding: 20px;">暂无历史对话</p>';
            }
        } else {
            container.innerHTML = '<p class="empty-state" style="text-align: center; color: var(--danger-color); padding: 20px;">加载失败: ' + (data.message || '未知错误') + '</p>';
        }
    } catch (error) {
        console.error('加载会话列表失败:', error);
        const errorMessage = error.message || '未知错误';
        container.innerHTML = `<p class="empty-state" style="text-align: center; color: var(--danger-color); padding: 20px;">加载失败: ${errorMessage}</p>`;
    }
}

// 加载指定会话
async function loadConversation(conversationId) {
    try {
        currentConversationId = conversationId;
        
        // 更新激活状态
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.toggle('active', item.dataset.conversationId === conversationId);
        });
        
        // 加载会话详情
        const data = await apiRequest(`/api/conversations?user_id=${getCurrentUserId()}&conversation_id=${conversationId}&all=1`);
        
        if (data.status === 'success' && data.conversation) {
            const conversation = data.conversation;
            const chatContainer = document.getElementById('chatContainer');
            clearChatMessages();
            
            // 渲染会话中的所有消息
            conversation.messages.forEach(msg => {
                if (msg.role === 'user') {
                    addMessage(msg.content, true);
                } else if (msg.role === 'assistant') {
                    let meta = null;
                    if (msg.metadata && typeof msg.metadata === 'object') {
                        meta = msg.metadata;
                    } else {
                        meta = {};
                        Object.keys(msg).forEach(k => {
                            if (k !== 'role' && k !== 'content' && k !== 'timestamp') {
                                meta[k] = msg[k];
                            }
                        });
                    }

                    if (meta?.plan_clarification) {
                        addMessage(renderPlanClarification(meta.plan_clarification), false);
                        return;
                    }
                    if (meta && meta.plan && meta.plan.weekly_plan) {
                        const planHtml = renderPlan(meta.plan, meta.message || msg.content);
                        addMessage(planHtml, false);
                        return;
                    }

                    // 清理答案中的冗余内容
                    let cleanAnswer = msg.content.trim();
                    cleanAnswer = cleanAnswer.replace(/^我理解你想了解[^。]+。让我为你详细解答：\s*/gm, '');
                    cleanAnswer = cleanAnswer.replace(/^关于[^，]+，我可以告诉你：\s*/gm, '');
                    
                    const responseData = {
                        ...(meta || {}),
                        answer: cleanAnswer
                    };
                    const responseHTML = renderAnswerWithSources(responseData);
                    addMessage(responseHTML, false);
                }
            });
            
            // 历史会话为空时也回到欢迎状态
            syncChatEmptyState();
            
            // 滚动到底部
            chatContainer.scrollTop = chatContainer.scrollHeight;
        } else {
            showError('加载会话失败');
        }
    } catch (error) {
        console.error('加载会话失败:', error);
        showError('加载会话失败: ' + error.message);
    }
}

// 删除会话
async function deleteConversation(conversationId) {
    if (!confirm('确定要删除这个会话吗？删除后无法恢复。')) {
        return;
    }
    
    try {
        const result = await apiRequest('/api/conversations', {
            method: 'DELETE',
            body: JSON.stringify({
                user_id: getCurrentUserId(),
                conversation_id: conversationId,
                all: 1
            })
        });
        
        if (result.status === 'success') {
            // 如果删除的是当前会话，清空聊天容器
            if (conversationId === currentConversationId) {
                currentConversationId = null;
                resetChatToWelcome();
            }
            
            // 重新加载会话列表
            loadConversationList();
            showSuccess('删除成功');
        } else {
            showError('删除失败: ' + (result.message || '未知错误'));
        }
    } catch (error) {
        console.error('删除会话失败:', error);
        showError('删除失败: ' + error.message);
    }
}

// 查看历史对话（保留原有功能作为备用）
function viewChatHistory() {
    switchPage('chatHistory');
    loadChatHistory();
}

// 加载历史对话
async function loadChatHistory() {
    const container = document.getElementById('chatHistoryList');
    if (!container) return;
    
    container.innerHTML = '<div class="loading-spinner"></div><p>正在加载历史对话...</p>';
    
    try {
        const data = await apiRequest(`/api/qa/history?user_id=${getCurrentUserId()}&limit=100&all=1`);
        
        if (data.status === 'success' && data.history && data.history.length > 0) {
            let html = '<div class="chat-history-container">';
            
            // 按时间倒序排列
            const sortedHistory = data.history.sort((a, b) => {
                const timeA = new Date(a.timestamp || 0);
                const timeB = new Date(b.timestamp || 0);
                return timeB - timeA;
            });
            
            sortedHistory.forEach((item, index) => {
                const time = new Date(item.timestamp || Date.now()).toLocaleString('zh-CN', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                });
                
                // 清理答案中的冗余内容（复用之前的清理逻辑）
                let cleanAnswer = (item.answer || '').trim();
                cleanAnswer = cleanAnswer.replace(/^我理解你想了解[^。]+。让我为你详细解答：\s*/gm, '');
                cleanAnswer = cleanAnswer.replace(/^关于[^，]+，我可以告诉你：\s*/gm, '');
                cleanAnswer = cleanAnswer.replace(/^根据我对[^，]+的了解，关于'[^']+'，我可以这样为你解答：\s*/gm, '');
                cleanAnswer = cleanAnswer.replace(/\n\n学习建议：\s*\n\d+\.\s*[^\n]+(\n\d+\.\s*[^\n]+)*/g, '');
                cleanAnswer = cleanAnswer.replace(/\n\n你对这个概念还有什么具体的疑问吗？[^]*$/g, '');
                cleanAnswer = cleanAnswer.replace(/\n\n你还有其他想了解的方面吗？\s*$/g, '');
                cleanAnswer = cleanAnswer.replace(/\n{3,}/g, '\n\n').trim();
                
                // 提取元数据
                const metadata = {
                    subject: item.subject || '通用',
                    source: item.source || 'default',
                    source_label: item.source_label || '',
                    context_snippets: item.context_snippets || [],
                    plan_clarification: item.plan_clarification || item.metadata?.plan_clarification
                };
                
                // 学习计划类历史：复用前端 renderPlan 渲染结构化 plan，避免后端拼 HTML
                let answerContent;
                if (item.plan && item.plan.weekly_plan) {
                    answerContent = renderPlan(item.plan, item.answer || '');
                } else if (metadata.plan_clarification) {
                    answerContent = renderPlanClarification(metadata.plan_clarification);
                } else {
                    const responseData = {
                        answer: cleanAnswer,
                        ...metadata
                    };
                    answerContent = renderAnswerWithSources(responseData);
                }
                
                html += `
                    <div class="chat-history-item">
                        <div class="chat-history-header">
                            <div class="chat-history-meta">
                                <span class="chat-history-time">${time}</span>
                                <span class="chat-history-subject">${item.subject || '通用'}</span>
                                ${item.source_label ? `<span class="chat-history-source">${item.source_label}</span>` : ''}
                                ${item.user_id && item.user_id !== getCurrentUserId() ? `<span class="conversation-owner" title="来自账号 ${escapeHtml(item.user_id)}">${escapeHtml(item.user_id)}</span>` : ''}
                            </div>
                            <button class="chat-history-delete-btn" 
                                    data-timestamp="${(item.timestamp || '').replace(/"/g, '&quot;')}"
                                    data-question="${(item.question || '').replace(/"/g, '&quot;').replace(/'/g, '&#39;')}"
                                    data-user-id="${escapeHtml(item.user_id || 'default')}"
                                    onclick="deleteChatHistoryItem(this)"
                                    title="删除此对话记录">
                                <i class="fas fa-trash-alt"></i>
                            </button>
                        </div>
                        <div class="chat-history-question">
                            <div style="font-weight:bold;margin-bottom:8px;color:var(--primary-color);"><i class="fas fa-user-circle"></i> 您</div>
                            <div>${item.question || ''}</div>
                        </div>
                        <div class="chat-history-answer">
                            <div style="font-weight:bold;margin-bottom:8px;color:var(--secondary-color);"><i class="fas fa-robot"></i> 启发式导师</div>
                            <div>${answerContent}</div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
        } else {
            container.innerHTML = '<p class="empty-state">暂无历史对话记录</p>';
        }
    } catch (error) {
        console.error('加载历史对话失败:', error);
        container.innerHTML = '<p class="empty-state">加载失败: ' + error.message + '</p>';
    }
}

// 删除历史对话项
async function deleteChatHistoryItem(button) {
    const timestamp = button.dataset.timestamp;
    const question = button.dataset.question;
    
    if (!timestamp) {
        showError('缺少时间戳信息，无法删除');
        return;
    }
    
    // 获取问题文本用于确认提示
    const questionText = question ? (question.length > 50 ? question.substring(0, 50) + '...' : question) : '这条';
    const confirmMessage = `确定要删除"${questionText}"这条历史对话记录吗？\n\n删除后无法恢复。`;
    
    if (!confirm(confirmMessage)) {
        return;
    }
    
    // 禁用按钮，防止重复点击，显示加载状态
    const originalHTML = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    button.style.cursor = 'not-allowed';
    
    try {
        const result = await apiRequest('/api/qa/history', {
            method: 'DELETE',
            body: JSON.stringify({
                user_id: button.dataset.userId || getCurrentUserId(),
                timestamp: timestamp,
                question: question || null,
                all: 1
            })
        });
        
        if (result.status === 'success') {
            showSuccess('删除成功');
            // 延迟一下再重新加载，让用户看到成功提示
            setTimeout(() => {
                loadChatHistory();
            }, 300);
        } else {
            showError('删除失败: ' + (result.message || '未知错误'));
            // 恢复按钮状态
            button.disabled = false;
            button.innerHTML = originalHTML;
            button.style.cursor = 'pointer';
        }
    } catch (error) {
        console.error('删除历史对话失败:', error);
        showError('删除失败: ' + error.message);
        // 恢复按钮状态
        button.disabled = false;
        button.innerHTML = originalHTML;
        button.style.cursor = 'pointer';
    }
}

// 开始指定的复习（旧实现保留，不覆盖主 startReview）
function startReviewByNext(reviewId) {
    // 调用获取复习API，获取指定ID的复习内容
    apiRequest(`/api/review/next?user_id=${getCurrentUserId()}`)
    .then(data => {
        if (data.status === 'success' && data.review) {
            // 检查是否是我们要的复习
            const review = data.review;
            if (review.review_id === reviewId || review.id === reviewId) {
                currentReview = review;
                
                // 显示当前复习内容
                document.getElementById('reviewHistorySection').style.display = 'none';
                document.getElementById('nextReviewSection').style.display = 'none';
                document.getElementById('currentReviewSection').style.display = 'block';
                
                const reviewContentElement = document.getElementById('currentReviewContent');
                reviewContentElement.innerHTML = `
                    <h4>${currentReview.subject || '复习内容'}</h4>
                    <div class="review-content-text">${currentReview.content}</div>
                    <div class="review-meta">
                        <span>重要性：${getImportanceText(currentReview.importance)}</span>
                        <span>安排时间：${new Date(currentReview.scheduled_at).toLocaleString('zh-CN')}</span>
                    </div>
                `;
            } else {
                showError('无法找到指定的复习内容，请使用"获取待复习内容"功能');
            }
        } else {
            showError('获取复习内容失败');
        }
    })
    .catch(error => {
        showError('开始复习失败: ' + error.message, error);
    });
}

// 安排新复习
function scheduleNewReview() {
    // 隐藏历史列表，显示安排部分
    document.getElementById('reviewHistorySection').style.display = 'none';
    document.getElementById('nextReviewSection').style.display = 'none';
    document.getElementById('currentReviewSection').style.display = 'none';
    document.getElementById('reviewScheduleSection').style.display = 'block';
    // 滚动到安排复习部分
    document.getElementById('reviewScheduleSection').scrollIntoView({ behavior: 'smooth' });
}

// 确认安排复习
function confirmScheduleReview() {
    const subject = document.getElementById('reviewSubject').value.trim();
    const content = document.getElementById('reviewContent').value.trim();
    const importance = document.getElementById('reviewImportance').value;
    const firstAtEl = document.getElementById('reviewFirstAt');
    const firstAt = firstAtEl && firstAtEl.value ? firstAtEl.value : '';
    
    if (!subject || !content) {
        showError('请输入主题和内容');
        return;
    }
    
    apiRequest('/api/review/schedule', {
        method: 'POST',
        body: JSON.stringify({
            user_id: getCurrentUserId(),
            subject: subject,
            content: content,
            importance: importance,
            first_review_at: firstAt
        })
    })
    .then(data => {
        if (data.status === 'success') {
            showSuccess('复习计划已安排成功！');
            
            // 清空表单
            document.getElementById('reviewSubject').value = '';
            document.getElementById('reviewContent').value = '';
            if (firstAtEl) firstAtEl.value = '';
            
            // 尝试获取下一次复习
            getNextReview();
        } else {
            showError('安排失败: ' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        showError('发生错误: ' + error.message, error);
    });
}

// 设置管理员界面
function setupAdminInterface() {
    const role = localStorage.getItem('role');
    if (role === 'admin') {
        // 显示管理员菜单项
        const adminMenuItem = document.querySelector('.nav-item[data-page="admin"]');
        if (adminMenuItem) {
            adminMenuItem.style.display = 'flex';
        }
    }
}

// 显示管理员页面
function showAdminPage() {
    // 这里使用querySelector来选择admin页面下的.admin-container，而不是整个page div
    const page = document.getElementById('admin');
    if (!page) return;
    
    // 检查权限
    const role = localStorage.getItem('role');
    const container = page.querySelector('.admin-container');
    
    if (role !== 'admin') {
        container.innerHTML = '<div class="error-state">您没有权限访问此页面</div>';
        return;
    }
    
    // 初始化页面结构（index.html 里预置了 adminStats，不能用“是否存在adminStats”来判断）
    const needsInit = !container.querySelector('#adminUserList') || !container.querySelector('#adminFeedback') || !container.querySelector('#adminRagLab');
    if (needsInit) {
        container.innerHTML = `
            <div id="adminStats"><div class="loading-spinner"></div><p>正在加载统计数据...</p></div>
            <div id="adminUserList" style="margin-top: 30px;"><div class="loading-spinner"></div><p>正在加载用户列表...</p></div>
            <div id="adminFeedback" style="margin-top: 30px;"><div class="loading-spinner"></div><p>正在加载用户反馈...</p></div>
            <div id="adminRagLab" style="margin-top: 30px;"><div class="loading-spinner"></div><p>正在加载RAG实验台...</p></div>
        `;
    }
    
    // 加载统计数据和用户列表
    Promise.all([
        loadAdminStats(),
        loadUserList(),
        loadFeedbackPanel(),
        loadRagLabPanel()
    ]).catch(error => {
        console.error('加载管理页面失败:', error);
        container.innerHTML = '<p class="error-state">加载失败: ' + error.message + '</p>';
    });
}

async function loadFeedbackPanel() {
    const container = document.getElementById('adminFeedback');
    if (!container) return;
    container.innerHTML = `<div class="loading-spinner"></div><p>正在加载用户反馈...</p>`;
    try {
        const [statsResp, listResp] = await Promise.all([
            apiRequest('/api/admin/feedback/stats'),
            apiRequest('/api/admin/feedback?limit=30&offset=0')
        ]);

        const stats = statsResp && statsResp.stats ? statsResp.stats : null;
        const items = listResp && listResp.data ? listResp.data : [];

        const statsHtml = stats ? `
            <div class="dashboard-grid">
                <div class="dashboard-card">
                    <div class="card-title">反馈总数</div>
                    <div class="card-value">${stats.total}</div>
                </div>
                <div class="dashboard-card">
                    <div class="card-title">正向</div>
                    <div class="card-value">${stats.positive}</div>
                </div>
                <div class="dashboard-card">
                    <div class="card-title">负向</div>
                    <div class="card-value">${stats.negative}</div>
                </div>
            </div>
        ` : `<div class="error-state">反馈统计不可用</div>`;

        const rows = (items || []).map(x => {
            const created = x.created_at ? new Date(x.created_at).toLocaleString('zh-CN') : '';
            const ratingText = Number(x.rating) > 0 ? '有用' : '没用';
            const src = x.source || '';
            const q = (x.question || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const c = (x.comment || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `
                <tr>
                    <td style="white-space:nowrap;">${created}</td>
                    <td style="white-space:nowrap;">${ratingText}</td>
                    <td style="white-space:nowrap;">${src}</td>
                    <td style="max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${q}">${q}</td>
                    <td style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${c}">${c}</td>
                </tr>
            `;
        }).join('');

        container.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
                <h3 style="margin:0;"><i class="fas fa-comment-alt" style="color: var(--primary-color);"></i> 用户反馈</h3>
                <button class="secondary-btn" onclick="loadFeedbackPanel()"><i class="fas fa-sync"></i> 刷新</button>
            </div>
            <div style="margin-top:12px;">${statsHtml}</div>
            <div style="margin-top:15px;background:var(--card-bg);border-radius:12px;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,0.05);">
                <table style="width:100%;border-collapse:collapse;">
                    <thead>
                        <tr style="background:rgba(0,0,0,0.03);text-align:left;">
                            <th style="padding:10px;">时间</th>
                            <th style="padding:10px;">评价</th>
                            <th style="padding:10px;">来源</th>
                            <th style="padding:10px;">问题</th>
                            <th style="padding:10px;">备注</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows || `<tr><td colspan="5" style="padding:12px;color:var(--text-light);">暂无反馈</td></tr>`}
                    </tbody>
                </table>
            </div>
        `;
    } catch (e) {
        container.innerHTML = `<div class="error-state">加载反馈失败：${e && e.message ? e.message : '未知错误'}</div>`;
    }
}

async function loadRagLabPanel() {
    const container = document.getElementById('adminRagLab');
    if (!container) return;
    container.innerHTML = `<div class="loading-spinner"></div><p>正在加载RAG实验台...</p>`;
    try {
        const [statsResp, jobsResp] = await Promise.all([
            apiRequest('/api/admin/rag/stats'),
            apiRequest('/api/admin/rag/eval/jobs')
        ]);
        const stats = statsResp && statsResp.stats ? statsResp.stats : {};
        const jobs = jobsResp && jobsResp.data ? jobsResp.data : [];

        const jobRows = (jobs || []).slice(0, 8).map(j => {
            const created = j.created_at ? new Date(j.created_at).toLocaleString('zh-CN') : '';
            const status = j.status || '';
            const prog = j.progress ? `${j.progress.done || 0}/${j.progress.total || 0}` : '';
            const hit = j.summary && typeof j.summary.hit_rate === 'number' ? (j.summary.hit_rate * 100).toFixed(1) + '%' : '--';
            const avgRt = j.summary && typeof j.summary.avg_response_time === 'number' ? j.summary.avg_response_time.toFixed(2) + 's' : '--';
            return `
                <tr style="cursor:pointer;" onclick="showRagEvalJobDetail('${j.id}')">
                    <td style="padding:10px;white-space:nowrap;">${created}</td>
                    <td style="padding:10px;white-space:nowrap;">${status}</td>
                    <td style="padding:10px;white-space:nowrap;">${prog}</td>
                    <td style="padding:10px;white-space:nowrap;">${hit}</td>
                    <td style="padding:10px;white-space:nowrap;">${avgRt}</td>
                </tr>
            `;
        }).join('');

        container.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
                <h3 style="margin:0;"><i class="fas fa-flask" style="color: var(--primary-color);"></i> RAG知识库与评估实验台</h3>
                <div style="display:flex;gap:10px;flex-wrap:wrap;">
                    <button class="secondary-btn" onclick="loadRagLabPanel()"><i class="fas fa-sync"></i> 刷新</button>
                    <button class="secondary-btn" onclick="runRagEval()"><i class="fas fa-play"></i> 运行评估</button>
                    <button class="danger-btn" onclick="rebuildRag()"><i class="fas fa-database"></i> 重建索引</button>
                </div>
            </div>

            <div style="margin-top:12px;">
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-title">文档数</div>
                        <div class="card-value">${stats.total_documents !== undefined ? stats.total_documents : '--'}</div>
                    </div>
                    <div class="dashboard-card">
                        <div class="card-title">切片数</div>
                        <div class="card-value">${stats.total_chunks !== undefined ? stats.total_chunks : '--'}</div>
                    </div>
                    <div class="dashboard-card">
                        <div class="card-title">主题数</div>
                        <div class="card-value">${stats.subject_count !== undefined ? stats.subject_count : '--'}</div>
                    </div>
                </div>
            </div>

            <div style="margin-top:15px;">
                <details style="border:1px solid var(--border-color);border-radius:12px;background:var(--card-bg);padding:12px;box-shadow:0 4px 6px rgba(0,0,0,0.05);">
                    <summary style="cursor:pointer;font-weight:700;user-select:none;display:flex;align-items:center;gap:8px;">
                        <i class="fas fa-upload" style="color: var(--primary-color);"></i> 导入资料（写入 data/ingested/*.json）
                    </summary>
                    <div style="margin-top:12px;display:grid;grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));gap:14px;">
                        <div style="border:1px solid var(--border-color);border-radius:10px;padding:12px;">
                            <div style="font-weight:700;margin-bottom:10px;">导入URL</div>
                            <input id="ragIngestUrl" type="text" placeholder="https://example.com" style="width:100%;padding:10px;border-radius:8px;border:1px solid var(--border-color);margin-bottom:10px;">
                            <input id="ragIngestUrlSubject" type="text" placeholder="主题（默认：通用）" style="width:100%;padding:10px;border-radius:8px;border:1px solid var(--border-color);margin-bottom:10px;">
                            <button class="secondary-btn" onclick="ingestRagUrl()" style="width:100%;"><i class="fas fa-cloud-download-alt"></i> 抓取并保存</button>
                            <div id="ragIngestUrlStatus" style="margin-top:10px;color:var(--text-light);font-size:0.9em;"></div>
                        </div>
                        <div style="border:1px solid var(--border-color);border-radius:10px;padding:12px;">
                            <div style="font-weight:700;margin-bottom:10px;">导入文本</div>
                            <input id="ragIngestTextTitle" type="text" placeholder="标题（可选）" style="width:100%;padding:10px;border-radius:8px;border:1px solid var(--border-color);margin-bottom:10px;">
                            <input id="ragIngestTextSubject" type="text" placeholder="主题（默认：通用）" style="width:100%;padding:10px;border-radius:8px;border:1px solid var(--border-color);margin-bottom:10px;">
                            <textarea id="ragIngestText" placeholder="粘贴要导入的文本内容..." style="width:100%;min-height:120px;padding:10px;border-radius:8px;border:1px solid var(--border-color);"></textarea>
                            <button class="secondary-btn" onclick="ingestRagText()" style="width:100%;margin-top:10px;"><i class="fas fa-file-import"></i> 保存文本</button>
                            <div id="ragIngestTextStatus" style="margin-top:10px;color:var(--text-light);font-size:0.9em;"></div>
                        </div>
                    </div>
                    <div style="margin-top:10px;color:var(--text-light);font-size:0.9em;">保存后点击“重建索引”使其生效。</div>
                </details>
            </div>

            <div style="margin-top:15px;background:var(--card-bg);border-radius:12px;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,0.05);">
                <div style="padding:12px 12px 0 12px;color:var(--text-light);font-size:0.9em;">最近评估任务（点击查看详情）</div>
                <table style="width:100%;border-collapse:collapse;">
                    <thead>
                        <tr style="background:rgba(0,0,0,0.03);text-align:left;">
                            <th style="padding:10px;">时间</th>
                            <th style="padding:10px;">状态</th>
                            <th style="padding:10px;">进度</th>
                            <th style="padding:10px;">命中率</th>
                            <th style="padding:10px;">平均耗时</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${jobRows || `<tr><td colspan="5" style="padding:12px;color:var(--text-light);">暂无任务</td></tr>`}
                    </tbody>
                </table>
            </div>
            <div id="ragEvalDetail" style="margin-top:15px;"></div>
        `;
    } catch (e) {
        container.innerHTML = `<div class="error-state">加载RAG实验台失败：${e && e.message ? e.message : '未知错误'}</div>`;
    }
}

async function rebuildRag() {
    try {
        showSuccess('正在重建索引，请稍等...');
        const res = await apiRequest('/api/admin/rag/rebuild', { method: 'POST', body: JSON.stringify({}) }, 600000);
        if (res && res.status === 'success') {
            showSuccess('重建完成');
            loadRagLabPanel();
        } else {
            showError('重建失败');
        }
    } catch (e) {
        showError('重建失败: ' + (e && e.message ? e.message : '未知错误'), e);
    }
}

async function runRagEval() {
    try {
        const datasetPath = 'data/rag_testcases.jsonl';
        const limit = 30;
        showSuccess('已启动评估任务（可在下方任务列表看到进度）');
        await apiRequest('/api/admin/rag/eval/run', {
            method: 'POST',
            body: JSON.stringify({ dataset_path: datasetPath, limit: limit })
        }, 60000);
        setTimeout(() => loadRagLabPanel(), 1200);
    } catch (e) {
        showError('启动评估失败: ' + (e && e.message ? e.message : '未知错误'), e);
    }
}

async function ingestRagUrl() {
    const urlEl = document.getElementById('ragIngestUrl');
    const subjEl = document.getElementById('ragIngestUrlSubject');
    const statusEl = document.getElementById('ragIngestUrlStatus');
    if (!urlEl || !statusEl) return;
    const url = (urlEl.value || '').trim();
    const subject = subjEl ? (subjEl.value || '').trim() : '';
    if (!url) {
        statusEl.textContent = '请输入URL';
        return;
    }
    statusEl.textContent = '正在抓取并保存...';
    try {
        const resp = await apiRequest('/api/admin/rag/ingest/url', {
            method: 'POST',
            body: JSON.stringify({ url, subject })
        }, 120000);
        if (resp && resp.status === 'success') {
            statusEl.textContent = '已保存：' + (resp.data && resp.data.file_name ? resp.data.file_name : 'success');
        } else {
            statusEl.textContent = '保存失败';
        }
    } catch (e) {
        statusEl.textContent = '失败：' + (e && e.message ? e.message : '未知错误');
    }
}

async function ingestRagText() {
    const textEl = document.getElementById('ragIngestText');
    const subjEl = document.getElementById('ragIngestTextSubject');
    const titleEl = document.getElementById('ragIngestTextTitle');
    const statusEl = document.getElementById('ragIngestTextStatus');
    if (!textEl || !statusEl) return;
    const text = (textEl.value || '').trim();
    const subject = subjEl ? (subjEl.value || '').trim() : '';
    const title = titleEl ? (titleEl.value || '').trim() : '';
    if (!text || text.length < 10) {
        statusEl.textContent = '文本太短（至少10字符）';
        return;
    }
    statusEl.textContent = '正在保存...';
    try {
        const resp = await apiRequest('/api/admin/rag/ingest/text', {
            method: 'POST',
            body: JSON.stringify({ text, subject, title })
        }, 60000);
        if (resp && resp.status === 'success') {
            statusEl.textContent = '已保存：' + (resp.data && resp.data.file_name ? resp.data.file_name : 'success');
            textEl.value = '';
        } else {
            statusEl.textContent = '保存失败';
        }
    } catch (e) {
        statusEl.textContent = '失败：' + (e && e.message ? e.message : '未知错误');
    }
}

async function showRagEvalJobDetail(jobId) {
    const detail = document.getElementById('ragEvalDetail');
    if (!detail) return;
    detail.innerHTML = `<div class="loading-spinner"></div><p>正在加载任务详情...</p>`;
    try {
        const resp = await apiRequest(`/api/admin/rag/eval/jobs/${jobId}`);
        const job = resp && resp.data ? resp.data : null;
        if (!job) {
            detail.innerHTML = `<div class="error-state">任务不存在</div>`;
            return;
        }
        const sum = job.summary || {};
        const avg = sum.avg_metrics || {};
        const metricLines = Object.keys(avg).slice(0, 8).map(k => {
            const v = avg[k];
            return `<div style="display:flex;justify-content:space-between;gap:12px;"><span>${k}</span><span style="font-weight:600;">${(typeof v === 'number' ? v.toFixed(3) : v)}</span></div>`;
        }).join('');

        detail.innerHTML = `
            <div style="background:var(--card-bg);border-radius:12px;padding:18px;box-shadow:0 4px 6px rgba(0,0,0,0.05);">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
                    <div style="font-weight:700;">评估任务详情</div>
                    <button class="secondary-btn" onclick="showRagEvalJobDetail('${jobId}')"><i class="fas fa-sync"></i> 刷新</button>
                </div>
                <div style="margin-top:10px;color:var(--text-light);font-size:0.9em;">状态：${job.status}，进度：${(job.progress && job.progress.done) || 0}/${(job.progress && job.progress.total) || 0}</div>
                <div style="margin-top:12px;display:grid;grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));gap:12px;">
                    <div style="background:rgba(0,0,0,0.03);border-radius:10px;padding:12px;">
                        <div style="color:var(--text-light);font-size:0.85em;">命中率</div>
                        <div style="font-size:1.4em;font-weight:700;">${typeof sum.hit_rate === 'number' ? (sum.hit_rate * 100).toFixed(1) + '%' : '--'}</div>
                    </div>
                    <div style="background:rgba(0,0,0,0.03);border-radius:10px;padding:12px;">
                        <div style="color:var(--text-light);font-size:0.85em;">平均耗时</div>
                        <div style="font-size:1.4em;font-weight:700;">${typeof sum.avg_response_time === 'number' ? sum.avg_response_time.toFixed(2) + 's' : '--'}</div>
                    </div>
                    <div style="background:rgba(0,0,0,0.03);border-radius:10px;padding:12px;">
                        <div style="color:var(--text-light);font-size:0.85em;">GT相似度</div>
                        <div style="font-size:1.4em;font-weight:700;">${typeof sum.avg_ground_truth_jaccard === 'number' ? sum.avg_ground_truth_jaccard.toFixed(3) : '--'}</div>
                    </div>
                </div>
                <div style="margin-top:12px;border-top:1px solid var(--border-color);padding-top:12px;">
                    <div style="font-weight:700;margin-bottom:8px;">平均指标（Top 8）</div>
                    <div style="display:grid;grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));gap:8px;">
                        ${metricLines || '<div style="color:var(--text-light);">暂无指标</div>'}
                    </div>
                </div>
                ${job.error ? `<div style="margin-top:12px;color:#e74c3c;">错误：${job.error}</div>` : ''}
            </div>
        `;
        if (job.status === 'running') {
            setTimeout(() => showRagEvalJobDetail(jobId), 1500);
        }
    } catch (e) {
        detail.innerHTML = `<div class="error-state">加载详情失败：${e && e.message ? e.message : '未知错误'}</div>`;
    }
}

// 加载管理员统计数据
async function loadAdminStats() {
    const statsContainer = document.getElementById('adminStats');
    if (!statsContainer) return;
    
    try {
        const __t0 = performance.now();
        const data = await apiRequest('/api/admin/stats');
        const apiElapsed = Math.round(performance.now() - __t0);
        
        if (data.status === 'success') {
            statsContainer.innerHTML = `
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-title">总用户数</div>
                        <div class="card-value">${data.stats.user_count}</div>
                    </div>
                    <div class="dashboard-card">
                        <div class="card-title">总会话数</div>
                        <div class="card-value">${data.stats.conversation_count}</div>
                    </div>
                    <div class="dashboard-card">
                        <div class="card-title">总消息数</div>
                        <div class="card-value">${data.stats.message_count}</div>
                    </div>
                </div>
                
                <div class="admin-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                    <!-- 系统健康状态 -->
                    <div class="admin-section" style="padding: 20px; background-color: var(--card-bg); border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        <h3 style="margin-top: 0; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; margin-bottom: 15px;">
                            <i class="fas fa-heartbeat" style="color: #e74c3c;"></i> 系统健康度
                        </h3>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>数据库状态:</span>
                            <span style="color: var(--success-color); font-weight: bold;">正常</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>API响应时间:</span>
                            <span style="color: var(--success-color); font-weight: bold;">${apiElapsed}ms</span>
                        </div>
                        <div style="margin-top: 15px; font-size: 0.85em; color: var(--text-light);">
                            <i class="fas fa-clock"></i> 系统运行时间: 3天 5小时
                        </div>
                    </div>

                    <!-- 快速操作面板 -->
                    <div class="admin-section" style="padding: 20px; background-color: var(--card-bg); border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        <h3 style="margin-top: 0; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; margin-bottom: 15px;">
                            <i class="fas fa-tools" style="color: #3498db;"></i> 维护工具
                        </h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <button class="secondary-btn" onclick="backupDatabase()" style="text-align: center;">
                                <i class="fas fa-database"></i> 备份数据
                            </button>
                            <button class="secondary-btn" onclick="clearSystemCache()" style="text-align: center;">
                                <i class="fas fa-broom"></i> 清理缓存
                            </button>
                            <button class="secondary-btn" onclick="exportSystemLogs()" style="text-align: center;">
                                <i class="fas fa-file-export"></i> 导出日志
                            </button>
                            <button class="secondary-btn" onclick="reloadSystemConfig()" style="text-align: center;">
                                <i class="fas fa-sync"></i> 重载配置
                            </button>
                        </div>
                    </div>
                </div>

                <div class="admin-section" style="margin-top: 20px; padding: 25px; background-color: var(--card-bg); border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <h3 style="margin-top: 0; border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; font-size: 1.1em; color: var(--text-main);">
                        <i class="fas fa-server" style="color: var(--primary-color); margin-right: 10px;"></i>系统概览
                    </h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                        <div style="display: flex; align-items: flex-start;">
                            <div style="width: 40px; height: 40px; border-radius: 8px; background-color: rgba(52, 152, 219, 0.1); display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                                <i class="fas fa-database" style="color: #3498db; font-size: 1.2em;"></i>
                            </div>
                            <div>
                                <div style="font-size: 0.85em; color: var(--text-light); margin-bottom: 4px;">数据库位置</div>
                                <div style="font-size: 0.95em; color: var(--text-main); word-break: break-all; font-family: monospace;">${data.stats.db_path}</div>
                            </div>
                        </div>
                        
                        <div style="display: flex; align-items: flex-start;">
                            <div style="width: 40px; height: 40px; border-radius: 8px; background-color: rgba(46, 204, 113, 0.1); display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                                <i class="fas fa-clock" style="color: #2ecc71; font-size: 1.2em;"></i>
                            </div>
                            <div>
                                <div style="font-size: 0.85em; color: var(--text-light); margin-bottom: 4px;">服务器时间</div>
                                <div style="font-size: 0.95em; color: var(--text-main); font-weight: 500;">${new Date().toLocaleString('zh-CN')}</div>
                            </div>
                        </div>

                        <div style="display: flex; align-items: flex-start;">
                            <div style="width: 40px; height: 40px; border-radius: 8px; background-color: rgba(155, 89, 182, 0.1); display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                                <i class="fas fa-code-branch" style="color: #9b59b6; font-size: 1.2em;"></i>
                            </div>
                            <div>
                                <div style="font-size: 0.85em; color: var(--text-light); margin-bottom: 4px;">系统版本</div>
                                <div style="font-size: 0.95em; color: var(--text-main); font-weight: 500;">v1.2.0 (Enterprise)</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            statsContainer.innerHTML = '<p class="error-state">加载统计失败: ' + (data.message || '未知错误') + '</p>';
        }
    } catch (error) {
        console.error('加载管理数据失败:', error);
        statsContainer.innerHTML = '<p class="error-state">加载统计失败: ' + error.message + '</p>';
    }
}


// 加载用户列表
async function loadUserList() {
    const page = document.getElementById('admin');
    if (!page) return;
    
    // 确保有容器
    if (!document.getElementById('adminUserList')) {
        // 如果没有特定容器，可能stats还没加载完或者结构不对，等待一下
        return; 
    }
    
    const listContainer = document.getElementById('adminUserList');
    listContainer.innerHTML = '<div class="loading-spinner-small"></div> 正在加载用户列表...';
    
    try {
        const data = await apiRequest('/api/admin/users');
        
        if (data.status === 'success' && data.users) {
            let html = `
                <div class="admin-section" style="padding: 15px; background-color: var(--card-bg); border-radius: 8px;">
                    <h3 style="margin-top: 0; margin-bottom: 15px;">用户列表</h3>
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="background-color: rgba(0,0,0,0.05); text-align: left;">
                                    <th style="padding: 10px; border-bottom: 1px solid var(--border-color);">ID</th>
                                    <th style="padding: 10px; border-bottom: 1px solid var(--border-color);">账号</th>
                                    <th style="padding: 10px; border-bottom: 1px solid var(--border-color);">角色</th>
                                    <th style="padding: 10px; border-bottom: 1px solid var(--border-color);">注册时间</th>
                                    <th style="padding: 10px; border-bottom: 1px solid var(--border-color);">操作</th>
                                </tr>
                            </thead>
                            <tbody>
            `;
            
            data.users.forEach(user => {
                const roleClass = user.role === 'admin' ? 'color: var(--primary-color); font-weight: bold;' : '';
                const roleText = user.role === 'admin' ? '管理员' : '普通用户';
                
                html += `
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid var(--border-color);">${user.id}</td>
                        <td style="padding: 10px; border-bottom: 1px solid var(--border-color);">${user.account}</td>
                        <td style="padding: 10px; border-bottom: 1px solid var(--border-color); ${roleClass}">${roleText}</td>
                        <td style="padding: 10px; border-bottom: 1px solid var(--border-color);">${user.created_at}</td>
                        <td style="padding: 10px; border-bottom: 1px solid var(--border-color);">
                            <div style="display: flex; gap: 5px;">
                                <button class="secondary-btn" style="padding: 4px 8px; font-size: 0.85em;" onclick="viewUserStats('${user.id}', '${user.account}')" title="查看详情"><i class="fas fa-chart-bar"></i></button>
                                <button class="warning-btn" style="padding: 4px 8px; font-size: 0.85em;" onclick="resetUserPassword('${user.id}', '${user.account}')" title="重置密码"><i class="fas fa-key"></i></button>
                                ${user.role !== 'admin' ? `<button class="danger-btn" style="padding: 4px 8px; font-size: 0.85em;" onclick="deleteUser('${user.id}', '${user.account}')" title="删除用户"><i class="fas fa-trash"></i></button>` : ''}
                            </div>
                        </td>
                    </tr>
                `;
            });
            
            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
            listContainer.innerHTML = html;
        } else {
            listContainer.innerHTML = '<p class="error-state">加载用户列表失败</p>';
        }
    } catch (error) {
        console.error('加载用户列表失败:', error);
        listContainer.innerHTML = '<p class="error-state">加载用户列表失败: ' + error.message + '</p>';
    }
}

// 删除用户
async function deleteUser(userId, account) {
    if (!confirm(`确定要删除用户 "${account}" 吗？\n此操作不可恢复，用户的所有数据将被清除。`)) {
        return;
    }
    
    try {
        const result = await apiRequest(`/api/admin/users/${userId}`, {
            method: 'DELETE'
        });
        
        if (result.status === 'success') {
            showSuccess('用户删除成功');
            loadUserList(); // 重新加载列表
            loadAdminStats(); // 重新加载统计
        } else {
            showError('删除失败: ' + (result.message || '未知错误'));
        }
    } catch (error) {
        showError('删除失败: ' + error.message);
    }
}

// 重置用户密码
async function resetUserPassword(userId, account) {
    const newPassword = prompt(`请输入用户 "${account}" 的新密码:`);
    if (newPassword === null) return; // 用户取消
    
    if (!newPassword.trim()) {
        showError('密码不能为空');
        return;
    }
    
    try {
        const result = await apiRequest(`/api/admin/users/${userId}/reset_password`, {
            method: 'POST',
            body: JSON.stringify({ password: newPassword })
        });
        
        if (result.status === 'success') {
            showSuccess('密码重置成功');
        } else {
            showError('重置失败: ' + (result.message || '未知错误'));
        }
    } catch (error) {
        showError('重置失败: ' + error.message);
    }
}

// 查看用户详细统计
async function viewUserStats(userId, account) {
    try {
        const data = await apiRequest(`/api/admin/users/${userId}/stats`);
        
        if (data.status === 'success') {
            const stats = data.stats;
            alert(`用户 "${account}" 的详细统计：\n\n` +
                  `📊 对话总数：${stats.conversation_count}\n` +
                  `💬 消息总数：${stats.message_count}\n` +
                  `📅 学习计划：${stats.plan_count}\n` +
                  `📈 学习进度：${stats.progress_count}`);
        } else {
            showError('获取详情失败: ' + (data.message || '未知错误'));
        }
    } catch (error) {
        showError('获取详情失败: ' + error.message);
    }
}

// 在switchPage函数中添加测验和复习页面的加载逻辑
function switchPage(pageId) {
    if (pageId !== 'chat') hidePlanInputGuide();
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
    }
    
    const activeNavItem = document.querySelector(`.nav-item[data-page="${pageId}"]`);
    if (activeNavItem) {
        activeNavItem.classList.add('active');
    }
    
    const loaded = pageLoadedState[pageId] === true;
    const runOnce = !loaded;

    if (pageId === 'plans' && runOnce) loadPlans();
    if (pageId === 'resources' && runOnce) loadResources();
    if (pageId === 'progress' && runOnce) loadProgress();
    if (pageId === 'admin' && runOnce) showAdminPage();
    if (pageId === 'profile' && runOnce) loadUserProfile();
    if (pageId === 'flashcards' && runOnce) loadFlashcards();
    if (pageId === 'quiz' && runOnce) generateNewQuiz();
    if (pageId === 'dashboard') updateDashboardData();
    if (pageId === 'notes') {
        initMonaco();
        if (runOnce) loadNotes();
    }
    if (pageId === 'review' && runOnce) {
        document.getElementById('nextReviewSection').style.display = 'block';
        document.getElementById('currentReviewSection').style.display = 'none';
        getNextReview();
    }
    if (pageId === 'chatHistory' && runOnce) loadChatHistory();
    if (pageId === 'chat') {
        loadConversationList();
        syncChatEmptyState();
    }

    pageLoadedState[pageId] = true;
}

// 初始化移动端导航
function setupMobileNavigation() {
    if (window.innerWidth <= 768) {
        const sidebar = document.querySelector('.sidebar');
        sidebar.classList.remove('open');
    }
}



// 备份数据库
async function backupDatabase() {
    if (!confirm('确定要备份当前数据库吗？')) return;
    
    try {
        showSuccess('正在备份数据库...');
        const data = await apiRequest('/api/admin/system/backup', { method: 'POST' });
        
        if (data.status === 'success') {
            alert(`备份成功！\n文件路径: ${data.path}`);
        } else {
            showError('备份失败: ' + (data.message || '未知错误'));
        }
    } catch (error) {
        showError('备份失败: ' + error.message);
    }
}

// 导出系统日志
async function exportSystemLogs() {
    try {
        const data = await apiRequest('/api/admin/system/logs');
        
        if (data.status === 'success') {
            // 创建一个Blob对象并下载
            const blob = new Blob([data.logs], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `system_logs_${new Date().toISOString().slice(0,10)}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            showSuccess('日志已开始下载');
        } else {
            showError('导出失败: ' + (data.message || '未知错误'));
        }
    } catch (error) {
        showError('导出失败: ' + error.message);
    }
}

// 清理系统缓存（模拟）
function clearSystemCache() {
    showSuccess('正在清理缓存...');
    setTimeout(() => {
        showSuccess('系统缓存已清理完成');
    }, 1500);
}

// 重载系统配置（模拟）
function reloadSystemConfig() {
    if (!confirm('确定要重载系统配置吗？这可能会导致短暂的服务中断。')) return;
    
    showSuccess('正在重载配置...');
    setTimeout(() => {
        showSuccess('系统配置已更新');
        location.reload();
    }, 2000);
}

// Monaco Editor 初始化
async function initMonaco() {
    if (noteEditorInstance) return noteEditorInstance;
    const container = document.getElementById('noteContent');
    if (!container) return null;

    try {
        await ensureMonacoLoaded();
    } catch (e) {
        showError('笔记编辑器加载失败，请稍后重试');
        return null;
    }

    const newContainer = document.createElement('div');
    newContainer.id = 'monacoEditorContainer';
    newContainer.style.height = '300px';
    newContainer.style.border = '1px solid var(--border-color)';
    newContainer.style.borderRadius = '8px';
    newContainer.style.marginTop = '10px';
    container.parentNode.replaceChild(newContainer, container);

    noteEditorInstance = monaco.editor.create(newContainer, {
        value: '',
        language: 'markdown',
        theme: document.documentElement.getAttribute('data-theme') === 'dark' ? 'vs-dark' : 'vs',
        automaticLayout: true,
        minimap: { enabled: false },
        fontSize: 14,
        lineHeight: 22
    });

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'data-theme') {
                const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
                monaco.editor.setTheme(isDark ? 'vs-dark' : 'vs');
            }
        });
    });
    observer.observe(document.documentElement, { attributes: true });
    return noteEditorInstance;
}

// 番茄钟功能
function setupPomodoro() {
    updatePomodoroDisplay();
}

function togglePomodoroModal() {
    const modal = document.getElementById('pomodoroModal');
    if (modal.style.display === 'flex') {
        modal.style.display = 'none';
    } else {
        modal.style.display = 'flex';
    }
}

function setPomodoroTime(minutes) {
    stopPomodoro();
    pomodoroTimeLeft = minutes * 60;
    pomodoroTotalTime = pomodoroTimeLeft;
    pomodoroMode = minutes === 25 ? 'focus' : 'break';
    updatePomodoroDisplay();
}

function updatePomodoroDisplay() {
    const minutes = Math.floor(pomodoroTimeLeft / 60);
    const seconds = pomodoroTimeLeft % 60;
    const timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    const timerDisplay = document.getElementById('pomodoroTimer');
    const floatDisplay = document.getElementById('pomodoroFloatTime');
    
    if (timerDisplay) timerDisplay.textContent = timeString;
    if (floatDisplay) floatDisplay.textContent = timeString;
    
    // 更新网页标题
    if (pomodoroIsRunning) {
        document.title = `(${timeString}) 知栈`;
    } else {
        document.title = '知栈 · 学习工作台';
    }
}

function startPomodoro() {
    if (pomodoroIsRunning) {
        stopPomodoro();
        return;
    }
    
    const startBtn = document.getElementById('pomodoroStartBtn');
    startBtn.innerHTML = '<i class="fas fa-pause"></i> 暂停';
    startBtn.className = 'warning-btn'; // 变为黄色警告色
    
    pomodoroIsRunning = true;
    
    // 播放白噪音
    const soundToggle = document.getElementById('pomodoroSoundToggle');
    const rainSound = document.getElementById('rainSound');
    if (soundToggle && soundToggle.checked && rainSound) {
        rainSound.volume = 0.3;
        rainSound.play().catch(e => console.log('Audio play failed:', e));
    }
    
    pomodoroTimer = setInterval(() => {
        pomodoroTimeLeft--;
        updatePomodoroDisplay();
        
        if (pomodoroTimeLeft <= 0) {
            finishPomodoro();
        }
    }, 1000);
}

function stopPomodoro() {
    clearInterval(pomodoroTimer);
    pomodoroIsRunning = false;
    
    const startBtn = document.getElementById('pomodoroStartBtn');
    if (startBtn) {
        startBtn.innerHTML = '<i class="fas fa-play"></i> 开始';
        startBtn.className = 'primary-btn';
    }
    
    const rainSound = document.getElementById('rainSound');
    if (rainSound) {
        rainSound.pause();
    }
    
    updatePomodoroDisplay();
}

function resetPomodoro() {
    stopPomodoro();
    // 恢复到当前模式的默认时间
    if (pomodoroMode === 'focus') pomodoroTimeLeft = 25 * 60;
    else if (pomodoroTotalTime < 20 * 60) pomodoroTimeLeft = 5 * 60; // 短休
    else pomodoroTimeLeft = 15 * 60; // 长休
    
    updatePomodoroDisplay();
}

function finishPomodoro() {
    stopPomodoro();
    
    // 播放提示音
    const bellSound = document.getElementById('bellSound');
    if (bellSound) {
        bellSound.play().catch(e => console.log('Bell play failed:', e));
    }
    
    // 如果是专注模式，记录学习时间
    if (pomodoroMode === 'focus') {
        const duration = Math.floor(pomodoroTotalTime / 60);
        showSuccess(`恭喜！完成了 ${duration} 分钟的专注学习！`);
        
        // 调用API记录时间
        apiRequest('/api/progress/update', {
            method: 'POST',
            body: JSON.stringify({
                user_id: getCurrentUserId(),
                subject: '专注模式',
                study_time: duration,
                completed_tasks: 0,
                total_tasks: 0,
                notes: [{ note: `完成一次番茄专注 (${new Date().toLocaleString()})` }]
            })
        }).then(() => {
            updateDashboardData(); // 刷新数据
        });
    } else {
        showSuccess('休息结束，准备开始新的学习吧！');
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', function() {
    initApp();
    setupMobileNavigation();
    
    // 延迟加载dashboard数据，避免与会话列表加载竞争
    setTimeout(function() {
        if (document.getElementById('dashboard').classList.contains('active')) {
            updateDashboardData();
        }
    }, 1000);
    
    // 检查聊天页面是否在初始加载时就已经激活，如果是则加载会话列表
    if (document.getElementById('chat').classList.contains('active')) {
        loadConversationList();
    }

    // 点击模态框外部关闭
    window.onclick = function(event) {
        const pomodoroModal = document.getElementById('pomodoroModal');
        const flashcardModal = document.getElementById('createFlashcardModal');
        
        if (event.target == pomodoroModal) {
            pomodoroModal.style.display = "none";
        }
        if (event.target == flashcardModal) {
            flashcardModal.style.display = "none";
        }
    }
});

// ===== 复习时间调整（2026-11-20） =====
function toggleReschedule(reviewId) {
    const row = document.getElementById(`rs-${reviewId}`);
    if (!row) return;
    row.hidden = !row.hidden;
    if (!row.hidden) {
        const input = document.getElementById(`rs-input-${reviewId}`);
        if (input && !input.value) {
            const d = new Date(Date.now() + 86400000);
            const pad = n => String(n).padStart(2, '0');
            input.value = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
        }
        input && input.focus();
    }
}

function saveReschedule(reviewId) {
    const input = document.getElementById(`rs-input-${reviewId}`);
    const value = input && input.value ? input.value : '';
    if (!value) { showError('请先选择时间'); return; }
    submitReschedule(reviewId, { next_review_at: value });
}

function postponeReview(reviewId, days) {
    submitReschedule(reviewId, { postpone_days: days });
}

function submitReschedule(reviewId, payload) {
    apiRequest(`/api/review/${reviewId}/reschedule`, {
        method: 'POST',
        body: JSON.stringify(Object.assign({ user_id: getCurrentUserId() }, payload))
    })
    .then(data => {
        if (data && data.status === 'success') {
            showSuccess('复习时间已调整');
            if (typeof loadReviewHistory === 'function') loadReviewHistory();
            if (typeof loadProgressOverview === 'function') loadProgressOverview();
        } else {
            showError('调整失败: ' + ((data && data.message) || '未知错误'));
        }
    })
    .catch(err => showError('调整失败: ' + (err.message || '网络错误')));
}

// ===== 回答区增强：代码块复制按钮 =====
function enhanceRenderedMarkdown(container) {
    if (!container) return;
    container.querySelectorAll('pre').forEach(pre => {
        if (pre.querySelector('.copy-code-btn')) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'copy-code-btn';
        btn.textContent = '复制';
        btn.addEventListener('click', () => {
            const codeEl = pre.querySelector('code');
            const text = codeEl ? codeEl.innerText : pre.innerText;
            const done = () => {
                btn.textContent = '已复制';
                setTimeout(() => { btn.textContent = '复制'; }, 1200);
            };
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(done).catch(done);
            } else {
                const ta = document.createElement('textarea');
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                try { document.execCommand('copy'); } catch (e) { /* 忽略 */ }
                document.body.removeChild(ta);
                done();
            }
        });
        pre.appendChild(btn);
    });
}

// ===== 个人中心：资料加载与签到 =====
const PROFILE_AVATAR_PRESETS = [
    { value: 'default', icon: 'fas fa-user', label: '默认' },
    { value: 'graduate', icon: 'fas fa-graduation-cap', label: '学生' },
    { value: 'book', icon: 'fas fa-book-open', label: '阅读' },
    { value: 'code', icon: 'fas fa-code', label: '编程' },
    { value: 'rocket', icon: 'fas fa-rocket', label: '进阶' },
    { value: 'star', icon: 'fas fa-star', label: '闪耀' }
];

let currentProfileAvatar = 'default';
let currentUploadAvatarFile = null;
let profileAvatarObjectUrl = null;

function getProfileAvatarIcon(avatar) {
    const preset = PROFILE_AVATAR_PRESETS.find(item => item.value === avatar);
    return preset ? preset.icon : 'fas fa-user';
}

function getProfileAvatarUrl(profile) {
    const userId = encodeURIComponent((profile && profile.user_id) || getCurrentUserId());
    const version = encodeURIComponent((profile && profile.avatar_version) || Date.now());
    return `/api/avatar?user_id=${userId}&v=${version}`;
}

function renderProfileAvatar(profile) {
    const el = document.getElementById('profileAvatar');
    if (!el) return;
    if (profile && profile.avatar === 'custom') {
        el.innerHTML = `<img src="${getProfileAvatarUrl(profile)}" alt="头像">`;
    } else {
        el.innerHTML = `<i class="${getProfileAvatarIcon(profile && profile.avatar)}"></i>`;
    }
}

function renderProfileAvatarPreview(profile) {
    const el = document.getElementById('profileAvatarPreview');
    if (!el) return;
    if (profile && profile.avatar === 'custom') {
        el.innerHTML = `<img src="${getProfileAvatarUrl(profile)}" alt="头像预览">`;
    } else {
        el.innerHTML = `<i class="${getProfileAvatarIcon(profile && profile.avatar)}"></i>`;
    }
}

function renderProfileAvatarOptions(selected) {
    const box = document.getElementById('profileAvatarOptions');
    if (!box) return;
    box.innerHTML = PROFILE_AVATAR_PRESETS.map(item => {
        const active = item.value === selected ? ' selected' : '';
        return `<button type="button" class="avatar-option${active}" data-avatar="${item.value}" title="${item.label}" onclick="selectProfileAvatar('${item.value}')"><i class="${item.icon}"></i></button>`;
    }).join('');
}

function selectProfileAvatar(value) {
    const preset = PROFILE_AVATAR_PRESETS.find(item => item.value === value);
    if (!preset) return;
    currentUploadAvatarFile = null;
    const fileInput = document.getElementById('profileAvatarFile');
    if (fileInput) fileInput.value = '';
    if (profileAvatarObjectUrl) {
        URL.revokeObjectURL(profileAvatarObjectUrl);
        profileAvatarObjectUrl = null;
    }
    const options = document.querySelectorAll('#profileAvatarOptions .avatar-option');
    options.forEach(item => item.classList.toggle('selected', item.dataset.avatar === value));
    const preview = document.getElementById('profileAvatarPreview');
    if (preview) preview.innerHTML = `<i class="${preset.icon}"></i>`;
}

async function openProfileEditModal() {
    try {
        const data = await apiRequest(`/api/auth/profile?user_id=${encodeURIComponent(getCurrentUserId())}`);
        const profile = data && data.status === 'success' && data.data ? data.data : {};
        const nicknameEl = document.getElementById('profileEditNickname');
        if (nicknameEl) nicknameEl.value = profile.nickname || '';
        currentProfileAvatar = profile.avatar || 'default';
        currentUploadAvatarFile = null;
        const fileInput = document.getElementById('profileAvatarFile');
        if (fileInput) fileInput.value = '';
        const avatar = PROFILE_AVATAR_PRESETS.some(item => item.value === currentProfileAvatar) ? currentProfileAvatar : 'default';
        renderProfileAvatarOptions(avatar);
        renderProfileAvatarPreview(profile);
        const modal = document.getElementById('profileEditModal');
        if (modal) modal.style.display = 'flex';
    } catch (e) {
        if (typeof showError === 'function') showError('加载个人资料失败: ' + (e.message || '网络错误'));
    }
}

function closeProfileEditModal() {
    const modal = document.getElementById('profileEditModal');
    if (modal) modal.style.display = 'none';
    currentUploadAvatarFile = null;
    if (profileAvatarObjectUrl) {
        URL.revokeObjectURL(profileAvatarObjectUrl);
        profileAvatarObjectUrl = null;
    }
}

function previewProfileAvatar(event) {
    const file = event && event.target && event.target.files ? event.target.files[0] : null;
    if (!file) return;

    const allowedTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
    const allowedExt = /\.(png|jpe?g|gif|webp)$/i.test(file.name || '');
    if (!allowedTypes.includes(file.type) && !allowedExt) {
        if (typeof showError === 'function') showError('仅支持 PNG、JPG、GIF、WEBP 图片');
        event.target.value = '';
        return;
    }
    if (file.size > 2 * 1024 * 1024) {
        if (typeof showError === 'function') showError('头像不能超过 2MB');
        event.target.value = '';
        return;
    }

    currentUploadAvatarFile = file;
    const options = document.querySelectorAll('#profileAvatarOptions .avatar-option');
    options.forEach(item => item.classList.remove('selected'));

    if (profileAvatarObjectUrl) URL.revokeObjectURL(profileAvatarObjectUrl);
    profileAvatarObjectUrl = URL.createObjectURL(file);
    const preview = document.getElementById('profileAvatarPreview');
    if (preview) preview.innerHTML = `<img src="${profileAvatarObjectUrl}" alt="头像预览">`;
}

async function uploadAvatarFile(nickname, file) {
    const form = new FormData();
    form.append('user_id', getCurrentUserId());
    form.append('nickname', nickname);
    form.append('file', file, file.name || 'avatar.png');

    const response = await fetch('/api/profile/avatar', {
        method: 'POST',
        body: form
    });
    const data = await response.json().catch(() => null);
    if (!data || data.status !== 'success') {
        if (typeof showError === 'function') showError((data && data.message) || '头像上传失败');
        return null;
    }
    return data;
}

async function saveProfileEdit() {
    const nicknameEl = document.getElementById('profileEditNickname');
    const nickname = (nicknameEl && nicknameEl.value || '').trim();
    if (!nickname) {
        if (typeof showError === 'function') showError('昵称不能为空');
        return;
    }
    if (nickname.length > 20) {
        if (typeof showError === 'function') showError('昵称最多 20 个字符');
        return;
    }

    const avatarFileInput = document.getElementById('profileAvatarFile');
    if (!currentUploadAvatarFile && avatarFileInput && avatarFileInput.files && avatarFileInput.files[0]) {
        previewProfileAvatar({ target: avatarFileInput });
        if (!currentUploadAvatarFile) return;
    }

    if (currentUploadAvatarFile) {
        try {
            const uploaded = await uploadAvatarFile(nickname, currentUploadAvatarFile);
            if (!uploaded) return;
            closeProfileEditModal();
            if (typeof showSuccess === 'function') showSuccess('头像和资料已更新');
            await loadProfilePage();
        } catch (e) {
            if (typeof showError === 'function') showError('上传失败: ' + (e.message || '网络错误'));
        }
        return;
    }

    const selected = document.querySelector('#profileAvatarOptions .avatar-option.selected');
    const payload = {
        user_id: getCurrentUserId(),
        nickname
    };
    if (selected && selected.dataset.avatar) {
        payload.avatar = selected.dataset.avatar;
    } else if (currentProfileAvatar !== 'custom'
        && PROFILE_AVATAR_PRESETS.some(item => item.value === currentProfileAvatar)) {
        payload.avatar = currentProfileAvatar;
    }

    try {
        const data = await apiRequest('/api/auth/profile', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
        if (!data || data.status !== 'success') {
            if (typeof showError === 'function') showError((data && data.message) || '保存失败');
            return;
        }
        closeProfileEditModal();
        if (typeof showSuccess === 'function') showSuccess('资料已更新');
        await loadProfilePage();
    } catch (e) {
        if (typeof showError === 'function') showError('保存失败: ' + (e.message || '网络错误'));
    }
}

async function loadProfilePage() {
    try {
        const data = await apiRequest(`/api/auth/profile?user_id=${getCurrentUserId()}`);
        if (!data || data.status !== 'success' || !data.data) return;
        const p = data.data;

        currentProfileAvatar = p.avatar || 'default';
        renderProfileAvatar(p);

        const nameEl = document.getElementById('profileName');
        if (nameEl) nameEl.textContent = p.nickname || '学习者';

        const lvlEl = document.getElementById('profileLevel');
        if (lvlEl) lvlEl.textContent = p.level || 1;

        const titleEl = document.getElementById('profileTitle');
        if (titleEl) {
            const lv = Number(p.level || 1);
            titleEl.textContent = lv >= 10 ? '学霸' : lv >= 5 ? '进阶' : lv >= 3 ? '熟练' : '新手';
        }

        const xp = Number(p.xp || 0);
        const bar = document.getElementById('profileXpBar');
        if (bar) bar.style.width = `${Math.min(100, xp % 100)}%`;
        const xpEl = document.getElementById('profileXp');
        if (xpEl) xpEl.textContent = xp;

        const streakEl = document.getElementById('streakDays');
        if (streakEl) streakEl.textContent = p.streak_days || 0;

        // 今天签过就把按钮置为「今日已签到」
        const btn = document.getElementById('checkInBtn');
        if (btn) {
            const last = p.last_login ? new Date(p.last_login) : null;
            const signedToday = last && !Number.isNaN(last.getTime())
                && last.toDateString() === new Date().toDateString();
            btn.disabled = !!signedToday;
            btn.innerHTML = signedToday
                ? '<i class="fas fa-check"></i> 今日已签到'
                : '<i class="fas fa-calendar-check"></i> 每日签到';
        }

        const achList = document.getElementById('achievementsList');
        const achievements = Array.isArray(p.achievements) ? p.achievements : [];
        if (achList) {
            if (!achievements.length) {
                achList.innerHTML = '<div class="empty-state">暂无成就</div>';
            } else {
                achList.innerHTML = achievements.slice(0, 8).map(a => {
                    const name = typeof a === 'string' ? a : (a.name || a.title || '成就');
                    return `<span class="achievement-badge"><i class="fas fa-medal"></i> ${name}</span>`;
                }).join('');
            }
        }
    } catch (e) {
        console.error('加载个人资料失败:', e);
    }
}

async function dailyCheckIn() {
    const btn = document.getElementById('checkInBtn');
    if (btn) btn.disabled = true;
    try {
        const data = await apiRequest('/api/checkin', {
            method: 'POST',
            body: JSON.stringify({ user_id: getCurrentUserId() })
        });
        if (data && data.status === 'success') {
            const days = (data.data && data.data.streak_days) || 0;
            const already = !!(data.data && data.data.already_checked_in);
            if (already) {
                if (typeof showError === 'function') showError('今天已经签到过了');
            } else {
                const xpGain = Number((data.data && data.data.xp_gain) || 0);
                const badges = (data.data && data.data.new_achievements) || [];
                let msg = `签到成功，已连续 ${days} 天`;
                if (xpGain > 0) msg += `，+${xpGain} XP`;
                if (badges.length) msg += `　解锁：${badges.join('、')}`;
                if (typeof showSuccess === 'function') showSuccess(msg);
            }
            await loadProfilePage();
        } else {
            if (typeof showError === 'function') showError((data && data.message) || '签到失败');
        }
    } catch (e) {
        if (typeof showError === 'function') showError('签到失败: ' + (e.message || '网络错误'));
    } finally {
        if (btn) btn.disabled = false;
    }
}

// ===== 删除闪卡 =====
async function deleteFlashcard(cardId) {
    if (!cardId) return;
    if (!confirm('确定删除这张闪卡吗？')) return;
    try {
        const data = await apiRequest(`/api/flashcards/${cardId}?user_id=${encodeURIComponent(getCurrentUserId())}`, {
            method: 'DELETE'
        });
        if (data && data.status === 'success') {
            if (typeof showSuccess === 'function') showSuccess('已删除');
            loadFlashcards();
        } else {
            if (typeof showError === 'function') showError((data && data.message) || '删除失败');
        }
    } catch (e) {
        if (typeof showError === 'function') showError('删除失败: ' + (e.message || '网络错误'));
    }
}

// ===== 学习进度：编辑与删除 =====
function editProgress(subject, completed, total, hours) {
    const s = document.getElementById('manualProgressSubject');
    const c = document.getElementById('manualProgressCompleted');
    const t = document.getElementById('manualProgressTotal');
    const h = document.getElementById('manualProgressHours');
    if (s) s.value = subject;
    if (c) c.value = completed;
    if (t) t.value = total;
    if (h) h.value = hours;
    if (s) { s.scrollIntoView({ behavior: 'smooth', block: 'center' }); s.focus(); }
    if (typeof showSuccess === 'function') showSuccess('已填入表单，修改后点「保存进度」');
}

async function deleteProgress(subject) {
    if (!subject) return;
    if (!confirm(`确定删除「${subject}」的学习进度吗？`)) return;
    try {
        const data = await apiRequest(`/api/progress/${encodeURIComponent(subject)}?user_id=${encodeURIComponent(getCurrentUserId())}`, {
            method: 'DELETE'
        });
        if (data && data.status === 'success') {
            if (typeof showSuccess === 'function') showSuccess('已删除');
            if (typeof loadProgress === 'function') loadProgress();
            if (typeof loadProgressOverview === 'function') loadProgressOverview();
        } else {
            if (typeof showError === 'function') showError((data && data.message) || '删除失败');
        }
    } catch (e) {
        if (typeof showError === 'function') showError('删除失败: ' + (e.message || '网络错误'));
    }
}

// ===== 今日一个概念 =====
let __concept = null;

async function loadDailyConcept(force) {
    const titleEl = document.getElementById('conceptTitle');
    const snippetEl = document.getElementById('conceptSnippet');
    if (!titleEl) return;
    if (!force && __concept) return;

    if (force) {
        titleEl.textContent = '正在挑选…';
        snippetEl.textContent = '从知识库里翻一页给你。';
    }

    try {
        const data = await apiRequest('/api/daily-concept');
        if (data && data.status === 'success' && data.data) {
            __concept = data.data;
            titleEl.textContent = data.data.title || '（未命名概念）';
            snippetEl.textContent = data.data.snippet || '';
        } else {
            titleEl.textContent = '知识库暂时取不到内容';
            snippetEl.textContent = (data && data.message) ? String(data.message).slice(0, 60) : '';
        }
    } catch (e) {
        titleEl.textContent = '知识库暂时取不到内容';
        snippetEl.textContent = '原因：' + (e && e.message ? String(e.message).slice(0, 80) : '未知');
    }
    if (typeof updateWelcomeTitle === 'function') updateWelcomeTitle();
}

// 标题：有待复习说任务，没待复习就说今天要搞懂的概念
function updateWelcomeTitle() {
    const el = document.getElementById('welcomeTitle');
    if (!el) return;
    const due = Number(window.__dueReviewCount || 0);
    if (due > 0) {
        el.textContent = '开始今天的学习';
        return;
    }
    if (__concept && __concept.title) {
        el.textContent = `今天，把「${__concept.title}」搞懂`;
        return;
    }
    el.textContent = '开始今天的学习';
}

function askAboutConcept() {
    if (!__concept || !__concept.title) return;
    switchPage('chat');
    setTimeout(() => {
        if (typeof prefillChatInput === 'function') {
            prefillChatInput(`讲讲「${__concept.title}」`);
        }
    }, 120);
}

// 按时间段问候
function updateGreeting() {
    const el = document.getElementById('welcomeGreet');
    if (!el) return;
    const h = new Date().getHours();
    let g = '晚上好';
    if (h >= 5 && h < 11) g = '早上好';
    else if (h >= 11 && h < 14) g = '中午好';
    else if (h >= 14 && h < 18) g = '下午好';
    el.textContent = `${g}，今天想学点什么？`;
}

// 仪表盘：概念卡整卡可点、统计卡整卡可点
function bindDashboardClickable() {
    const concept = document.getElementById('dailyConcept');
    if (concept && !concept.dataset.bound) {
        concept.dataset.bound = '1';
        concept.style.cursor = 'pointer';
        concept.addEventListener('click', (e) => {
            if (e.target.closest('.concept-refresh')) return;
            askAboutConcept();
        });
    }
    document.querySelectorAll('.bento-top .stat-item').forEach(card => {
        if (card.dataset.bound) return;
        card.dataset.bound = '1';
        const link = card.querySelector('.stat-link');
        if (link) link.style.display = 'none';
        card.style.cursor = 'pointer';
        card.addEventListener('click', () => { if (link) link.click(); });
    });
}

// 近 7 天柱状图
async function loadWeekChart() {
    const box = document.getElementById('weekChart');
    if (!box) return;
    try {
        const data = await apiRequest(`/api/activity-week?user_id=${encodeURIComponent(getCurrentUserId())}`);
        if (!data || data.status !== 'success' || !data.data) return;
        const days = data.data.days || [];
        const totalEl = document.getElementById('weekTotal');
        if (totalEl) totalEl.textContent = `${data.data.total || 0} 次`;
        const max = Math.max(1, ...days.map(d => d.count));
        box.innerHTML = days.map(d => {
            const h = Math.round((d.count / max) * 100);
            const cls = d.is_today ? 'week-bar today' : (d.count > 0 ? 'week-bar' : 'week-bar empty');
            return `<div class="week-col" title="${d.date}：${d.count} 次">
                <div class="${cls}" style="height:${Math.max(d.count > 0 ? 12 : 4, h)}%"></div>
                <span class="week-day">${d.label}</span>
            </div>`;
        }).join('');
    } catch (e) {
        console.warn('加载周视图失败', e);
    }
}

// 在流式回答下方追加来源面板（与 renderAnswerWithSources 同一套结构）
function appendSourcesPanel(msgDiv, metadata) {
    if (!msgDiv || !metadata) return;
    if (metadata.is_general) return;
    const snippets = Array.isArray(metadata.context_snippets) ? metadata.context_snippets : [];
    const filtered = snippets.filter(s => s && s.content).slice(0, 3);
    if (!filtered.length) return;
    if (msgDiv.querySelector('.sources-panel')) return;

    const wrap = document.createElement('div');
    wrap.className = 'sources-panel';
    wrap.style.marginTop = '12px';

    let inner = '<details class="sources-details">';
    inner += '<summary>查看参考资料来源（' + filtered.length + ' 段）</summary>';
    inner += '<div class="sources-body">';
    filtered.forEach((s, idx) => {
        const meta = s.metadata || {};
        const fileName = meta.file_name || meta.title || meta.url || '';
        const title = s.title || fileName || ('资料片段 ' + (idx + 1));
        inner += '<div class="source-item' + (idx === 0 ? ' first' : '') + '">';
        inner += '<div class="source-title">' + title + '</div>';
        if (fileName && fileName !== title) {
            inner += '<div class="source-file">来源：' + fileName + '</div>';
        }
        inner += '<div class="source-content">' + (s.content || '') + '</div>';
        inner += '</div>';
    });
    inner += '</div></details>';

    wrap.innerHTML = inner;
    msgDiv.appendChild(wrap);
}

// 极简 Markdown 兜底（marked 不可用时）：支持标题、无序列表、加粗、行内代码
function renderLiteMarkdown(text) {
    const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    const inline = s => esc(s)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    const lines = String(text || '').split('\n');
    let html = '';
    let inList = false;

    const closeList = () => { if (inList) { html += '</ul>'; inList = false; } };

    lines.forEach(raw => {
        const line = raw.trim();
        if (!line) { closeList(); return; }
        const h = line.match(/^(#{2,4})\s+(.*)$/);
        if (h) { closeList(); html += `<h4>${inline(h[2])}</h4>`; return; }
        const li = line.match(/^[-*]\s+(.*)$/);
        if (li) {
            if (!inList) { html += '<ul>'; inList = true; }
            html += `<li>${inline(li[1])}</li>`;
            return;
        }
        closeList();
        html += `<p>${inline(line)}</p>`;
    });
    closeList();
    return html;
}

