/**
 * INDIA CRICKET PERFORMANCE ANALYTICS - JAVASCRIPT ENGINE
 * Handles interactive prediction, Chart.js visualizations, comparison,
 * live squad search, and localStorage prediction history.
 */

document.addEventListener('DOMContentLoaded', () => {
  initMobileNav();
  initPredictionModule();
  initComparisonModule();
  initDashboardCharts();
  initPlayerSearch();
});

/* ==========================================================================
   1. MOBILE NAVIGATION TOGGLE
   ========================================================================== */
function initMobileNav() {
  const toggleBtn = document.querySelector('.nav-toggle');
  const navMenu = document.querySelector('.nav-menu');

  if (toggleBtn && navMenu) {
    toggleBtn.addEventListener('click', () => {
      navMenu.classList.toggle('open');
      const isOpen = navMenu.classList.contains('open');
      toggleBtn.setAttribute('aria-expanded', isOpen);
    });
  }
}

/* ==========================================================================
   2. PREDICTION MODULE & FORM LOGIC
   ========================================================================== */
function initPredictionModule() {
  const form = document.getElementById('predictionForm');
  if (!form) return;

  const playerSelect = document.getElementById('selectExistingPlayer');
  const formatSelect = document.getElementById('matchFormat');
  const clearFormBtn = document.getElementById('clearFormBtn');

  // Load existing players list for autofill
  let cachedPlayers = [];
  fetch('/api/players')
    .then(res => res.json())
    .then(data => {
      if (data.success && data.players) {
        cachedPlayers = data.players;
        populatePlayerDropdown(cachedPlayers);
        checkUrlPrefill(cachedPlayers);
      }
    })
    .catch(err => console.error('Error fetching player database:', err));

  // When player or format changes, auto-fill form with verified stats
  if (playerSelect) {
    playerSelect.addEventListener('change', () => autofillPlayerStats(cachedPlayers));
  }
  if (formatSelect) {
    formatSelect.addEventListener('change', () => {
      if (playerSelect && playerSelect.value) {
        autofillPlayerStats(cachedPlayers);
      }
    });
  }

  // Clear Form Button
  if (clearFormBtn) {
    clearFormBtn.addEventListener('click', () => {
      form.reset();
      if (playerSelect) playerSelect.value = '';
      hideResults();
      showToast('Form cleared successfully.');
    });
  }

  // Form Submit Handler
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await handlePredictionSubmit(form);
  });

  // Initialize History Table from localStorage
  renderPredictionHistory();
  initHistoryControls();
}

function populatePlayerDropdown(players) {
  const select = document.getElementById('selectExistingPlayer');
  if (!select) return;

  const currentVal = select.value;
  select.innerHTML = '<option value="">-- Choose Indian Cricketer (Optional Autofill) --</option>';

  players.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = `${p.name} (${p.role})`;
    select.appendChild(opt);
  });

  if (currentVal) select.value = currentVal;
}

function checkUrlPrefill(players) {
  const urlParams = new URLSearchParams(window.location.search);
  const playerId = urlParams.get('player');
  const format = urlParams.get('format');

  if (format) {
    const formatSelect = document.getElementById('matchFormat');
    if (formatSelect) formatSelect.value = format;
  }

  if (playerId) {
    const playerSelect = document.getElementById('selectExistingPlayer');
    if (playerSelect) {
      playerSelect.value = playerId;
      autofillPlayerStats(players);
    }
  }
}

function autofillPlayerStats(players) {
  const playerSelect = document.getElementById('selectExistingPlayer');
  const formatSelect = document.getElementById('matchFormat');
  if (!playerSelect || !formatSelect) return;

  const playerId = playerSelect.value;
  const format = formatSelect.value;
  if (!playerId) return;

  const player = players.find(p => p.id === playerId);
  if (!player || !player.stats || !player.stats[format]) return;

  const st = player.stats[format];

  document.getElementById('playerName').value = player.name;
  document.getElementById('playerRole').value = player.role;
  document.getElementById('matchesPlayed').value = st.matches || 0;
  document.getElementById('totalRuns').value = st.runs || 0;
  document.getElementById('battingAvg').value = st.batting_avg || 0.0;
  document.getElementById('strikeRate').value = st.strike_rate || 0.0;
  document.getElementById('totalFours').value = st.fours || 0;
  document.getElementById('totalSixes').value = st.sixes || 0;
  document.getElementById('wicketsTaken').value = st.wickets || 0;
  document.getElementById('bowlingAvg').value = st.bowling_avg || 0.0;
  document.getElementById('bowlingEcon').value = st.bowling_econ || 0.0;

  showToast(`Autofilled verified ${format} statistics for ${player.name}`);
}

async function handlePredictionSubmit(form) {
  const submitBtn = form.querySelector('button[type="submit"]');
  const originalBtnText = submitBtn.innerHTML;

  // Validation
  const pName = document.getElementById('playerName').value.trim();
  const matches = parseFloat(document.getElementById('matchesPlayed').value);

  if (!pName) {
    showToast('Please enter or select a player name.', 'error');
    return;
  }
  if (isNaN(matches) || matches < 1) {
    showToast('Matches played must be at least 1.', 'error');
    return;
  }

  const payload = {
    player_name: pName,
    role: document.getElementById('playerRole').value,
    format: document.getElementById('matchFormat').value,
    matches: matches,
    runs: parseFloat(document.getElementById('totalRuns').value) || 0,
    batting_avg: parseFloat(document.getElementById('battingAvg').value) || 0.0,
    strike_rate: parseFloat(document.getElementById('strikeRate').value) || 0.0,
    fours: parseFloat(document.getElementById('totalFours').value) || 0,
    sixes: parseFloat(document.getElementById('totalSixes').value) || 0,
    wickets: parseFloat(document.getElementById('wicketsTaken').value) || 0,
    bowling_avg: parseFloat(document.getElementById('bowlingAvg').value) || 0.0,
    bowling_econ: parseFloat(document.getElementById('bowlingEcon').value) || 0.0
  };

  try {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '⚡ Running ML Model...';

    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      throw new Error(result.error || 'Prediction calculation failed');
    }

    displayPredictionResult(result);
    saveToLocalStorage(result);
    renderPredictionHistory();

    // Smooth scroll down to result
    const resultBox = document.getElementById('predictionResultBox');
    if (resultBox) {
      resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    showToast(`Prediction generated for ${result.player_name}: Score ${result.predicted_score}/100`);

  } catch (error) {
    console.error('Prediction error:', error);
    showToast(`Error: ${error.message}`, 'error');
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalBtnText;
  }
}

function displayPredictionResult(res) {
  const resultBox = document.getElementById('predictionResultBox');
  if (!resultBox) return;

  resultBox.style.display = 'block';

  // Update text
  document.getElementById('resPlayerName').textContent = res.player_name;
  document.getElementById('resPlayerMeta').textContent = `${res.role} • ${res.format} Cricket`;

  // Animate Number Counter
  const scoreEl = document.getElementById('resScoreNum');
  animateCounter(scoreEl, 0, res.predicted_score, 800);

  // Animate Circular Gauge
  const circleProgress = document.getElementById('scoreCircleProgress');
  if (circleProgress) {
    const circumference = 377; // 2 * PI * 60
    const offset = circumference - (res.predicted_score / 100) * circumference;
    circleProgress.style.strokeDashoffset = offset;
    circleProgress.style.stroke = res.color || '#00C853';
  }

  // Category Badge
  const catBadge = document.getElementById('resCategoryBadge');
  catBadge.className = `badge-category ${res.badge_class}`;
  catBadge.textContent = `${res.category} (${res.category_label})`;

  document.getElementById('resCategoryDesc').textContent = res.category_description;

  // Analysis Cards
  document.getElementById('resBattingAnalysis').textContent = res.batting_analysis;
  document.getElementById('resBowlingAnalysis').textContent = res.bowling_analysis;

  // Suggestions List
  const sugList = document.getElementById('resSuggestionsList');
  sugList.innerHTML = '';
  res.suggestions.forEach(sug => {
    const li = document.createElement('li');
    li.textContent = sug;
    sugList.appendChild(li);
  });
}

function hideResults() {
  const resultBox = document.getElementById('predictionResultBox');
  if (resultBox) resultBox.style.display = 'none';
}

function animateCounter(element, start, end, duration) {
  let startTime = null;
  const step = (timestamp) => {
    if (!startTime) startTime = timestamp;
    const progress = Math.min((timestamp - startTime) / duration, 1);
    const current = Math.floor(progress * (end - start) + start);
    element.textContent = current;
    if (progress < 1) {
      window.requestAnimationFrame(step);
    } else {
      element.textContent = end;
    }
  };
  window.requestAnimationFrame(step);
}

/* ==========================================================================
   3. PREDICTION HISTORY (LOCALSTORAGE)
   ========================================================================== */
const STORAGE_KEY = 'india_cricket_prediction_history';

function getHistory() {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch (e) {
    return [];
  }
}

function saveToLocalStorage(res) {
  const history = getHistory();
  const entry = {
    id: Date.now(),
    date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
    name: res.player_name,
    format: res.format,
    role: res.role,
    score: res.predicted_score,
    category: res.category,
    badgeClass: res.badge_class
  };
  // Store up to 20 entries
  history.unshift(entry);
  if (history.length > 20) history.pop();

  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

function renderPredictionHistory() {
  const tableBody = document.getElementById('historyTableBody');
  const emptyMsg = document.getElementById('historyEmptyMsg');
  if (!tableBody) return;

  const history = getHistory();

  if (history.length === 0) {
    tableBody.innerHTML = '';
    if (emptyMsg) emptyMsg.style.display = 'block';
    return;
  }

  if (emptyMsg) emptyMsg.style.display = 'none';

  tableBody.innerHTML = history.map(item => `
    <tr>
      <td>${item.date}</td>
      <td><strong>${item.name}</strong></td>
      <td><span class="player-role-badge">${item.format}</span></td>
      <td>${item.role}</td>
      <td><span style="font-weight: 800; font-size: 1.1rem; color: #FFFFFF;">${item.score}</span>/100</td>
      <td><span class="badge-category ${item.badgeClass}" style="padding: 0.15rem 0.6rem; font-size: 0.72rem;">${item.category}</span></td>
      <td>
        <button class="btn-icon" onclick="deleteHistoryItem(${item.id})" title="Delete Record">✕</button>
      </td>
    </tr>
  `).join('');
}

function initHistoryControls() {
  const clearBtn = document.getElementById('clearHistoryBtn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear your prediction history?')) {
        localStorage.removeItem(STORAGE_KEY);
        renderPredictionHistory();
        showToast('Prediction history cleared.');
      }
    });
  }
}

window.deleteHistoryItem = function(id) {
  let history = getHistory();
  history = history.filter(item => item.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  renderPredictionHistory();
  showToast('Entry removed from history.');
};

/* ==========================================================================
   4. PLAYER COMPARISON MODULE
   ========================================================================== */
let comparisonRadarChart = null;
let comparisonBarChart = null;

function initComparisonModule() {
  const compForm = document.getElementById('comparisonForm');
  if (!compForm) return;

  const p1Select = document.getElementById('comparePlayer1');
  const p2Select = document.getElementById('comparePlayer2');
  const formatSelect = document.getElementById('compareFormat');

  compForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const p1 = p1Select.value;
    const p2 = p2Select.value;
    const fmt = formatSelect.value;

    if (p1 === p2) {
      showToast('Please select two different Indian players to compare.', 'error');
      return;
    }

    await runComparison(p1, p2, fmt);
  });

  // Run on initial page load with defaults
  if (p1Select && p2Select && formatSelect) {
    runComparison(p1Select.value, p2Select.value, formatSelect.value);
  }
}

async function runComparison(p1Id, p2Id, fmt) {
  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ p1_id: p1Id, p2_id: p2Id, format: fmt })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    renderComparisonUI(data.player1, data.player2, data.format);
  } catch (err) {
    console.error('Comparison error:', err);
    showToast(`Comparison error: ${err.message}`, 'error');
  }
}

function renderComparisonUI(p1, p2, fmt) {
  // Update Header Cards
  document.getElementById('p1Name').textContent = p1.name;
  document.getElementById('p1Role').textContent = `${p1.role} (${fmt})`;
  document.getElementById('p1Score').textContent = `${p1.score}/100`;

  document.getElementById('p2Name').textContent = p2.name;
  document.getElementById('p2Role').textContent = `${p2.role} (${fmt})`;
  document.getElementById('p2Score').textContent = `${p2.score}/100`;

  // Compare Stats & Highlight Winners
  const s1 = p1.stats || {};
  const s2 = p2.stats || {};

  setStatRow('Runs', s1.runs || 0, s2.runs || 0, 'higher');
  setStatRow('BatAvg', s1.batting_avg || 0.0, s2.batting_avg || 0.0, 'higher');
  setStatRow('StrikeRate', s1.strike_rate || 0.0, s2.strike_rate || 0.0, 'higher');
  setStatRow('Centuries', s1.centuries || 0, s2.centuries || 0, 'higher');
  setStatRow('Wickets', s1.wickets || 0, s2.wickets || 0, 'higher');
  setStatRow('BowlAvg', s1.bowling_avg || 0.0, s2.bowling_avg || 0.0, 'lower_pos');
  setStatRow('Econ', s1.bowling_econ || 0.0, s2.bowling_econ || 0.0, 'lower_pos');
  setStatRow('MLScore', p1.score, p2.score, 'higher');

  // Render Charts
  renderComparisonCharts(p1, p2, fmt);
}

function setStatRow(metricId, val1, val2, rule) {
  const el1 = document.getElementById(`p1_${metricId}`);
  const el2 = document.getElementById(`p2_${metricId}`);
  if (!el1 || !el2) return;

  el1.textContent = val1;
  el2.textContent = val2;
  el1.className = '';
  el2.className = '';

  if (val1 === val2) return;

  if (rule === 'higher') {
    if (val1 > val2) el1.className = 'stat-winner';
    else el2.className = 'stat-winner';
  } else if (rule === 'lower_pos') {
    // For bowling average and economy, lower is better (if > 0)
    if (val1 > 0 && (val2 === 0 || val1 < val2)) el1.className = 'stat-winner';
    else if (val2 > 0 && (val1 === 0 || val2 < val1)) el2.className = 'stat-winner';
  }
}

function renderComparisonCharts(p1, p2, fmt) {
  if (typeof Chart === 'undefined') return;

  const s1 = p1.stats;
  const s2 = p2.stats;

  // Normalized Radar Metrics (0 to 100 scale for visual balance)
  const norm = (val, max) => Math.min(100, Math.round((val / max) * 100));

  const radarData1 = [
    norm(s1.batting_avg || 0, 60),
    norm(s1.strike_rate || 0, fmt === 'T20I' ? 170 : 110),
    norm(s1.runs || 0, 10000),
    norm(s1.wickets || 0, 250),
    s1.bowling_econ > 0 ? Math.max(0, 100 - norm(s1.bowling_econ, 10)) : 20,
    p1.score
  ];

  const radarData2 = [
    norm(s2.batting_avg || 0, 60),
    norm(s2.strike_rate || 0, fmt === 'T20I' ? 170 : 110),
    norm(s2.runs || 0, 10000),
    norm(s2.wickets || 0, 250),
    s2.bowling_econ > 0 ? Math.max(0, 100 - norm(s2.bowling_econ, 10)) : 20,
    p2.score
  ];

  const radarCanvas = document.getElementById('comparisonRadarCanvas');
  if (radarCanvas) {
    if (comparisonRadarChart) comparisonRadarChart.destroy();
    comparisonRadarChart = new Chart(radarCanvas, {
      type: 'radar',
      data: {
        labels: ['Batting Avg', 'Strike Rate', 'Runs Index', 'Wickets Index', 'Econ Control', 'AI Score'],
        datasets: [
          {
            label: p1.name,
            data: radarData1,
            backgroundColor: 'rgba(0, 180, 216, 0.25)',
            borderColor: '#00B4D8',
            borderWidth: 2,
            pointBackgroundColor: '#00B4D8'
          },
          {
            label: p2.name,
            data: radarData2,
            backgroundColor: 'rgba(255, 119, 0, 0.25)',
            borderColor: '#FF7700',
            borderWidth: 2,
            pointBackgroundColor: '#FF7700'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            pointLabels: { color: '#CBD5E1', font: { size: 12, weight: 600 } },
            ticks: { display: false, max: 100, min: 0 }
          }
        },
        plugins: {
          legend: { labels: { color: '#FFFFFF', font: { size: 13, weight: 600 } } }
        }
      }
    });
  }

  // Grouped Bar Chart
  const barCanvas = document.getElementById('comparisonBarCanvas');
  if (barCanvas) {
    if (comparisonBarChart) comparisonBarChart.destroy();
    comparisonBarChart = new Chart(barCanvas, {
      type: 'bar',
      data: {
        labels: ['Batting Avg', 'Strike Rate', 'Predicted Score'],
        datasets: [
          {
            label: p1.name,
            data: [s1.batting_avg || 0, s1.strike_rate || 0, p1.score],
            backgroundColor: '#00B4D8'
          },
          {
            label: p2.name,
            data: [s2.batting_avg || 0, s2.strike_rate || 0, p2.score],
            backgroundColor: '#FF7700'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#FFFFFF' } }
        },
        scales: {
          x: { ticks: { color: '#94A3B8' }, grid: { display: false } },
          y: { ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255, 255, 255, 0.08)' } }
        }
      }
    });
  }
}

/* ==========================================================================
   5. TEAM DASHBOARD CHARTS (CHART.JS)
   ========================================================================== */
function initDashboardCharts() {
  if (typeof Chart === 'undefined') return;

  const runsCanvas = document.getElementById('topRunsChart');
  const wktsCanvas = document.getElementById('topWicketsChart');
  const roleCanvas = document.getElementById('roleDistChart');

  if (!runsCanvas && !wktsCanvas && !roleCanvas) return;

  fetch('/api/players')
    .then(r => r.json())
    .then(data => {
      if (!data.success) return;
      const players = data.players;
      renderRunsLeaderboard(players, runsCanvas);
      renderWicketsLeaderboard(players, wktsCanvas);
      renderRoleDistribution(players, roleCanvas);
    });
}

function renderRunsLeaderboard(players, canvas) {
  if (!canvas) return;

  // Aggregate total international runs across all formats
  const list = players.map(p => {
    const totalRuns = (p.stats.Test.runs || 0) + (p.stats.ODI.runs || 0) + (p.stats.T20I.runs || 0);
    return { name: p.name, runs: totalRuns };
  }).sort((a, b) => b.runs - a.runs).slice(0, 6);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: list.map(item => item.name),
      datasets: [{
        label: 'Total International Runs',
        data: list.map(item => item.runs),
        backgroundColor: [
          '#FF9933', '#00B4D8', '#0077B6', '#1E3A8A', '#0D9488', '#3B82F6'
        ],
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255, 255, 255, 0.08)' } },
        y: { ticks: { color: '#FFFFFF', font: { weight: 600 } }, grid: { display: false } }
      }
    }
  });
}

function renderWicketsLeaderboard(players, canvas) {
  if (!canvas) return;

  const list = players.map(p => {
    const totalWkts = (p.stats.Test.wickets || 0) + (p.stats.ODI.wickets || 0) + (p.stats.T20I.wickets || 0);
    return { name: p.name, wickets: totalWkts };
  }).filter(item => item.wickets > 0).sort((a, b) => b.wickets - a.wickets).slice(0, 6);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: list.map(item => item.name),
      datasets: [{
        label: 'Total International Wickets',
        data: list.map(item => item.wickets),
        backgroundColor: '#00B4D8',
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255, 255, 255, 0.08)' } },
        y: { ticks: { color: '#FFFFFF', font: { weight: 600 } }, grid: { display: false } }
      }
    }
  });
}

function renderRoleDistribution(players, canvas) {
  if (!canvas) return;

  const roles = {};
  players.forEach(p => {
    roles[p.category] = (roles[p.category] || 0) + 1;
  });

  new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: Object.keys(roles),
      datasets: [{
        data: Object.values(roles),
        backgroundColor: ['#00B4D8', '#FF9933', '#10B981', '#8B5CF6'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#CBD5E1', padding: 15, font: { size: 12 } }
        }
      },
      cutout: '65%'
    }
  });
}

/* ==========================================================================
   6. PLAYER SQUAD LIVE SEARCH & FILTER
   ========================================================================== */
function initPlayerSearch() {
  const searchInput = document.getElementById('playerDirectorySearch');
  const cards = document.querySelectorAll('.player-card[data-player-name]');

  if (searchInput && cards.length > 0) {
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      cards.forEach(card => {
        const name = card.getAttribute('data-player-name').toLowerCase();
        const role = card.getAttribute('data-player-role').toLowerCase();
        if (name.includes(q) || role.includes(q)) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  }
}

/* ==========================================================================
   7. TOAST NOTIFICATIONS
   ========================================================================== */
function showToast(message, type = 'info') {
  let toast = document.getElementById('appToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'appToast';
    toast.style.position = 'fixed';
    toast.style.bottom = '24px';
    toast.style.right = '24px';
    toast.style.zIndex = '9999';
    toast.style.padding = '12px 20px';
    toast.style.borderRadius = '10px';
    toast.style.color = '#FFFFFF';
    toast.style.fontSize = '0.9rem';
    toast.style.fontWeight = '600';
    toast.style.boxShadow = '0 10px 25px rgba(0,0,0,0.4)';
    toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    document.body.appendChild(toast);
  }

  if (type === 'error') {
    toast.style.backgroundColor = '#FF3D00';
  } else {
    toast.style.backgroundColor = '#0077B6';
  }

  toast.textContent = message;
  toast.style.opacity = '1';
  toast.style.transform = 'translateY(0)';

  clearTimeout(window.toastTimer);
  window.toastTimer = setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
  }, 3200);
}

