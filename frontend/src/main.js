import './style.css'

const API_BASE = '/api'

// State
let currentFile = null
let files = []
let extracting = false

// DOM Elements
const app = document.getElementById('app')

// Initialize
async function init() {
  app.innerHTML = `
    <div class="container">
      <aside class="sidebar">
        <h2>Files</h2>
        <div class="stats" id="stats"></div>
        <ul class="file-list" id="file-list"></ul>
      </aside>
      <main class="main">
        <header class="header">
          <h1>Vietnamese Medical NER</h1>
          <span class="subtitle">Agentic Pipeline</span>
          <div class="spacer"></div>
          <label class="toggle">
            <input type="checkbox" id="use-llm" />
            <span>RunPod LLM</span>
          </label>
        </header>
        <div class="content">
          <div class="panel panel-left">
            <div class="panel-header">Input & Results</div>
            <div class="panel-body">
              <textarea id="input-text" placeholder="Enter Vietnamese clinical text..."></textarea>
              <div class="controls">
                <button class="btn btn-primary" id="extract-btn" onclick="runExtract()">Extract</button>
                <span id="status"></span>
              </div>
              <div id="result-info" class="result-info hidden">
                <span>Mentions: <strong id="ri-count">0</strong></span>
                <span>Time: <strong id="ri-elapsed">0s</strong></span>
              </div>
              <div id="highlighted" class="highlighted-text hidden"></div>
              <div id="results"></div>
            </div>
          </div>
          <div class="panel panel-right">
            <div class="panel-header">
              Agent Logs
              <span id="log-count"></span>
              <div class="spacer"></div>
              <button class="btn btn-sm" onclick="toggleDetail()">Detail</button>
            </div>
            <div class="panel-body" id="logs">
              <div class="empty">Agent logs will appear here...</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  `

  await loadStats()
  await loadFiles()
}

// API calls
async function apiGet(path) {
  const resp = await fetch(API_BASE + path)
  return resp.json()
}

async function apiPost(path, data) {
  const resp = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  return resp
}

// Load stats
async function loadStats() {
  const stats = await apiGet('/stats')
  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="value">${stats.icd10_codes?.toLocaleString() || '-'}</div><div class="label">ICD-10</div></div>
    <div class="stat"><div class="value">${stats.rxnorm_concepts?.toLocaleString() || '-'}</div><div class="label">RxNorm</div></div>
    <div class="stat"><div class="value">${stats.input_files || '-'}</div><div class="label">Files</div></div>
    <div class="stat"><div class="value">${stats.output_files || '-'}</div><div class="label">Outputs</div></div>
  `
}

// Load files
async function loadFiles() {
  files = await apiGet('/files')
  const list = document.getElementById('file-list')
  list.innerHTML = files.map(f => `
    <li onclick="selectFile('${f.id}')" data-id="${f.id}">
      <span>${f.name}</span>
      <span class="file-status ${f.cached ? 'done' : 'pending'}">${f.cached ? '✓' + f.count : '●'}</span>
    </li>
  `).join('')
}

// Select file
async function selectFile(id) {
  const data = await apiGet('/file/' + id)
  document.getElementById('input-text').value = data.text
  currentFile = id

  // Highlight selected file
  document.querySelectorAll('.file-list li').forEach(li => li.classList.remove('active'))
  document.querySelector(`.file-list li[data-id="${id}"]`)?.classList.add('active')

  // Show cached result
  if (data.result) {
    showResult(data.result.mentions, data.result.elapsed)
  } else {
    clearResults()
  }
}

// Run extraction
async function runExtract() {
  const text = document.getElementById('input-text').value.trim()
  if (!text || extracting) return

  extracting = true
  const btn = document.getElementById('extract-btn')
  const status = document.getElementById('status')
  btn.disabled = true
  btn.innerHTML = '<span class="spinner"></span> Processing...'
  status.textContent = 'Running...'

  clearResults()
  clearLogs()

  const useLlm = document.getElementById('use-llm').checked
  const startTime = Date.now()

  try {
    const resp = await apiPost('/extract', { text, use_llm: useLlm, file_id: currentFile })
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const msg = JSON.parse(line)
          if (msg.type === 'log') {
            addLog(msg.data)
          } else if (msg.type === 'result') {
            const elapsed = msg.data.elapsed || ((Date.now() - startTime) / 1000).toFixed(1)
            showResult(msg.data.mentions, elapsed)
            status.textContent = `${msg.data.count} mentions (${elapsed}s)`
            loadFiles()
          } else if (msg.type === 'error') {
            addLog({ agent: 'error', action: msg.data.message, ts: Date.now() / 1000 })
            status.textContent = 'Error'
          }
        } catch (e) {}
      }
    }
  } catch (e) {
    status.textContent = 'Error: ' + e.message
  } finally {
    btn.disabled = false
    btn.textContent = 'Extract'
    extracting = false
  }
}

// Show result
function showResult(mentions, elapsed) {
  const resultsDiv = document.getElementById('results')
  const highlightedDiv = document.getElementById('highlighted')
  const resultInfo = document.getElementById('result-info')

  if (!mentions || mentions.length === 0) {
    resultsDiv.innerHTML = '<div class="empty">No mentions found</div>'
    return
  }

  // Update result info
  resultInfo.classList.remove('hidden')
  document.getElementById('ri-count').textContent = mentions.length
  document.getElementById('ri-elapsed').textContent = elapsed + 's'

  // Highlight text
  const text = document.getElementById('input-text').value
  highlightText(text, mentions)

  // Render mention cards
  resultsDiv.innerHTML = mentions.map(m => `
    <div class="mention-card">
      <div class="text">${escapeHtml(m.text)}</div>
      <span class="type type-${m.type}">${m.type}</span>
      <span class="position">[${m.position[0]}, ${m.position[1]}]</span>
      <div class="assertions">
        <span class="assertion-tag ${m.assertions?.isNegated ? 'active' : ''}">neg</span>
        <span class="assertion-tag ${m.assertions?.isFamily ? 'active' : ''}">fam</span>
        <span class="assertion-tag ${m.assertions?.isHistorical ? 'active' : ''}">hist</span>
      </div>
      ${m.candidates?.length ? `
        <div class="candidates">
          ${m.candidates.map(c => `<span class="candidate">${c}</span>`).join('')}
        </div>
      ` : ''}
    </div>
  `).join('')
}

// Highlight text with colored citations
function highlightText(text, mentions) {
  const highlightedDiv = document.getElementById('highlighted')
  if (!mentions?.length) {
    highlightedDiv.classList.add('hidden')
    return
  }

  const sorted = [...mentions].sort((a, b) => b.position[0] - a.position[0])
  let html = escapeHtml(text)

  for (const m of sorted) {
    const start = m.position[0]
    const end = m.position[1]
    const mentionText = escapeHtml(text.substring(start, end))
    const cssClass = `cite-${m.type}`
    const before = html.substring(0, start)
    const after = html.substring(end)
    html = `${before}<span class="${cssClass}" title="${m.type}: ${mentionText}">${mentionText}</span>${after}`
  }

  highlightedDiv.innerHTML = html
  highlightedDiv.classList.remove('hidden')
}

// Clear results
function clearResults() {
  document.getElementById('results').innerHTML = ''
  document.getElementById('highlighted').classList.add('hidden')
  document.getElementById('result-info').classList.add('hidden')
}

// Add log entry
function addLog(entry) {
  const logsDiv = document.getElementById('logs')
  if (logsDiv.querySelector('.empty')) logsDiv.innerHTML = ''

  const agent = entry.agent || 'system'
  const action = entry.action || ''
  const input = entry.input || null
  const output = entry.output || null
  const message = entry.message || entry.error || ''

  let detailHtml = ''
  if (input || output) {
    detailHtml = '<div class="log-detail">'
    if (input) {
      const inputStr = typeof input === 'object' ? JSON.stringify(input, null, 2) : String(input)
      detailHtml += `<div class="log-io"><div class="log-label">Input</div><div class="log-value">${escapeHtml(inputStr)}</div></div>`
    }
    if (output) {
      const outputStr = typeof output === 'object' ? JSON.stringify(output, null, 2) : String(output)
      detailHtml += `<div class="log-io"><div class="log-label">Output</div><div class="log-value output">${escapeHtml(outputStr)}</div></div>`
    }
    detailHtml += '</div>'
  }

  const div = document.createElement('div')
  div.className = `log-entry ${agent}`
  div.innerHTML = `
    <div class="log-header">
      <span>
        <span class="agent">[${agent}]</span>
        <span class="action">${action}</span>
        ${message ? `<span class="error">${message}</span>` : ''}
      </span>
      <span class="ts">${new Date(entry.ts * 1000).toLocaleTimeString()}</span>
    </div>
    ${detailHtml}
  `

  logsDiv.appendChild(div)
  logsDiv.scrollTop = logsDiv.scrollHeight

  document.getElementById('log-count').textContent = `${logsDiv.children.length} logs`
}

// Clear logs
function clearLogs() {
  document.getElementById('logs').innerHTML = '<div class="empty">Agent logs will appear here...</div>'
  document.getElementById('log-count').textContent = ''
}

// Toggle detail visibility
function toggleDetail() {
  document.querySelectorAll('.log-detail').forEach(el => {
    el.style.display = el.style.display === 'none' ? 'block' : 'none'
  })
}

// Escape HTML
function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

// Initialize
init()
