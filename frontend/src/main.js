import './style.css'

const API_BASE = '/api'
let currentFile = null
let currentFolder = 'outputs'
let extracting = false

const app = document.getElementById('app')

async function init() {
  app.innerHTML = `
    <div class="container">
      <aside class="sidebar">
        <h2>Files</h2>
        <div class="folder-bar">
          <select id="folder-select" class="folder-select">
            <option value="outputs">outputs</option>
            <option value="outputs_new">outputs_new</option>
          </select>
          <button class="btn btn-sm" onclick="refreshFiles()">⟳</button>
          <button class="btn btn-sm btn-primary" id="run-all-btn">▶ Run All</button>
        </div>
        <div id="run-progress" class="run-progress hidden">
          <div class="run-progress-bar" id="run-progress-bar"></div>
          <div class="run-progress-text" id="run-progress-text"></div>
        </div>
        <div class="stats" id="stats"></div>
        <div class="stats-summary" id="stats-summary"></div>
        <ul class="file-list" id="file-list"></ul>
      </aside>
      <main class="main">
        <header class="header">
          <h1>Vietnamese Medical NER</h1>
          <span class="subtitle">Agentic Pipeline</span>
          <div class="spacer"></div>
          <select id="model-select" class="folder-select" style="width:auto">
            <option value="">No LLM</option>
            <option value="runpod" selected>RunPod</option>
            <option value="local">Local (models/)</option>
          </select>
        </header>
        <div class="content">
          <div class="panel panel-left">
            <div class="panel-header">Input & Results</div>
            <div class="panel-body">
              <textarea id="input-text" placeholder="Enter Vietnamese clinical text..."></textarea>
              <div class="controls">
                <button class="btn btn-primary" id="extract-btn">Extract</button>
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
              <button class="btn btn-sm" id="detail-btn">Detail</button>
              <button class="btn btn-sm" id="list-btn">List</button>
            </div>
            <div class="panel-body" id="logs">
              <div class="empty">Agent logs will appear here...</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  `

  document.getElementById('folder-select').addEventListener('change', async (e) => {
    currentFolder = e.target.value
    await loadFiles()
  })

  document.getElementById('extract-btn').addEventListener('click', runExtract)
  document.getElementById('detail-btn').addEventListener('click', toggleDetail)
  document.getElementById('list-btn').addEventListener('click', toggleList)
  document.addEventListener('click', async (e) => {
    const li = e.target.closest('.file-list li')
    if (li) {
      const id = li.dataset.id
      if (id) await selectFile(id)
    }
  })

  await loadStats()
  await loadFiles()
}

function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

async function apiGet(path) {
  const resp = await fetch(API_BASE + path)
  return resp.json()
}

async function apiPost(path, data) {
  return fetch(API_BASE + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
}

async function loadStats() {
  const stats = await apiGet('/stats')
  const folders = stats.folders || {}
  let html = `<div class="stat"><div class="value">${stats.icd10_codes?.toLocaleString() || '-'}</div><div class="label">ICD-10</div></div>
    <div class="stat"><div class="value">${stats.rxnorm_concepts?.toLocaleString() || '-'}</div><div class="label">RxNorm</div></div>
    <div class="stat"><div class="value">${stats.input_files || '-'}</div><div class="label">Files</div></div>`
  document.getElementById('stats').innerHTML = html

  // Folder summary
  let summaryHtml = ''
  for (const [name, info] of Object.entries(folders)) {
    const active = name === currentFolder ? 'active' : ''
    summaryHtml += `<div class="folder-stat ${active}" onclick="switchFolder('${name}')">
      <span class="folder-name">${name}</span>
      <span class="folder-count">${info.count || 0} files, ${(info.mentions || 0).toLocaleString()} mentions</span>
    </div>`
  }
  document.getElementById('stats-summary').innerHTML = summaryHtml
  document.getElementById('folder-select').value = currentFolder
}

async function switchFolder(name) {
  currentFolder = name
  document.getElementById('folder-select').value = name
  document.getElementById('stats-summary').querySelectorAll('.folder-stat').forEach(el => el.classList.remove('active'))
  await loadFiles()
  await loadStats()
}

async function loadFiles() {
  const resp = await fetch(`${API_BASE}/files?folder=${currentFolder}`)
  const data = await resp.json()
  const files = data.files || []
  document.getElementById('stats-summary').querySelectorAll('.folder-stat').forEach(el => {
    if (el.querySelector('.folder-name')?.textContent === currentFolder) {
      el.querySelector('.folder-count').textContent = `${files.length} files, ${data.total?.toLocaleString() || 0} mentions`
    }
  })
  const list = document.getElementById('file-list')
  list.innerHTML = files.map(f => `
    <li data-id="${f.id}">
      <span>${f.name}</span>
      <span class="file-status ${f.has_output ? 'done' : 'pending'}">${f.has_output ? '✓' + f.count : '●'}</span>
    </li>
  `).join('')
}

window.refreshFiles = async function() {
  await loadFiles()
}

async function selectFile(id) {
  const resp = await fetch(`${API_BASE}/file/${id}?folder=${currentFolder}`)
  const data = await resp.json()
  document.getElementById('input-text').value = data.text
  currentFile = id

  document.querySelectorAll('.file-list li').forEach(li => li.classList.remove('active'))
  document.querySelector(`.file-list li[data-id="${id}"]`)?.classList.add('active')

  if (data.result) {
    showResult(data.result.mentions || [], data.result.elapsed)
  } else {
    clearResults()
  }

  // Load cached agent logs
  loadCachedLogs(id)
}

async function loadCachedLogs(fileId) {
  clearLogs()
  try {
    const resp = await fetch(`${API_BASE}/logs/${fileId}`)
    const data = await resp.json()
    if (data.exists && data.logs?.length > 0) {
      // Show system message
      addLog({ agent: 'system', action: `Loaded ${data.logs.length} cached logs`, ts: Date.now() / 1000 })
      for (const entry of data.logs) {
        addLog(entry)
      }
    }
  } catch (e) {
    // No cached logs - fine
  }
}

async function runExtract() {
  const text = document.getElementById('input-text').value.trim()
  if (!text || extracting) return

  extracting = true
  const btn = document.getElementById('extract-btn')
  const statusEl = document.getElementById('status')
  btn.disabled = true
  btn.innerHTML = '<span class="spinner"></span> Processing...'
  statusEl.textContent = 'Running...'

  clearResults()
  clearLogs()
  const modelType = document.getElementById('model-select').value
  const useLlm = modelType !== ''
  const startTime = Date.now()

  try {
    const resp = await apiPost('/extract', { text, use_llm: useLlm, model: modelType, file_id: currentFile, folder: currentFolder })
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
          if (msg.type === 'log') addLog(msg.data)
          else if (msg.type === 'result') {
            const elapsed = msg.data.elapsed || ((Date.now() - startTime) / 1000).toFixed(1)
            showResult(msg.data.mentions, elapsed)
            statusEl.textContent = `${msg.data.count} mentions (${elapsed}s)`
            loadFiles()
            loadStats()
          } else if (msg.type === 'error') {
            addLog({ agent: 'error', action: msg.data.message, ts: Date.now() / 1000 })
            statusEl.textContent = 'Error'
          }
        } catch (e) {}
      }
    }
  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message
  } finally {
    btn.disabled = false
    btn.textContent = 'Extract'
    extracting = false
  }
}

function showResult(mentions, elapsed) {
  const resultsDiv = document.getElementById('results')
  const highlightedDiv = document.getElementById('highlighted')
  const resultInfo = document.getElementById('result-info')

  if (!mentions?.length) {
    resultsDiv.innerHTML = '<div class="empty">No mentions found</div>'
    return
  }

  resultInfo.classList.remove('hidden')
  document.getElementById('ri-count').textContent = mentions.length
  document.getElementById('ri-elapsed').textContent = (elapsed || '-') + 's'

  // Type breakdown bar
  const types = {}
  mentions.forEach(m => { types[m.type] = (types[m.type] || 0) + 1 })
  const barHtml = Object.entries(types).map(([t, c]) => {
    const pct = (c / mentions.length * 100).toFixed(0)
    return `<div class="type-bar-item" style="flex:${c}">
      <div class="type-bar-fill type-${t}" style="width:${pct}%"></div>
      <span class="type-bar-label">${t}: ${c}</span>
    </div>`
  }).join('')
  resultInfo.innerHTML += `<div class="type-bar">${barHtml}</div>`

  const text = document.getElementById('input-text').value
  highlightText(text, mentions)

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
      ${m.candidates?.length ? `<div class="candidates">${m.candidates.map(c => `<span class="candidate">${c}</span>`).join('')}</div>` : ''}
    </div>
  `).join('')
}

function highlightText(text, mentions) {
  const el = document.getElementById('highlighted')
  if (!mentions?.length) { el.classList.add('hidden'); return }
  const sorted = [...mentions].sort((a, b) => b.position[0] - a.position[0])
  let html = escapeHtml(text)
  for (const m of sorted) {
    const st = m.position[0], en = m.position[1]
    const mt = escapeHtml(text.substring(st, en))
    html = html.substring(0, st) + `<span class="cite-${m.type}" title="${m.type}">${mt}</span>` + html.substring(en)
  }
  el.innerHTML = html
  el.classList.remove('hidden')
}

function clearResults() {
  document.getElementById('results').innerHTML = ''
  document.getElementById('highlighted').classList.add('hidden')
  document.getElementById('result-info').classList.add('hidden')
}

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
      const inpStr = typeof input === 'object' ? JSON.stringify(input, null, 2) : String(input)
      detailHtml += `<div class="log-io"><div class="log-label">Input</div><div class="log-value">${escapeHtml(inpStr)}</div></div>`
    }
    if (output) {
      const outStr = typeof output === 'object' ? JSON.stringify(output, null, 2) : String(output)
      detailHtml += `<div class="log-io"><div class="log-label">Output</div><div class="log-value output">${escapeHtml(outStr)}</div></div>`
    }
    detailHtml += '</div>'
  }

  const div = document.createElement('div')
  let extraClass = ''
  if (agent === 'llm_verify') extraClass = ' llm_verify'
  if (action === 'rule_extracted' || action === 'ner_qwen_extracted' || 
      action === 'ner_healthbert_extracted' || action === 'merged' ||
      action === 'positions_fixed' || action === 'llm_verified') extraClass = ' extraction-summary'
  div.className = `log-entry ${agent}${extraClass}`
  div.innerHTML = `<div class="log-header">
    <span><span class="agent">[${agent}]</span> <span class="action">${action}</span>${message ? ` <span class="error">${message}</span>` : ''}</span>
    <span class="ts">${new Date(entry.ts * 1000).toLocaleTimeString()}</span>
  </div>${detailHtml}`
  logsDiv.appendChild(div)
  logsDiv.scrollTop = logsDiv.scrollHeight
  document.getElementById('log-count').textContent = `${logsDiv.children.length} logs`
}

function clearLogs() {
  document.getElementById('logs').innerHTML = '<div class="empty">Agent logs will appear here...</div>'
  document.getElementById('log-count').textContent = ''
}

function toggleDetail() {
  document.querySelectorAll('.log-detail').forEach(el => {
    el.style.display = el.style.display === 'none' ? 'block' : 'none'
  })
}

function toggleList() {
  document.querySelectorAll('.llm_verify, .extraction-summary').forEach(el => {
    el.style.display = el.style.display === 'none' ? '' : 'none'
  })
}

// Run All files
let runningAll = false
async function runAll() {
  if (runningAll) return
  runningAll = true
  const btn = document.getElementById('run-all-btn')
  const progress = document.getElementById('run-progress')
  const bar = document.getElementById('run-progress-bar')
  const text = document.getElementById('run-progress-text')
  const modelType = document.getElementById('model-select').value
  const useLlm = modelType !== ''
  const folder = document.getElementById('folder-select').value

  btn.disabled = true
  btn.textContent = '■ Stop'
  progress.classList.remove('hidden')
  text.textContent = 'Starting...'
  bar.style.width = '0%'
  clearLogs()

  const resp = await fetch('/api/run_all', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ use_llm: useLlm, model: modelType, folder })
  })
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
        if (msg.type === 'progress') {
          const d = msg.data
          const pct = Math.round(d.completed / d.total * 100)
          bar.style.width = pct + '%'
          text.textContent = `[${d.completed}/${d.total}] File ${d.file}: ${d.mentions} mentions (${d.elapsed}s) | Qwen=${d.stats?.qwen || 0} HB=${d.stats?.healthbert || 0} Rules=${d.stats?.rule || 0} LLMc=${d.stats?.llm_corrected || 0}`
        } else if (msg.type === 'log') {
          addLog(msg.data)
        } else if (msg.type === 'complete') {
          const d = msg.data
          bar.style.width = '100%'
          text.textContent = `Complete: ${d.completed} files, ${d.total_mentions} total mentions → ${d.folder}`
          await loadFiles()
          await loadStats()
        } else if (msg.type === 'error') {
          addLog({ agent: 'system', action: `Error: ${msg.data.file}`, message: msg.data.message, ts: Date.now() / 1000 })
        }
      } catch (e) {}
    }
  }

  btn.disabled = false
  btn.textContent = '▶ Run All'
  runningAll = false
}

init()

// Event listeners
document.getElementById('run-all-btn').addEventListener('click', runAll)
