const FLABELS = {
  age: 'Age', gender: 'Sex', hypertension: 'Hypertension', heart_disease: 'Heart Disease',
  ever_married: 'Marital Status', work_type: 'Work Type', Residence_type: 'Residence',
  avg_glucose_level: 'Avg Glucose', bmi: 'BMI', smoking_status: 'Smoking',
  arrhythmia_risk_weight: 'Arrhythmia Risk',
}

function cleanFeature(name) {
  if (name.startsWith('arr_')) return `Arrhythmia: ${name.slice(4)}`
  return FLABELS[name] || name
}

function buildReportHTML({ ecgResult, xrayResult, cascadeResult, clinicalPayload }) {
  const stage     = cascadeResult.cascade_stage
  const tl        = cascadeResult.timeline
  const cf        = cascadeResult.counterfactuals || {}
  const shap      = (cascadeResult.shap_explanation || []).slice(0, 8)
  const stageColors = { 1: '#22c55e', 2: '#eab308', 3: '#f97316', 4: '#ef4444' }
  const stageLabels = { 1: 'Stage 1 — Early', 2: 'Stage 2 — Developing', 3: 'Stage 3 — Advanced', 4: 'Stage 4 — Critical' }
  const sc = stageColors[stage] || '#64748b'

  const ivInfo = {
    anticoagulation: { label: 'Anticoagulation', color: '#3b82f6' },
    beta_blocker:    { label: 'Beta-Blocker',     color: '#8b5cf6' },
    lifestyle:       { label: 'Lifestyle',         color: '#22c55e' },
    all:             { label: 'Combined Therapy',  color: '#f59e0b' },
  }

  const maxShap  = Math.max(...shap.map(d => Math.abs(d.shap_value)), 0.01)
  const alertText = { critical: '#ef4444', high: '#ef4444', moderate: '#eab308', normal: '#22c55e' }

  const patientRows = clinicalPayload ? [
    ['Age',          clinicalPayload.age + ' yrs'],
    ['Sex',          clinicalPayload.gender === 1 ? 'Male' : 'Female'],
    ['Hypertension', clinicalPayload.hypertension ? 'Yes' : 'No'],
    ['Heart Disease',clinicalPayload.heart_disease ? 'Yes' : 'No'],
    ['Avg Glucose',  clinicalPayload.avg_glucose_level + ' mg/dL'],
    ['BMI',          clinicalPayload.bmi + ' kg/m²'],
    ['Smoking',      ['Never','Formerly','Currently'][clinicalPayload.smoking_status] || '-'],
    ['Arrhythmia',   clinicalPayload.arrhythmia_name],
  ] : []

  const now = new Date().toLocaleString('en-US', { dateStyle: 'long', timeStyle: 'short' })

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CascadeIQ Patient Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #080d1a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 13px; }
  @media print {
    body { background: white; color: #111; }
    .no-print { display: none !important; }
    .card { break-inside: avoid; }
  }
  h1 { font-size: 22px; font-weight: 900; color: #fff; }
  .header { background: #0a1020; border-bottom: 3px solid #3b82f6; padding: 18px 28px; display: flex; justify-content: space-between; align-items: center; }
  .header-right { text-align: right; font-size: 11px; color: #64748b; }
  .badge { background: linear-gradient(135deg,#3b82f6,#8b5cf6); color: white; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px; }
  .body { padding: 24px 28px; display: grid; gap: 20px; }
  .card { background: #0f1729; border: 1px solid #1e2d4d; border-radius: 10px; padding: 18px; }
  .label { font-size: 10px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #64748b; margin-bottom: 4px; }
  .value { font-size: 22px; font-weight: 900; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .kv-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .kv { }
  .kv .k { font-size: 10px; color: #64748b; font-weight: 600; text-transform: uppercase; }
  .kv .v { font-size: 13px; color: #e2e8f0; font-weight: 600; margin-top: 2px; }
  .stage-banner { border-radius: 10px; padding: 20px; display: flex; justify-content: space-between; align-items: center; }
  .bar-track { background: #0a1020; border-radius: 4px; height: 8px; overflow: hidden; margin-top: 4px; }
  .bar-fill { height: 100%; border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-size: 10px; color: #64748b; font-weight: 700; text-transform: uppercase; padding: 6px 8px; border-bottom: 1px solid #1e2d4d; }
  td { padding: 7px 8px; border-bottom: 1px solid #0a1020; font-size: 12px; }
  .action-note { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.25); border-radius: 8px; padding: 12px 16px; color: #93c5fd; line-height: 1.6; }
  .print-btn { background: linear-gradient(135deg,#3b82f6,#8b5cf6); color: white; border: none; padding: 10px 24px; border-radius: 8px; font-size: 14px; font-weight: 700; cursor: pointer; margin: 16px 28px; display: block; }
  .footer { text-align: center; padding: 16px; font-size: 10px; color: #334155; border-top: 1px solid #1e2d4d; margin-top: 8px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>CascadeIQ</h1>
    <div style="color:#64748b;font-size:12px;margin-top:2px;">Predictive Disease Cascade Intelligence — Patient Report</div>
  </div>
  <div class="header-right">
    <div style="color:#e2e8f0;font-weight:700;margin-bottom:4px;">${now}</div>
    <span class="badge">Clinical Decision Support</span>
  </div>
</div>

<button class="print-btn no-print" onclick="window.print()">🖨️ Print / Save as PDF</button>

<div class="body">

  <!-- CASCADE STAGE BANNER -->
  <div class="card stage-banner" style="border:2px solid ${sc};box-shadow:0 0 24px ${sc}22;">
    <div>
      <div class="label" style="color:${sc}">CASCADE STAGE</div>
      <div class="value" style="color:${sc}">${stageLabels[stage] || 'Stage ' + stage}</div>
      <div style="color:#94a3b8;font-size:12px;margin-top:4px;">${
        { 1: 'Low cascade risk. Preventive monitoring recommended.',
          2: 'Moderate risk. Consider intervention within 4 weeks.',
          3: 'High cascade risk. Prioritize intervention now.',
          4: 'Imminent cascade event. Immediate action required.' }[stage] || ''
      }</div>
    </div>
    <div class="grid2">
      <div style="text-align:center;">
        <div class="label">Base Cascade Risk</div>
        <div class="value" style="color:${sc}">${tl.base_probability}%</div>
      </div>
      <div style="text-align:center;">
        <div class="label">Cardiac Risk</div>
        <div class="value" style="color:${cascadeResult.cardiac_risk > 60 ? '#ef4444' : '#f97316'}">${cascadeResult.cardiac_risk}%</div>
      </div>
    </div>
  </div>

  <!-- RECOMMENDED ACTION -->
  <div class="card">
    <div class="label">AI Recommended Action</div>
    <div class="action-note" style="margin-top:8px;">${cascadeResult.recommended_action.replace(/[🔴🟡🟢⚠️🩺]/gu, '').trim()}</div>
    ${xrayResult && xrayResult.alert_level !== 'normal' ? `
    <div class="action-note" style="margin-top:8px;border-color:rgba(245,158,11,0.3);color:#fbbf24;">
      🩻 X-Ray co-finding: <strong>${xrayResult.predicted_class}</strong> (${(xrayResult.confidence*100).toFixed(0)}% conf) — ${xrayResult.clinical_note}
    </div>` : ''}
  </div>

  <!-- ECG + XRAY -->
  ${(ecgResult || xrayResult) ? `
  <div class="grid2">
    ${ecgResult ? `
    <div class="card">
      <div class="label" style="color:#3b82f6">MODULE 01 — ECG Arrhythmia</div>
      <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:10px;">
        <div>
          <div style="font-size:11px;color:#94a3b8;">Detected Rhythm</div>
          <div style="font-size:20px;font-weight:900;color:${alertText[ecgResult.alert_level] || '#22c55e'}">${ecgResult.predicted_class}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:11px;color:#94a3b8;">Confidence</div>
          <div style="font-size:18px;font-weight:900;color:#fff;">${(ecgResult.confidence*100).toFixed(1)}%</div>
        </div>
      </div>
      <div style="margin-top:10px;">
        ${(ecgResult.top5 || []).map(item => `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
          <div style="width:60px;font-size:11px;color:#94a3b8;">${item.class}</div>
          <div class="bar-track" style="flex:1;"><div class="bar-fill" style="width:${(item.probability*100).toFixed(1)}%;background:${alertText[ecgResult.alert_level]||'#22c55e'};"></div></div>
          <div style="width:38px;text-align:right;font-size:11px;color:#64748b;">${(item.probability*100).toFixed(1)}%</div>
        </div>`).join('')}
      </div>
    </div>` : ''}
    ${xrayResult ? `
    <div class="card">
      <div class="label" style="color:#f59e0b">MODULE 03 — Chest X-Ray</div>
      <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:10px;">
        <div>
          <div style="font-size:11px;color:#94a3b8;">Diagnosis</div>
          <div style="font-size:18px;font-weight:900;color:${alertText[xrayResult.alert_level]||'#22c55e'}">${xrayResult.predicted_class}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:11px;color:#94a3b8;">Confidence</div>
          <div style="font-size:18px;font-weight:900;color:#fff;">${(xrayResult.confidence*100).toFixed(1)}%</div>
        </div>
      </div>
      <div style="margin-top:8px;">
        ${(xrayResult.top3 || []).map(item => {
          const cls3Colors = {'Normal':'#22c55e','Bacterial Pneumonia':'#ef4444','Viral Pneumonia':'#eab308'}
          return `
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <div style="width:100px;font-size:11px;color:#94a3b8;">${item.class}</div>
            <div class="bar-track" style="flex:1;"><div class="bar-fill" style="width:${(item.probability*100).toFixed(1)}%;background:${cls3Colors[item.class]||'#64748b'};"></div></div>
            <div style="width:38px;text-align:right;font-size:11px;color:#64748b;">${(item.probability*100).toFixed(1)}%</div>
          </div>`
        }).join('')}
      </div>
      <div style="margin-top:8px;font-size:11px;color:#94a3b8;">${xrayResult.clinical_note}</div>
    </div>` : ''}
  </div>` : ''}

  <!-- PATIENT PROFILE -->
  ${patientRows.length ? `
  <div class="card">
    <div class="label">Patient Clinical Profile</div>
    <div class="kv-grid" style="margin-top:12px;">
      ${patientRows.map(([k, v]) => `<div class="kv"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('')}
    </div>
  </div>` : ''}

  <!-- CASCADE TIMELINE -->
  <div class="card">
    <div class="label">Cascade Risk Timeline — 3 Scenarios</div>
    <table style="margin-top:12px;">
      <tr>
        <th>Scenario</th><th>30 Days</th><th>90 Days</th><th>1 Year</th>
      </tr>
      <tr>
        <td style="color:#e2e8f0;font-weight:600;">No Intervention</td>
        <td style="color:#ef4444;font-weight:700;">${tl.no_intervention.d30}%</td>
        <td style="color:#ef4444;font-weight:700;">${tl.no_intervention.d90}%</td>
        <td style="color:#ef4444;font-weight:700;">${tl.no_intervention.d365}%</td>
      </tr>
      <tr>
        <td style="color:#e2e8f0;font-weight:600;">Conservative Rx</td>
        <td style="color:#f97316;font-weight:700;">${tl.conservative.d30}%</td>
        <td style="color:#f97316;font-weight:700;">${tl.conservative.d90}%</td>
        <td style="color:#f97316;font-weight:700;">${tl.conservative.d365}%</td>
      </tr>
      <tr>
        <td style="color:#e2e8f0;font-weight:600;">Aggressive Rx</td>
        <td style="color:#22c55e;font-weight:700;">${tl.aggressive.d30}%</td>
        <td style="color:#22c55e;font-weight:700;">${tl.aggressive.d90}%</td>
        <td style="color:#22c55e;font-weight:700;">${tl.aggressive.d365}%</td>
      </tr>
    </table>
  </div>

  <!-- SHAP + COUNTERFACTUALS -->
  <div class="grid2">

    <!-- SHAP -->
    <div class="card">
      <div class="label">Top Risk Factors (SHAP)</div>
      <div style="margin-top:12px;">
        ${shap.map(d => {
          const pct   = Math.abs(d.shap_value) / maxShap * 100
          const color = d.shap_value > 0 ? '#ef4444' : '#22c55e'
          return `
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">
            <div style="width:100px;font-size:11px;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${cleanFeature(d.feature)}</div>
            <div class="bar-track" style="flex:1;"><div class="bar-fill" style="width:${pct.toFixed(1)}%;background:${color};"></div></div>
            <div style="width:46px;text-align:right;font-size:11px;color:${color};font-family:monospace;">${d.shap_value>0?'+':''}${d.shap_value.toFixed(3)}</div>
          </div>`
        }).join('')}
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:10px;color:#334155;">
        <span>🟢 Reduces risk</span><span>🔴 Increases risk</span>
      </div>
    </div>

    <!-- COUNTERFACTUALS -->
    <div class="card">
      <div class="label">Counterfactual Interventions</div>
      <div style="margin-top:12px;display:flex;flex-direction:column;gap:10px;">
        ${Object.entries(ivInfo).map(([k, v]) => {
          const d = cf[k]
          if (!d) return ''
          return `
          <div style="background:#0a1020;border:1px solid #1e2d4d;border-radius:8px;padding:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:12px;font-weight:700;color:${v.color};">${v.label}</span>
              <span style="font-size:18px;font-weight:900;color:${v.color};">−${d.risk_reduction.toFixed(1)}%</span>
            </div>
            <div style="display:flex;gap:16px;margin-top:4px;font-size:11px;color:#64748b;">
              <span>Before: <strong style="color:#ef4444">${d.original_risk}%</strong></span>
              <span>After: <strong style="color:#22c55e">${d.modified_risk}%</strong></span>
            </div>
          </div>`
        }).join('')}
      </div>
    </div>
  </div>

</div>

<div class="footer">
  CascadeIQ · For clinical decision support only · Not a substitute for professional medical advice · Generated ${now}
</div>

<script>
  // Auto-open print dialog after a short delay so content renders
  setTimeout(() => window.print(), 800)
</script>
</body>
</html>`
}

export default function ReportExport({ ecgResult, xrayResult, cascadeResult, clinicalPayload }) {
  if (!cascadeResult) return null

  const generate = () => {
    const html = buildReportHTML({ ecgResult, xrayResult, cascadeResult, clinicalPayload })
    const win  = window.open('', '_blank', 'width=900,height=700')
    win.document.write(html)
    win.document.close()
  }

  return (
    <button
      onClick={generate}
      className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all"
      style={{
        background: 'linear-gradient(135deg, #1e3a5f, #2d1b69)',
        border: '1px solid #3b82f6',
        color: '#93c5fd',
        cursor: 'pointer',
      }}
      onMouseEnter={e => { e.currentTarget.style.background = 'linear-gradient(135deg,#2563eb,#7c3aed)'; e.currentTarget.style.color = 'white' }}
      onMouseLeave={e => { e.currentTarget.style.background = 'linear-gradient(135deg, #1e3a5f, #2d1b69)'; e.currentTarget.style.color = '#93c5fd' }}
    >
      <span>📄</span>
      Download PDF Report
    </button>
  )
}
