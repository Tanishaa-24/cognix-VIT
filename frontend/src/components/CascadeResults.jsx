import { useState } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine
} from 'recharts'

const STAGE_CONFIG = {
  1: { color: '#22c55e', label: 'Stage 1 — Early',     desc: 'Low cascade risk. Preventive monitoring recommended.' },
  2: { color: '#eab308', label: 'Stage 2 — Developing', desc: 'Moderate risk. Consider intervention within 4 weeks.' },
  3: { color: '#f97316', label: 'Stage 3 — Advanced',   desc: 'High cascade risk. Prioritize intervention now.' },
  4: { color: '#ef4444', label: 'Stage 4 — Critical',   desc: 'Imminent cascade event. Immediate action required.' },
}

const IV_LABELS = {
  anticoagulation: { label: 'Anticoagulation',    icon: '💊', color: '#3b82f6' },
  beta_blocker:    { label: 'Beta-Blocker',        icon: '🫀', color: '#8b5cf6' },
  lifestyle:       { label: 'Lifestyle Changes',   icon: '🥗', color: '#22c55e' },
  all:             { label: 'Combined Therapy',    icon: '⚡', color: '#f59e0b' },
}

const FEATURE_LABELS = {
  age: 'Age', gender: 'Sex', hypertension: 'Hypertension', heart_disease: 'Heart Disease',
  ever_married: 'Marital Status', work_type: 'Work Type', Residence_type: 'Residence',
  avg_glucose_level: 'Glucose Level', bmi: 'BMI', smoking_status: 'Smoking',
  arrhythmia_risk_weight: 'Arrhythmia Risk Weight',
}

function clean(name) {
  if (name.startsWith('arr_')) return `Arrhythmia: ${name.slice(4)}`
  return FEATURE_LABELS[name] || name
}

export default function CascadeResults({ result, ecgResult, xrayResult }) {
  const [activeIv, setActiveIv] = useState('all')
  const stage   = STAGE_CONFIG[result.cascade_stage]
  const tl      = result.timeline
  const cf      = result.counterfactuals || {}

  // Build timeline chart data
  const timelineData = [
    {
      name: 'Day 30',
      'No Intervention': tl.no_intervention.d30,
      'Conservative':    tl.conservative.d30,
      'Aggressive':      tl.aggressive.d30,
    },
    {
      name: 'Day 90',
      'No Intervention': tl.no_intervention.d90,
      'Conservative':    tl.conservative.d90,
      'Aggressive':      tl.aggressive.d90,
    },
    {
      name: '1 Year',
      'No Intervention': tl.no_intervention.d365,
      'Conservative':    tl.conservative.d365,
      'Aggressive':      tl.aggressive.d365,
    },
  ]

  // SHAP: filter to meaningful features
  const shapData = result.shap_explanation
    .filter(d => !d.feature.startsWith('arr_') || Math.abs(d.shap_value) > 0.05)
    .slice(0, 8)
    .map(d => ({ ...d, label: clean(d.feature) }))

  const maxShap = Math.max(...shapData.map(d => Math.abs(d.shap_value)), 0.01)

  return (
    <div className="fade-in space-y-4">
      {/* Cascade stage banner */}
      <div className="rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4"
        style={{ background: '#0f1729', border: `1px solid ${stage.color}`, boxShadow: `0 0 24px ${stage.color}22` }}>
        <div>
          <div className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: stage.color }}>
            CASCADE STAGE
          </div>
          <div className="text-2xl font-black text-white">{stage.label}</div>
          <div className="text-sm mt-1" style={{ color: '#94a3b8' }}>{stage.desc}</div>
        </div>
        <div className="flex gap-4">
          <div className="text-center">
            <div className="text-xs mb-1" style={{ color: '#64748b' }}>Base Cascade Risk</div>
            <div className="text-2xl font-black" style={{ color: stage.color }}>{tl.base_probability}%</div>
          </div>
          <div className="text-center">
            <div className="text-xs mb-1" style={{ color: '#64748b' }}>Cardiac Risk</div>
            <div className="text-2xl font-black" style={{ color: result.cardiac_risk > 60 ? '#ef4444' : '#f97316' }}>
              {result.cardiac_risk}%
            </div>
          </div>
        </div>
      </div>

      {/* Recommended action */}
      <div className="rounded-xl p-4" style={{ background: '#0a1020', border: '1px solid #1e2d4d' }}>
        <div className="text-xs font-bold uppercase tracking-widest mb-2" style={{ color: '#64748b' }}>AI Recommended Action</div>
        <div className="text-sm font-medium text-white">{result.recommended_action}</div>
        {xrayResult && xrayResult.alert_level !== 'normal' && (
          <div className="mt-2 text-xs p-2 rounded"
            style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)', color: '#fbbf24' }}>
            🩻 X-Ray co-finding: <strong>{xrayResult.predicted_class}</strong> ({(xrayResult.confidence*100).toFixed(0)}% conf) — {xrayResult.clinical_note}
          </div>
        )}
      </div>

      {/* Clinical Insights */}
      {result.insights && (
        <div className="rounded-xl p-4 space-y-3 fade-in" style={{ background: '#0f1729', border: '1px solid #1e2d4d' }}>
          <div className="text-xs font-bold uppercase tracking-widest" style={{ color: '#64748b' }}>Clinical Insights</div>

          {/* Danger flag — cross-modal combination */}
          {result.insights.danger_flag && (
            <div className="rounded-lg p-3 flex gap-3 items-start pulse-critical"
              style={{ background: 'rgba(239,68,68,0.12)', border: '2px solid #ef4444' }}>
              <span className="text-lg">🚨</span>
              <div>
                <div className="text-xs font-black mb-1" style={{ color: '#ef4444' }}>DANGEROUS COMBINATION DETECTED</div>
                <div className="text-xs" style={{ color: '#fca5a5', lineHeight: 1.6 }}>{result.insights.danger_flag}</div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">

            {/* Urgency window */}
            {result.insights.urgency && (() => {
              const u = result.insights.urgency
              const colors = { critical: '#ef4444', high: '#f97316', moderate: '#eab308', normal: '#22c55e' }
              const c = colors[u.color] || '#64748b'
              return (
                <div className="rounded-lg p-3" style={{ background: '#0a1020', border: `1px solid ${c}` }}>
                  <div className="text-xs font-bold uppercase tracking-wider mb-1" style={{ color: '#64748b' }}>Intervention Window</div>
                  <div className="text-lg font-black" style={{ color: c }}>{u.label}</div>
                  {u.days > 0 && <div className="text-xs mt-1" style={{ color: '#475569' }}>Optimal window: next {u.days} days</div>}
                  {u.days === 0 && <div className="text-xs mt-1" style={{ color: '#ef4444' }}>Act now — window closing</div>}
                </div>
              )
            })()}

            {/* CHA₂DS₂-VASc */}
            {result.insights.chadsvasc ? (
              <div className="rounded-lg p-3" style={{ background: '#0a1020', border: '1px solid #3b82f6' }}>
                <div className="text-xs font-bold uppercase tracking-wider mb-1" style={{ color: '#64748b' }}>CHA₂DS₂-VASc Score</div>
                <div className="flex items-end gap-2">
                  <div className="text-2xl font-black" style={{ color: '#3b82f6' }}>{result.insights.chadsvasc.score}</div>
                  <div className="text-xs mb-1" style={{ color: '#475569' }}>/ 9</div>
                </div>
                <div className="text-xs font-semibold" style={{ color: '#f97316' }}>{result.insights.chadsvasc.annual_risk}% annual stroke risk</div>
                <div className="text-xs mt-1" style={{ color: '#94a3b8' }}>{result.insights.chadsvasc.recommendation}</div>
              </div>
            ) : (
              <div className="rounded-lg p-3" style={{ background: '#0a1020', border: '1px solid #1e2d4d' }}>
                <div className="text-xs font-bold uppercase tracking-wider mb-1" style={{ color: '#64748b' }}>CHA₂DS₂-VASc Score</div>
                <div className="text-xs mt-2" style={{ color: '#334155' }}>Applicable for AFib / AFL patients only</div>
              </div>
            )}

            {/* Drug warnings */}
            {result.insights.drug_warnings?.length > 0 && (
              <div className="rounded-lg p-3" style={{ background: '#0a1020', border: '1px solid #1e2d4d' }}>
                <div className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: '#64748b' }}>Drug Contraindications</div>
                <div className="space-y-2">
                  {result.insights.drug_warnings.map((w, i) => {
                    const c = w.severity === 'high' ? '#ef4444' : w.severity === 'moderate' ? '#f97316' : '#22c55e'
                    return (
                      <div key={i} className="text-xs rounded p-2" style={{ background: `${c}11`, border: `1px solid ${c}44`, color: '#cbd5e1', lineHeight: 1.5 }}>
                        <span style={{ color: c, fontWeight: 700 }}>{w.severity === 'high' ? '⛔' : w.severity === 'moderate' ? '⚠️' : '✓'} </span>
                        {w.text}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main results grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Cascade Timeline Chart */}
        <div style={{ background: '#0f1729', border: '1px solid #1e2d4d', borderRadius: 12, padding: 20 }}>
          <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: '#64748b' }}>
            Cascade Timeline — 3 Futures
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={timelineData} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4d" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis tick={{ fontSize: 11, fill: '#64748b' }} unit="%" domain={[0, 'auto']} />
              <Tooltip
                contentStyle={{ background: '#0f1729', border: '1px solid #1e2d4d', borderRadius: 8, fontSize: 12 }}
                formatter={(v, n) => [`${v}%`, n]}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area type="monotone" dataKey="No Intervention" stroke="#ef4444" fill="rgba(239,68,68,0.1)" strokeWidth={2} />
              <Area type="monotone" dataKey="Conservative"    stroke="#f97316" fill="rgba(249,115,22,0.1)" strokeWidth={2} />
              <Area type="monotone" dataKey="Aggressive"      stroke="#22c55e" fill="rgba(34,197,94,0.1)"  strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* SHAP Explanation */}
        <div style={{ background: '#0f1729', border: '1px solid #1e2d4d', borderRadius: 12, padding: 20 }}>
          <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: '#64748b' }}>
            SHAP Feature Impact
          </div>
          <div className="space-y-2">
            {shapData.map((d, i) => {
              const pct   = Math.abs(d.shap_value) / maxShap * 100
              const color = d.shap_value > 0 ? '#ef4444' : '#22c55e'
              return (
                <div key={i} className="flex items-center gap-2">
                  <div className="w-28 text-xs truncate" style={{ color: '#94a3b8' }} title={d.label}>{d.label}</div>
                  <div className="flex-1 h-5 rounded flex items-center" style={{ background: '#0a1020' }}>
                    <div className="h-full rounded shap-bar flex items-center px-1"
                      style={{ width: `${pct}%`, background: color, minWidth: 4 }}>
                    </div>
                  </div>
                  <div className="w-14 text-right text-xs font-mono" style={{ color }}>
                    {d.shap_value > 0 ? '+' : ''}{d.shap_value.toFixed(3)}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="flex justify-between mt-3 text-xs" style={{ color: '#334155' }}>
            <span>🟢 Reduces risk</span>
            <span>🔴 Increases risk</span>
          </div>
        </div>
      </div>

      {/* Counterfactual Intervention Simulator */}
      <div style={{ background: '#0f1729', border: '1px solid #1e2d4d', borderRadius: 12, padding: 20 }}>
        <div className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: '#64748b' }}>
          Counterfactual Intervention Simulator
        </div>
        <p className="text-xs mb-4" style={{ color: '#475569' }}>
          Patient-specific simulation — "what if we intervene now?"
        </p>

        {/* Intervention selector tabs */}
        <div className="flex gap-2 flex-wrap mb-4">
          {Object.entries(IV_LABELS).map(([k, v]) => (
            <button key={k} onClick={() => setActiveIv(k)}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all"
              style={{
                background:  activeIv === k ? v.color : '#0a1020',
                color:       activeIv === k ? 'white' : '#64748b',
                border:      `1px solid ${activeIv === k ? v.color : '#1e2d4d'}`,
              }}>
              {v.icon} {v.label}
            </button>
          ))}
        </div>

        {/* Active intervention result */}
        {cf[activeIv] && (
          <div className="fade-in grid grid-cols-3 gap-4">
            <div className="rounded-xl p-4 text-center" style={{ background: '#0a1020', border: '1px solid #1e2d4d' }}>
              <div className="text-xs mb-1" style={{ color: '#64748b' }}>Before</div>
              <div className="text-2xl font-black" style={{ color: '#ef4444' }}>
                {cf[activeIv].original_risk}%
              </div>
              <div className="text-xs mt-1" style={{ color: '#475569' }}>Cascade risk</div>
            </div>
            <div className="rounded-xl p-4 text-center flex flex-col items-center justify-center"
              style={{ background: 'rgba(59,130,246,0.05)', border: '1px solid rgba(59,130,246,0.2)' }}>
              <div className="text-3xl mb-1">→</div>
              <div className="text-xs font-bold" style={{ color: '#3b82f6' }}>
                −{cf[activeIv].risk_reduction}%
              </div>
              <div className="text-xs mt-0.5" style={{ color: '#475569' }}>reduction</div>
            </div>
            <div className="rounded-xl p-4 text-center" style={{ background: '#0a1020', border: '1px solid rgba(34,197,94,0.3)' }}>
              <div className="text-xs mb-1" style={{ color: '#64748b' }}>After</div>
              <div className="text-2xl font-black" style={{ color: '#22c55e' }}>
                {cf[activeIv].modified_risk}%
              </div>
              <div className="text-xs mt-1" style={{ color: '#475569' }}>Cascade risk</div>
            </div>
          </div>
        )}

        {/* All interventions comparison bar */}
        <div className="mt-4">
          <div className="text-xs mb-3" style={{ color: '#475569' }}>Risk reduction by intervention</div>
          <div className="space-y-2">
            {Object.entries(IV_LABELS).map(([k, v]) => {
              if (!cf[k]) return null
              const maxRed = Math.max(...Object.keys(IV_LABELS).filter(x => cf[x]).map(x => cf[x].risk_reduction), 0.1)
              const pct    = (cf[k].risk_reduction / maxRed) * 100
              return (
                <div key={k} className="flex items-center gap-2">
                  <div className="w-28 text-xs" style={{ color: '#94a3b8' }}>{v.icon} {v.label}</div>
                  <div className="flex-1 h-4 rounded overflow-hidden" style={{ background: '#0a1020' }}>
                    <div className="h-full rounded shap-bar"
                      style={{ width: `${pct}%`, background: v.color, opacity: activeIv === k ? 1 : 0.45 }} />
                  </div>
                  <div className="w-12 text-right text-xs font-mono" style={{ color: v.color }}>
                    −{cf[k].risk_reduction.toFixed(1)}%
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
