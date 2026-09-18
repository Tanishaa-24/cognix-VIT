import { useState } from 'react'
import axios from 'axios'

const DEFAULTS = {
  age: 63, gender: 1, hypertension: 1, heart_disease: 0,
  ever_married: 1, work_type: 2, Residence_type: 1,
  avg_glucose_level: 142.5, bmi: 28.4, smoking_status: 1,
}

const FIELD_META = {
  age:               { label: 'Age (years)',          type: 'number', min: 18,   max: 120 },
  gender:            { label: 'Biological Sex',        type: 'select', opts: [['0','Female'],['1','Male']] },
  hypertension:      { label: 'Hypertension',          type: 'select', opts: [['0','No'],['1','Yes']] },
  heart_disease:     { label: 'Heart Disease History', type: 'select', opts: [['0','No'],['1','Yes']] },
  avg_glucose_level: { label: 'Avg Glucose (mg/dL)',   type: 'number', min: 60,   max: 400,  step: 0.1 },
  bmi:               { label: 'BMI (kg/m²)',            type: 'number', min: 10,   max: 70,   step: 0.1 },
  smoking_status:    { label: 'Smoking Status',         type: 'select', opts: [['0','Never'],['1','Formerly'],['2','Currently']] },
  ever_married:      { label: 'Ever Married',           type: 'select', opts: [['0','No'],['1','Yes']] },
  work_type:         { label: 'Work Type',              type: 'select', opts: [['0','Children'],['1','Govt Job'],['2','Private'],['3','Self-employed'],['4','Never worked']] },
  Residence_type:    { label: 'Residence',              type: 'select', opts: [['0','Rural'],['1','Urban']] },
}

export default function ClinicalForm({ ecgResult, xrayResult, onResult, loading, setLoading }) {
  const [fields, setFields] = useState(DEFAULTS)
  const [error, setError]   = useState(null)

  const set = (k, v) => setFields(p => ({ ...p, [k]: parseFloat(v) }))

  const submit = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const payload = {
        ...fields,
        arrhythmia_name:       ecgResult?.predicted_class  || 'NSR',
        arrhythmia_confidence: ecgResult?.confidence       || 1.0,
        xray_class:            xrayResult?.predicted_class || 'Normal',
      }
      const { data } = await axios.post('/cascade/predict', payload)
      // Fetch all counterfactuals in parallel
      const { data: cfs } = await axios.post('/cascade/all-interventions', payload)
      onResult({ ...data, counterfactuals: cfs, payload })
    } catch (e) {
      setError(e.response?.data?.detail || 'Analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ background: '#0f1729', border: '1px solid #1e2d4d', borderRadius: 12, padding: 20, height: '100%' }}>
      <div className="mb-4">
        <div className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: '#8b5cf6' }}>MODULE 02</div>
        <h2 className="text-white font-bold text-lg">Patient Clinical Profile</h2>
        <p className="text-xs mt-1" style={{ color: '#64748b' }}>Cascade chain analysis · XGBoost + SHAP</p>
      </div>

      {ecgResult && (
        <div className="rounded-lg p-3 mb-4 flex items-center gap-2"
          style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)' }}>
          <span className="text-sm">⚡</span>
          <span className="text-xs" style={{ color: '#93c5fd' }}>
            ECG injected: <strong>{ecgResult.predicted_class}</strong> ({(ecgResult.confidence * 100).toFixed(0)}% conf) → cascade model
          </span>
        </div>
      )}

      <form onSubmit={submit} className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(FIELD_META).map(([key, meta]) => (
            <div key={key}>
              <label style={{ fontSize: 11, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                {meta.label}
              </label>
              {meta.type === 'select' ? (
                <select value={fields[key]} onChange={e => set(key, e.target.value)}
                  style={{ background: '#080d1a', border: '1px solid #1e2d4d', borderRadius: 6, color: '#e2e8f0', padding: '7px 10px', fontSize: 13, width: '100%', marginTop: 3 }}>
                  {meta.opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              ) : (
                <input type="number" min={meta.min} max={meta.max} step={meta.step || 1}
                  value={fields[key]} onChange={e => set(key, e.target.value)}
                  style={{ background: '#080d1a', border: '1px solid #1e2d4d', borderRadius: 6, color: '#e2e8f0', padding: '7px 10px', fontSize: 13, width: '100%', marginTop: 3 }} />
              )}
            </div>
          ))}
        </div>

        {error && (
          <div className="text-xs p-3 rounded" style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171', border: '1px solid rgba(239,68,68,0.3)' }}>
            {error}
          </div>
        )}

        <button type="submit" disabled={loading}
          style={{ width: '100%', background: loading ? '#1e293b' : 'linear-gradient(135deg,#3b82f6,#8b5cf6)', border: 'none', borderRadius: 8, color: 'white', padding: '11px 0', fontSize: 14, fontWeight: 700, cursor: loading ? 'not-allowed' : 'pointer', transition: 'all 0.2s', marginTop: 4 }}>
          {loading ? '⏳ Running Cascade Analysis...' : '⚡ Run Cascade Analysis'}
        </button>
      </form>
    </div>
  )
}
