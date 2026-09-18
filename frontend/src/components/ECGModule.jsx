import { useState, useRef } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts'

const ALERT_COLORS = {
  critical: { bg: 'rgba(239,68,68,0.1)',  border: '#ef4444', text: '#ef4444',  label: '🔴 CRITICAL' },
  high:     { bg: 'rgba(249,115,22,0.1)', border: '#f97316', text: '#f97316',  label: '🟠 HIGH RISK' },
  moderate: { bg: 'rgba(234,179,8,0.1)',  border: '#eab308', text: '#eab308',  label: '🟡 MODERATE' },
  normal:   { bg: 'rgba(34,197,94,0.1)',  border: '#22c55e', text: '#22c55e',  label: '🟢 NORMAL' },
}

export default function ECGModule({ onResult, loading, setLoading }) {
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [fileName, setFileName] = useState(null)
  const inputRef = useRef()

  const handleFile = async (file) => {
    if (!file) return
    setFileName(file.name)
    setError(null)
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const { data } = await axios.post('/ecg/predict', form)
      setResult(data)
      onResult(data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Upload failed. Check file format.')
    } finally {
      setLoading(false)
    }
  }

  const waveformData = result?.waveform?.map((v, i) => ({ t: i, v })) || []
  const saliencyData = result?.saliency  || []
  const alert        = result ? ALERT_COLORS[result.alert_level] : null

  return (
    <div style={{ background: '#0f1729', border: '1px solid #1e2d4d', borderRadius: 12, padding: 20 }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: '#3b82f6' }}>MODULE 01</div>
          <h2 className="text-white font-bold text-lg">ECG Arrhythmia Classifier</h2>
          <p className="text-xs mt-1" style={{ color: '#64748b' }}>17-class 1D ResNet · MIT-BIH Dataset</p>
        </div>
        {result && alert && (
          <div className="px-3 py-1.5 rounded-full text-xs font-bold"
            style={{ background: alert.bg, border: `1px solid ${alert.border}`, color: alert.text }}>
            {alert.label}
          </div>
        )}
      </div>

      {/* Upload zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
        className="cursor-pointer rounded-lg p-4 text-center transition-all mb-4"
        style={{ border: '2px dashed #1e2d4d', background: '#080d1a' }}
        onMouseEnter={e => e.currentTarget.style.borderColor='#3b82f6'}
        onMouseLeave={e => e.currentTarget.style.borderColor='#1e2d4d'}
      >
        <input ref={inputRef} type="file" accept=".mat,.csv" className="hidden"
          onChange={e => handleFile(e.target.files[0])} />
        <div className="text-2xl mb-1">📁</div>
        <div className="text-sm font-medium" style={{ color: '#94a3b8' }}>
          {fileName || 'Drop ECG file or click to upload'}
        </div>
        <div className="text-xs mt-1" style={{ color: '#475569' }}>.mat (MIT-BIH) or .csv</div>
      </div>

      {loading && (
        <div className="text-center py-4">
          <div className="inline-block w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <div className="text-xs mt-2" style={{ color: '#64748b' }}>Analyzing ECG signal...</div>
        </div>
      )}

      {error && (
        <div className="text-xs p-3 rounded-lg mb-3" style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)' }}>
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4 fade-in">
          {/* Prediction result */}
          <div className="rounded-lg p-4" style={{ background: alert?.bg, border: `1px solid ${alert?.border}` }}>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs mb-1" style={{ color: '#94a3b8' }}>Detected Rhythm</div>
                <div className="text-2xl font-black" style={{ color: alert?.text }}>{result.predicted_class}</div>
              </div>
              <div className="text-right">
                <div className="text-xs mb-1" style={{ color: '#94a3b8' }}>Confidence</div>
                <div className="text-xl font-bold text-white">{(result.confidence * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          {/* ECG waveform */}
          <div>
            <div className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: '#475569' }}>
              ECG Waveform · MLII Lead
            </div>
            <div style={{ height: 100 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={waveformData} margin={{ top: 2, right: 4, bottom: 2, left: -30 }}>
                  <XAxis dataKey="t" hide />
                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 9, fill: '#475569' }} />
                  <Line type="monotone" dataKey="v" stroke="#22c55e" dot={false} strokeWidth={1.2} isAnimationActive={true} animationDuration={1500} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Saliency / attention heatmap */}
          <div>
            <div className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: '#475569' }}>
              Temporal Saliency · Regions Driving Classification
            </div>
            <div className="flex gap-px h-8 rounded overflow-hidden">
              {saliencyData.slice(0, 120).map((v, i) => {
                const norm = Math.min(v / (Math.max(...saliencyData) + 1e-6), 1)
                const r = Math.round(norm * 239)
                const g = Math.round((1 - norm) * 197)
                return <div key={i} className="flex-1" style={{ background: `rgb(${r},${g},68)`, opacity: 0.7 + norm * 0.3 }} title={`t=${i}: ${v.toFixed(3)}`} />
              })}
            </div>
            <div className="flex justify-between mt-1 text-xs" style={{ color: '#475569' }}>
              <span>Low attention</span><span>High attention</span>
            </div>
          </div>

          {/* Top-5 classes */}
          <div>
            <div className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: '#475569' }}>Top 5 Predictions</div>
            <div className="space-y-1.5">
              {result.top5.map((item, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className="w-16 text-xs font-mono" style={{ color: i === 0 ? alert?.text : '#64748b' }}>{item.class}</div>
                  <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: '#1e2d4d' }}>
                    <div className="h-full rounded-full transition-all shap-bar"
                      style={{ width: `${item.probability * 100}%`, background: i === 0 ? alert?.border : '#334155' }} />
                  </div>
                  <div className="w-10 text-right text-xs font-mono" style={{ color: '#94a3b8' }}>
                    {(item.probability * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
