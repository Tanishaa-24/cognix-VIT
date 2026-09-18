import { useState, useRef } from 'react'
import axios from 'axios'

const ALERT_COLORS = {
  normal:   { bg: 'rgba(34,197,94,0.1)',   border: '#22c55e', text: '#22c55e',  label: '🟢 NORMAL' },
  moderate: { bg: 'rgba(234,179,8,0.1)',   border: '#eab308', text: '#eab308',  label: '🟡 VIRAL' },
  high:     { bg: 'rgba(239,68,68,0.1)',   border: '#ef4444', text: '#ef4444',  label: '🔴 BACTERIAL' },
}

const CLASS_COLORS = {
  'Normal':              '#22c55e',
  'Bacterial Pneumonia': '#ef4444',
  'Viral Pneumonia':     '#eab308',
}

export default function XRayModule({ onResult, loading, setLoading }) {
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)
  const [fileName, setFileName] = useState(null)
  const [preview, setPreview]   = useState(null)
  const inputRef = useRef()

  const handleFile = async (file) => {
    if (!file) return
    setFileName(file.name)
    setError(null)
    setLoading(true)

    // Show image preview
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(file)

    try {
      const form = new FormData()
      form.append('file', file)
      const { data } = await axios.post('/xray/predict', form)
      setResult(data)
      if (onResult) onResult(data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Upload failed. Send a JPEG chest X-ray image.')
    } finally {
      setLoading(false)
    }
  }

  const alert = result ? ALERT_COLORS[result.alert_level] : null

  return (
    <div style={{ background: '#0f1729', border: '1px solid #1e2d4d', borderRadius: 12, padding: 20 }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: '#f59e0b' }}>MODULE 03</div>
          <h2 className="text-white font-bold text-lg">Chest X-Ray Classifier</h2>
          <p className="text-xs mt-1" style={{ color: '#64748b' }}>3-class ResNet18 · Normal / Bacterial / Viral</p>
        </div>
        {result && alert && (
          <div className="px-3 py-1.5 rounded-full text-xs font-bold"
            style={{ background: alert.bg, border: `1px solid ${alert.border}`, color: alert.text }}>
            {alert.label}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Upload + Preview column */}
        <div>
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
            className="cursor-pointer rounded-lg text-center transition-all mb-3"
            style={{
              border: '2px dashed #1e2d4d',
              background: '#080d1a',
              padding: preview ? '4px' : '20px 12px',
              overflow: 'hidden',
              minHeight: preview ? 'auto' : 80,
            }}
            onMouseEnter={e => e.currentTarget.style.borderColor = '#f59e0b'}
            onMouseLeave={e => e.currentTarget.style.borderColor = '#1e2d4d'}
          >
            <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" className="hidden"
              onChange={e => handleFile(e.target.files[0])} />
            {preview ? (
              <img src={preview} alt="X-ray preview"
                style={{ width: '100%', maxHeight: 180, objectFit: 'contain', borderRadius: 6, filter: 'brightness(0.9) contrast(1.1)' }} />
            ) : (
              <>
                <div className="text-2xl mb-1">🩻</div>
                <div className="text-sm font-medium" style={{ color: '#94a3b8' }}>
                  {fileName || 'Drop X-ray image or click to upload'}
                </div>
                <div className="text-xs mt-1" style={{ color: '#475569' }}>.jpg / .jpeg / .png</div>
              </>
            )}
          </div>

          {preview && (
            <button onClick={() => inputRef.current?.click()}
              className="w-full text-xs py-1.5 rounded-lg transition-all"
              style={{ background: '#0a1020', border: '1px solid #1e2d4d', color: '#64748b', cursor: 'pointer' }}>
              Upload different image
            </button>
          )}
        </div>

        {/* Results column */}
        <div className="flex flex-col justify-center">
          {loading && (
            <div className="text-center py-4">
              <div className="inline-block w-6 h-6 border-2 border-yellow-500 border-t-transparent rounded-full animate-spin" />
              <div className="text-xs mt-2" style={{ color: '#64748b' }}>Analyzing X-ray...</div>
            </div>
          )}

          {error && (
            <div className="text-xs p-3 rounded-lg" style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)' }}>
              {error}
            </div>
          )}

          {!loading && !error && !result && (
            <div className="text-center py-6" style={{ color: '#334155' }}>
              <div className="text-3xl mb-2">🩺</div>
              <div className="text-xs">Upload a chest X-ray to detect pneumonia</div>
            </div>
          )}

          {result && !loading && (
            <div className="space-y-3 fade-in">
              {/* Main prediction */}
              <div className="rounded-lg p-3" style={{ background: alert?.bg, border: `1px solid ${alert?.border}` }}>
                <div className="text-xs mb-1" style={{ color: '#94a3b8' }}>Diagnosis</div>
                <div className="text-lg font-black" style={{ color: alert?.text }}>{result.predicted_class}</div>
                <div className="text-xs mt-0.5 font-bold" style={{ color: alert?.text }}>
                  {(result.confidence * 100).toFixed(1)}% confidence
                </div>
              </div>

              {/* Clinical note */}
              <div className="text-xs p-2 rounded" style={{ background: '#0a1020', border: '1px solid #1e2d4d', color: '#94a3b8', lineHeight: '1.5' }}>
                {result.clinical_note}
              </div>

              {/* Top 3 bars */}
              <div className="space-y-1.5">
                {result.top3.map((item, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="w-28 text-xs truncate" style={{ color: i === 0 ? alert?.text : '#64748b' }}>{item.class}</div>
                    <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: '#1e2d4d' }}>
                      <div className="h-full rounded-full shap-bar transition-all"
                        style={{ width: `${item.probability * 100}%`, background: CLASS_COLORS[item.class] || '#64748b' }} />
                    </div>
                    <div className="w-10 text-right text-xs font-mono" style={{ color: '#94a3b8' }}>
                      {(item.probability * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
