import { useState } from 'react'
import Header from './components/Header'
import ECGModule from './components/ECGModule'
import XRayModule from './components/XRayModule'
import ClinicalForm from './components/ClinicalForm'
import CascadeResults from './components/CascadeResults'
import ReportExport from './components/ReportExport'
import './index.css'

function StepBadge({ n, label, done }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-black flex-shrink-0"
        style={{ background: done ? '#22c55e' : 'linear-gradient(135deg,#3b82f6,#8b5cf6)', color: 'white' }}>
        {done ? '✓' : n}
      </div>
      <span className="text-xs font-semibold" style={{ color: done ? '#22c55e' : '#94a3b8' }}>{label}</span>
    </div>
  )
}

export default function App() {
  const [ecgResult,     setEcgResult]     = useState(null)
  const [xrayResult,    setXrayResult]    = useState(null)
  const [cascadeResult, setCascadeResult] = useState(null)
  const [loading, setLoading] = useState({ ecg: false, xray: false, cascade: false })

  return (
    <div className="min-h-screen" style={{ background: '#080d1a' }}>
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-5 space-y-5">

        {/* Workflow stepper */}
        <div className="flex items-center gap-6 px-1">
          <StepBadge n="1" label="Upload ECG"    done={!!ecgResult} />
          <div className="flex-1 h-px" style={{ background: '#1e2d4d' }} />
          <StepBadge n="2" label="Upload X-Ray"  done={!!xrayResult} />
          <div className="flex-1 h-px" style={{ background: '#1e2d4d' }} />
          <StepBadge n="3" label="Clinical Data" done={!!cascadeResult} />
          <div className="flex-1 h-px" style={{ background: '#1e2d4d' }} />
          <StepBadge n="4" label="Cascade Analysis" done={!!cascadeResult} />
        </div>

        {/* Main 2-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

          {/* Left column — ECG + X-Ray stacked */}
          <div className="flex flex-col gap-5">
            <ECGModule
              onResult={setEcgResult}
              loading={loading.ecg}
              setLoading={(v) => setLoading(p => ({ ...p, ecg: v }))}
            />
            <XRayModule
              onResult={setXrayResult}
              loading={loading.xray}
              setLoading={(v) => setLoading(p => ({ ...p, xray: v }))}
            />
          </div>

          {/* Right column — Clinical form (full height) */}
          <ClinicalForm
            ecgResult={ecgResult}
            xrayResult={xrayResult}
            onResult={setCascadeResult}
            loading={loading.cascade}
            setLoading={(v) => setLoading(p => ({ ...p, cascade: v }))}
          />
        </div>

        {/* Cascade results */}
        {cascadeResult && (
          <div className="space-y-4 fade-in">
            <div className="flex items-center justify-between px-1">
              <div className="text-xs font-bold uppercase tracking-widest" style={{ color: '#3b82f6' }}>
                Cascade Analysis Results
              </div>
              <ReportExport
                ecgResult={ecgResult}
                xrayResult={xrayResult}
                cascadeResult={cascadeResult}
                clinicalPayload={cascadeResult.payload}
              />
            </div>
            <CascadeResults result={cascadeResult} ecgResult={ecgResult} xrayResult={xrayResult} />
          </div>
        )}
      </main>

      <footer className="text-center py-5 text-xs" style={{ color: '#1e2d4d' }}>
        CascadeIQ · Predictive Disease Cascade Intelligence · For clinical decision support only
      </footer>
    </div>
  )
}
