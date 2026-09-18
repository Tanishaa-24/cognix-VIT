import { useState } from 'react'
import Header from './components/Header'
import ECGModule from './components/ECGModule'
import ClinicalForm from './components/ClinicalForm'
import CascadeResults from './components/CascadeResults'
import './index.css'

export default function App() {
  const [ecgResult, setEcgResult]         = useState(null)
  const [cascadeResult, setCascadeResult] = useState(null)
  const [loading, setLoading]             = useState({ ecg: false, cascade: false })

  return (
    <div className="min-h-screen" style={{ background: '#080d1a' }}>
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">

        {/* Top row: ECG + Clinical form side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ECGModule
            onResult={setEcgResult}
            loading={loading.ecg}
            setLoading={(v) => setLoading(p => ({ ...p, ecg: v }))}
          />
          <ClinicalForm
            ecgResult={ecgResult}
            onResult={setCascadeResult}
            loading={loading.cascade}
            setLoading={(v) => setLoading(p => ({ ...p, cascade: v }))}
          />
        </div>

        {/* Cascade results — full width */}
        {cascadeResult && (
          <CascadeResults result={cascadeResult} ecgResult={ecgResult} />
        )}
      </main>

      <footer className="text-center py-6 text-xs" style={{ color: '#334155' }}>
        CascadeIQ · Predictive Disease Cascade Intelligence · For clinical decision support only
      </footer>
    </div>
  )
}
