export default function Header() {
  return (
    <header style={{ background: '#0a1020', borderBottom: '1px solid #1e2d4d' }} className="sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: 'linear-gradient(135deg,#3b82f6,#8b5cf6)' }}>
            <span className="text-white text-sm font-black">C</span>
          </div>
          <div>
            <div className="font-bold text-white text-base leading-none">CascadeIQ</div>
            <div className="text-xs mt-0.5" style={{ color: '#475569' }}>Predictive Disease Cascade Intelligence</div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-3 text-xs" style={{ color: '#475569' }}>
            <span>ECG · X-Ray · Cascade Chain · SHAP · Counterfactuals</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1 rounded-full" style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
            <div className="w-1.5 h-1.5 rounded-full bg-green-400" style={{ animation: 'pulse 2s infinite' }} />
            <span className="text-xs font-semibold" style={{ color: '#22c55e' }}>AI Engine Active</span>
          </div>
        </div>
      </div>
    </header>
  )
}
