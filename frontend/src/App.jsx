import { useState } from 'react'
import './index.css'

const API_GATEWAY = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function ForecastApplication() {
  const [formData, setFormData] = useState({
    age: 50,
    gender: 1,
    height: 165,
    weight: 70,
    ap_hi: 120,
    ap_lo: 80,
    cholesterol: 1,
    gluc: 1,
    smoke: 0,
    alco: 0,
    active: 1
  })
  const [isProcessing, setIsProcessing] = useState(false)
  const [forecastData, setForecastData] = useState(null)
  const [fetchError, setFetchError] = useState('')

  const handleInputChange = (e) => {
    const { name, value, type } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : Number(value)
    }))
  }

  const executeForecastQuery = async () => {
    setFetchError('')
    setForecastData(null)
    setIsProcessing(true)

    try {
      const response = await fetch(`${API_GATEWAY}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      })
      const jsonPayload = await response.json()
      
      if (!response.ok) {
        throw new Error(jsonPayload.detail || `Server exception ${response.status}`)
      }
      setForecastData(jsonPayload)
    } catch (err) {
      if (err.message.includes('fetch') || err.message.includes('NetworkError')) {
        setFetchError('CONNECTION REFUSED: Diagnostic engine offline.')
      } else {
        setFetchError(err.message || 'RUNTIME EXCEPTION OCCURRED.')
      }
    } finally {
      setIsProcessing(false)
    }
  }

  const isHighRisk = forecastData?.prediction === 1
  const prob = forecastData?.probability !== null && forecastData?.probability !== undefined ? forecastData.probability : (isHighRisk ? 1 : 0)
  const riskProb = isHighRisk ? prob : 1 - prob

  return (
    <div className="document-wrapper">
      
      <header className="doc-header">
        <div className="doc-title-block">
          <h1>MED.DIAGNOSTIC</h1>
          <p>Algorithmic Cardiovascular Risk Assessment Model</p>
        </div>
        <div className="doc-meta">
          <span>REPORT ID: {Math.floor(Math.random() * 1000000)}</span>
          <span>SYSTEM: STANDBY</span>
        </div>
      </header>

      <section className="doc-controls">
        <div className="input-grid">
          <div className="input-block">
            <label>Age (Years)</label>
            <input type="number" name="age" className="data-input" value={formData.age} onChange={handleInputChange} />
          </div>
          <div className="input-block">
            <label>Gender</label>
            <select name="gender" className="data-input" value={formData.gender} onChange={handleInputChange}>
              <option value={1}>Female</option>
              <option value={2}>Male</option>
            </select>
          </div>
          <div className="input-block">
            <label>Height (cm)</label>
            <input type="number" name="height" className="data-input" value={formData.height} onChange={handleInputChange} />
          </div>
          <div className="input-block">
            <label>Weight (kg)</label>
            <input type="number" name="weight" className="data-input" value={formData.weight} onChange={handleInputChange} />
          </div>
          <div className="input-block">
            <label>Systolic BP</label>
            <input type="number" name="ap_hi" className="data-input" value={formData.ap_hi} onChange={handleInputChange} />
          </div>
          <div className="input-block">
            <label>Diastolic BP</label>
            <input type="number" name="ap_lo" className="data-input" value={formData.ap_lo} onChange={handleInputChange} />
          </div>
          <div className="input-block">
            <label>Cholesterol</label>
            <select name="cholesterol" className="data-input" value={formData.cholesterol} onChange={handleInputChange}>
              <option value={1}>Normal</option>
              <option value={2}>Above Normal</option>
              <option value={3}>Well Above Normal</option>
            </select>
          </div>
          <div className="input-block">
            <label>Glucose</label>
            <select name="gluc" className="data-input" value={formData.gluc} onChange={handleInputChange}>
              <option value={1}>Normal</option>
              <option value={2}>Above Normal</option>
              <option value={3}>Well Above Normal</option>
            </select>
          </div>
          <div className="input-block">
            <label>Smoker</label>
            <select name="smoke" className="data-input" value={formData.smoke} onChange={handleInputChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div className="input-block">
            <label>Alcohol Intake</label>
            <select name="alco" className="data-input" value={formData.alco} onChange={handleInputChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div className="input-block">
            <label>Physical Activity</label>
            <select name="active" className="data-input" value={formData.active} onChange={handleInputChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
        </div>
        <div className="action-row">
          <button 
            className="execute-btn" 
            onClick={executeForecastQuery} 
            disabled={isProcessing}
          >
            {isProcessing ? 'COMPUTING...' : 'INITIALIZE DIAGNOSIS'}
          </button>
        </div>
      </section>

      {fetchError && <div className="error-banner">{fetchError}</div>}

      {!forecastData && !isProcessing && (
        <div className="status-msg">
          [ AWAITING PATIENT DATA ENTRY ]
        </div>
      )}

      {isProcessing && (
        <div className="status-msg">
          [ PROCESSING MEDICAL RECORDS... ]
        </div>
      )}

      {forecastData && (
        <main className="doc-content">
          
          <div className="doc-section">
            <div className="section-title">I. Primary Diagnosis & Confidence</div>
            
            <div className="verdict-row">
              <div className="verdict-main">
                <div className={`verdict-text ${isHighRisk ? 'bearish' : 'bullish'}`}>
                  {isHighRisk ? '- HIGH RISK' : '- LOW RISK'}
                </div>
                <div className="verdict-sub">Patient Evaluation Complete</div>
              </div>

              <div className="prob-metrics">
                <div className="prob-line">
                  <div className="prob-header">
                    <span>CONFIDENCE METRIC</span>
                    <span className={isHighRisk ? 'red' : 'green'}>{(riskProb * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prob-track">
                    <div className={`prob-fill ${isHighRisk ? 'red' : 'green'}`} style={{ width: `${riskProb * 100}%` }}></div>
                  </div>
                </div>
                <div className="prob-line">
                  <div className="prob-header">
                    <span>SYSTEM MESSAGE</span>
                  </div>
                  <div className="verdict-sub" style={{marginTop: 0}}>{forecastData.message}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="doc-section">
            <div className="section-title">II. Patient Vitals Snapshot</div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Age / Gender</th>
                  <th>Height / Weight</th>
                  <th>Blood Pressure</th>
                  <th>Lifestyle Risks</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>{formData.age} yrs / {formData.gender === 1 ? 'F' : 'M'}</td>
                  <td>{formData.height} cm / {formData.weight} kg</td>
                  <td>{formData.ap_hi} / {formData.ap_lo}</td>
                  <td>
                    Smoker: {formData.smoke === 1 ? 'Y' : 'N'} | Alco: {formData.alco === 1 ? 'Y' : 'N'}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

        </main>
      )}

    </div>
  )
}
