import axios from 'axios'

// Vite exposes env vars via import.meta.env
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function predict(text){
  const resp = await axios.post(`${API_BASE}/predict`, { text })
  return resp.data
}

export async function transform(text){
  const resp = await axios.get(`${API_BASE}/transform`, { params: { text } })
  return resp.data
}

export async function train(payload){
  const resp = await axios.post(`${API_BASE}/train`, payload)
  return resp.data
}

export async function artifacts(){
  const resp = await axios.get(`${API_BASE}/artifacts`)
  return resp.data
}
