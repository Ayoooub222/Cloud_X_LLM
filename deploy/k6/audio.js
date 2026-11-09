// Load test of the audio server: one transcription of AUDIO_FILE per iteration.
import http from 'k6/http';
import { check } from 'k6';
import { authHeaders, exportSummary, stepOptions } from './common.js';

const SLO_P95_MS = 5000;      // p95 latency target for a 66 s clip
const MAX_VUS = 64;
const GRACEFUL_STOP_S = 60;
const REQUEST_TIMEOUT = '120s';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8001';
const MODEL = __ENV.AUDIO_MODEL || 'openai/whisper-large-v3-turbo';
const AUDIO = open(__ENV.AUDIO_FILE, 'b');
const AUDIO_NAME = __ENV.AUDIO_FILE.split('/').pop();

export const options = stepOptions(SLO_P95_MS, MAX_VUS, GRACEFUL_STOP_S);

export default function () {
  const form = {
    file: http.file(AUDIO, AUDIO_NAME),
    model: MODEL,
    language: 'en',
    temperature: '0',
    response_format: 'json',
  };
  const res = http.post(`${BASE_URL}/v1/audio/transcriptions`, form, {
    headers: authHeaders(),
    timeout: REQUEST_TIMEOUT,
  });
  check(res, { 'status 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  return exportSummary(data, SLO_P95_MS);
}
