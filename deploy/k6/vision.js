// Load test of the vision server: the fictitious quote page, same request as runner_vision.
// The request body is built by runner_load (PDF rendering + schema) and passed as BODY_FILE.
import http from 'k6/http';
import { check } from 'k6';
import { authHeaders, exportSummary, stepOptions } from './common.js';

const SLO_P95_MS = 30000;     // p95 latency target for one page
const MAX_VUS = 32;           // upper bound of requests in flight
const GRACEFUL_STOP_S = 180;  // let the last pages finish after the step
const REQUEST_TIMEOUT = '300s';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000/v1';
const BODY = open(__ENV.BODY_FILE);

export const options = stepOptions(SLO_P95_MS, MAX_VUS, GRACEFUL_STOP_S);

export default function () {
  const res = http.post(`${BASE_URL}/chat/completions`, BODY, {
    headers: Object.assign({ 'Content-Type': 'application/json' }, authHeaders()),
    timeout: REQUEST_TIMEOUT,
  });
  check(res, { 'status 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  return exportSummary(data, SLO_P95_MS);
}
