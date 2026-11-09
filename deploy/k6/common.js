// Shared helpers of the load scripts: one constant arrival rate per k6 run.
// The Python driver (python -m src.runner.runner_load) runs one k6 process per rate step
// and turns the exported summary into a benchmark record (task "load").

export const TREND_STATS = ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'];

// RATE_RPM: requests per minute (integer), DURATION_S: length of the step.
export function stepOptions(sloP95Ms, maxVus, gracefulStopS) {
  const rate = parseInt(__ENV.RATE_RPM || '6', 10);
  const duration = parseInt(__ENV.DURATION_S || '60', 10);
  return {
    discardResponseBodies: true,
    summaryTrendStats: TREND_STATS,
    scenarios: {
      step: {
        executor: 'constant-arrival-rate',
        rate: rate,
        timeUnit: '1m',
        duration: `${duration}s`,
        // All VUs up front: a VU started mid-test makes k6 drop the iteration that needed it.
        preAllocatedVUs: maxVus,
        maxVUs: maxVus,
        gracefulStop: `${gracefulStopS}s`,
      },
    },
    thresholds: {
      'http_req_duration{expected_response:true}': [`p(95)<${sloP95Ms}`],
      http_req_failed: ['rate<0.01'],
    },
  };
}

export function authHeaders() {
  const key = __ENV.VLLM_API_KEY;
  return key ? { Authorization: `Bearer ${key}` } : {};
}

function value(data, metric, stat) {
  const m = data.metrics[metric];
  return m && m.values[stat] !== undefined ? m.values[stat] : null;
}

// Compact JSON read by runner_load: the SLO is defined once, in the task script.
export function exportSummary(data, sloP95Ms) {
  const ok = 'http_req_duration{expected_response:true}';
  const durationMetric = data.metrics[ok] ? ok : 'http_req_duration';
  const summary = {
    slo_p95_ms: sloP95Ms,
    rate_rpm: parseInt(__ENV.RATE_RPM || '6', 10),
    duration_s: parseInt(__ENV.DURATION_S || '60', 10),
    test_run_duration_ms: data.state.testRunDurationMs,
    requests: value(data, 'http_reqs', 'count'),
    failed_rate: value(data, 'http_req_failed', 'rate'),
    dropped_iterations: value(data, 'dropped_iterations', 'count') || 0,
    vus_max: value(data, 'vus_max', 'max'),
    latency_ms: {},
  };
  for (const stat of TREND_STATS) {
    summary.latency_ms[stat] = value(data, durationMetric, stat);
  }
  const out = {};
  out[__ENV.SUMMARY_FILE || 'k6-summary.json'] = JSON.stringify(summary, null, 2);
  out.stdout = `rate=${summary.rate_rpm}/min requests=${summary.requests} ` +
    `p95=${summary.latency_ms['p(95)']}ms failed=${summary.failed_rate} dropped=${summary.dropped_iterations}\n`;
  return out;
}
