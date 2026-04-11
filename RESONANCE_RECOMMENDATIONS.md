## Resonance Recommendations (Focused)

Use this as the strict roadmap for keeping Myco aligned with Project Resonance.

### Keep

- Three-layer flow only: `raw_data -> agent_metabolism -> crystallized_substrate`.
- Thermal selection loop with 24h dissolution policy.
- Trace fossil record in `.secrets/agent_trace_log.jsonl`.
- Autonomy Slider (Awe Metric) and 3-D layer visualization.
- Human approval gate for high-impact or irreversible actions.

### Remove or Deprioritize

- UI copy or metrics that do not map to gas/liquid/bedrock behavior.
- Non-Resonance surfaces in main navigation for default operator workflows.
- Any synthetic/test dwellers shipped as default runtime state.
- Features that cannot be traced to entropy reduction or thermodynamic utility.

### Next Recommended Additions

1. **Resonance-only operator mode**
   - Add a toggle that hides non-Resonance nav routes and dashboards by default.
2. **Bedrock admission policy**
   - Enforce `>=3 spikes` and a minimum efficiency score at API level before sedimentation.
3. **Entropy budget guardrail**
   - Reject mutations when rolling entropy trend worsens for N consecutive cycles.
4. **Cycle replay**
   - Add `/api/resonance/cycles/{id}` for deterministic replay of one thermal cycle from trace + memory.
5. **Automated cleanup**
   - Run a scheduled job to archive stale `.noise` artifacts and compact trace/memory indexes.
