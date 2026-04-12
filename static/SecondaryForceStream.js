/* SecondaryForceStream.js
 * Simulated secondary-force generator (chaos coefficient C_s).
 * Models external volatility using bounded random walk + periodic spikes.
 */

(function attachSecondaryForceStream(global) {
  function clip(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function makeRng(seedStart) {
    let seed = Number(seedStart || Date.now() % 10000);
    return function rand() {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    };
  }

  function create(options = {}) {
    const rand = makeRng(options.seed);
    const listeners = new Set();
    let coefficient = clip(Number(options.baseCoefficient ?? options.start ?? 0.1), 0, 1);
    const volatility = clip(Number(options.volatility ?? 0.08), 0.01, 0.5);
    const spikeChance = clip(Number(options.spikeChance ?? 0.08), 0.0, 0.8);
    const spikeStrength = clip(Number(options.spikeStrength ?? 0.45), 0.05, 1.0);
    const tickMs = Math.max(2000, Number(options.tickMs || 8000));
    let lastSignal = "seed";

    function emit(sample) {
      listeners.forEach((fn) => {
        try {
          fn(sample);
        } catch (_e) {}
      });
    }

    function sampleFromStream() {
      const drift = (rand() - 0.5) * volatility;
      let next = coefficient + drift;
      let signal = "ambient-drift";
      if (rand() < spikeChance) {
        next += spikeStrength * (0.55 + rand() * 0.45);
        signal = "volatility-spike";
      }
      coefficient = clip(next, 0, 1);
      lastSignal = signal;
      const sample = {
        coefficient: Number(coefficient.toFixed(4)),
        signal,
        sampled_at: new Date().toISOString(),
      };
      emit(sample);
      return sample;
    }

    const timer = global.setInterval(sampleFromStream, tickMs);

    return {
      currentCoefficient() {
        return Number(coefficient.toFixed(4));
      },
      currentSignal() {
        return String(lastSignal || "ambient-drift");
      },
      setCoefficient(value, meta = {}) {
        coefficient = clip(Number(value || 0), 0, 1);
        lastSignal = String(meta.source || "manual");
        const sample = {
          coefficient: Number(coefficient.toFixed(4)),
          signal: lastSignal,
          sampled_at: new Date().toISOString(),
        };
        emit(sample);
      },
      subscribe(listener) {
        if (typeof listener !== "function") return () => {};
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      stop() {
        global.clearInterval(timer);
      },
      sampleNow() {
        return sampleFromStream();
      },
    };
  }

  global.SecondaryForceStream = { create };
})(window);
