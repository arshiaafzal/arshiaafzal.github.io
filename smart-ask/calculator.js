export const presets = Object.freeze({
  mechanical: { baseline: 25, sonnetShare: 78, handoffs: 1, overhead: 5 },
  mixed: { baseline: 25, sonnetShare: 58, handoffs: 2, overhead: 6 },
  architecture: { baseline: 25, sonnetShare: 25, handoffs: 3, overhead: 7 },
});

/**
 * Estimate routed spend from an observed Opus-only baseline.
 * Sonnet work is modeled at 1/5 of the equivalent Opus workload. Handoffs add
 * a user-controlled 0.6% of baseline each; classifier overhead is explicit.
 */
export function calculateEstimate({ baseline, sonnetShare, handoffs, overhead }) {
  const safeBaseline = Math.max(0, Number(baseline) || 0);
  const share = Math.min(100, Math.max(0, Number(sonnetShare) || 0)) / 100;
  const safeHandoffs = Math.max(0, Number(handoffs) || 0);
  const overheadRate = Math.max(0, Number(overhead) || 0) / 100;
  const opusWork = safeBaseline * (1 - share);
  const sonnetWork = safeBaseline * share * 0.2;
  const classifierCost = safeBaseline * overheadRate;
  const handoffCost = safeBaseline * safeHandoffs * 0.006;
  const routedCost = opusWork + sonnetWork + classifierCost + handoffCost;
  const saving = safeBaseline - routedCost;
  const savingPercent = safeBaseline ? (saving / safeBaseline) * 100 : 0;
  return { opusWork, sonnetWork, classifierCost, handoffCost, routedCost, saving, savingPercent };
}
