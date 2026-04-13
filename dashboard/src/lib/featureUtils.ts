// dashboard/src/lib/featureUtils.ts

import type { FeatureDefinition } from "@/types/api";

/**
 * Decide whether a numerical feature should be rendered as a slider
 * rather than a free-text number input.
 *
 * Sliders make sense for small bounded integer ranges (e.g. installment
 * commitment 1..4, residence_since 1..4) where every value is meaningful
 * and the user benefits from a visual sense of position. Open-ended or
 * very wide ranges (credit_amount 1..100M) stay as text inputs.
 */
export function shouldUseSlider(feature: FeatureDefinition): boolean {
  if (feature.type !== "numerical") return false;
  if (feature.min === null || feature.max === null) return false;

  const span = feature.max - feature.min;
  // Small discrete ranges: ideal for sliders.
  return span > 0 && span <= 20;
}

/**
 * Pick a sensible step for a slider.
 *
 * Integer-bounded ranges step by 1; everything else uses a granularity
 * of roughly 1% of the span, capped to two decimals.
 */
export function sliderStep(feature: FeatureDefinition): number {
  if (feature.min === null || feature.max === null) return 1;
  const span = feature.max - feature.min;
  if (Number.isInteger(feature.min) && Number.isInteger(feature.max)) return 1;
  return Math.max(0.01, Math.round((span / 100) * 100) / 100);
}
