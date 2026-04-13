// dashboard/src/lib/buildPredictionSchema.ts

import { z, type ZodTypeAny } from "zod";
import type { DatasetSchemaResponse, FeatureDefinition } from "@/types/api";

/**
 * The shape of values held by react-hook-form for a prediction form.
 *
 * Numerical fields are kept as `number` (RHF `valueAsNumber`) and categorical
 * as `string`. We use `unknown` indexing because the keys are dataset-driven
 * and only known at runtime.
 */
export type PredictionFormValues = Record<string, number | string>;

/**
 * The shape of the payload we send to the API: identical to the form state
 * after Zod validation has narrowed each field to its expected type.
 */
export type PredictionPayload = Record<string, number | string>;

/**
 * Build a Zod schema for one feature definition coming from the API.
 *
 * Numerical features honour min/max bounds (when present) so the form
 * mirrors the server-side Pydantic constraints exactly.
 * Categorical features are restricted to the declared option list.
 */
function buildFieldSchema(feature: FeatureDefinition): ZodTypeAny {
  if (feature.type === "numerical") {
    let schema = z.coerce
      .number({
        invalid_type_error: `${feature.description || feature.name} must be a number`,
      })
      .finite();

    if (feature.min !== null && feature.min !== undefined) {
      schema = schema.gte(
        feature.min,
        `Must be ≥ ${feature.min}`,
      );
    }
    if (feature.max !== null && feature.max !== undefined) {
      schema = schema.lte(
        feature.max,
        `Must be ≤ ${feature.max}`,
      );
    }
    return schema;
  }

  // Categorical
  if (feature.options.length > 0) {
    return z
      .string({
        required_error: "Please choose an option",
        invalid_type_error: "Please choose an option",
      })
      .min(1, "Please choose an option")
      .refine((v) => feature.options.includes(v), {
        message: `Must be one of: ${feature.options.join(", ")}`,
      });
  }

  return z.string().min(1, "Required");
}

/**
 * Build a Zod object schema for an entire dataset, keyed by feature name.
 *
 * The resulting schema is consumed by `zodResolver` in react-hook-form so
 * that validation errors surface inline before any API call is made.
 */
export function buildPredictionSchema(
  schema: DatasetSchemaResponse,
): z.ZodType<PredictionPayload> {
  const shape: Record<string, ZodTypeAny> = {};
  for (const feature of schema.features) {
    shape[feature.name] = buildFieldSchema(feature);
  }
  return z.object(shape) as unknown as z.ZodType<PredictionPayload>;
}

/**
 * Build the initial default values object for react-hook-form based on
 * the schema's `default_value` per feature.
 */
export function buildDefaultValues(
  schema: DatasetSchemaResponse,
): PredictionFormValues {
  const defaults: PredictionFormValues = {};
  for (const feature of schema.features) {
    defaults[feature.name] =
      feature.type === "numerical"
        ? Number(feature.default_value)
        : String(feature.default_value);
  }
  return defaults;
}
