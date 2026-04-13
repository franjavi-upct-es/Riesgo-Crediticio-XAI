// dashboard/src/components/prediction/FieldRenderer.tsx

import { Controller, type Control, type FieldErrors } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { sliderStep, shouldUseSlider } from "@/lib/featureUtils";
import type { PredictionFormValues } from "@/lib/buildPredictionSchema";
import type { FeatureDefinition } from "@/types/api";

interface Props {
  feature: FeatureDefinition;
  control: Control<PredictionFormValues>;
  errors: FieldErrors<PredictionFormValues>;
}

export default function FieldRenderer({ feature, control, errors }: Props) {
  const error = errors[feature.name];
  const errorMessage = error?.message as string | undefined;
  const labelText = feature.description || feature.name;
  const fieldId = `field-${feature.name}`;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor={fieldId} className="flex items-center gap-1.5">
          <span className="text-foreground/80">{labelText}</span>
          {feature.protected && (
            <span className="rounded bg-amber-100 px-1 text-[9px] font-bold text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
              Protected
            </span>
          )}
        </Label>
        {feature.type === "numerical" &&
          shouldUseSlider(feature) &&
          feature.min !== null &&
          feature.max !== null && (
            <span className="font-mono text-[10px] text-muted-foreground">
              {feature.min}–{feature.max}
            </span>
          )}
      </div>

      <Controller
        name={feature.name}
        control={control}
        render={({ field, fieldState }) => {
          const invalid = fieldState.invalid;

          // --- Categorical → Select ---
          if (feature.type === "categorical" && feature.options.length > 0) {
            return (
              <Select
                value={(field.value as string) ?? ""}
                onValueChange={field.onChange}
              >
                <SelectTrigger
                  id={fieldId}
                  aria-invalid={invalid}
                  onBlur={field.onBlur}
                >
                  <SelectValue placeholder="Choose an option…" />
                </SelectTrigger>
                <SelectContent>
                  {feature.options.map((opt) => (
                    <SelectItem key={opt} value={opt}>
                      {opt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            );
          }

          // --- Numerical (small range) → Slider with live value readout ---
          if (feature.type === "numerical" && shouldUseSlider(feature)) {
            const numericValue =
              typeof field.value === "number"
                ? field.value
                : Number(field.value ?? feature.min ?? 0);
            return (
              <div className="flex items-center gap-3 pt-1">
                <Slider
                  id={fieldId}
                  min={feature.min ?? 0}
                  max={feature.max ?? 100}
                  step={sliderStep(feature)}
                  value={[numericValue]}
                  onValueChange={(v) => field.onChange(v[0])}
                  onBlur={field.onBlur}
                  aria-invalid={invalid}
                  className="flex-1"
                />
                <span className="w-10 rounded bg-muted px-2 py-1 text-center font-mono text-xs text-foreground">
                  {numericValue}
                </span>
              </div>
            );
          }

          // --- Numerical (open / wide range) → Input ---
          return (
            <Input
              id={fieldId}
              type="number"
              inputMode="decimal"
              value={
                field.value === undefined || field.value === null
                  ? ""
                  : String(field.value)
              }
              min={feature.min ?? undefined}
              max={feature.max ?? undefined}
              onChange={(e) => {
                const raw = e.target.value;
                field.onChange(raw === "" ? "" : Number(raw));
              }}
              onBlur={field.onBlur}
              aria-invalid={invalid}
              className="font-mono"
            />
          );
        }}
      />

      {errorMessage && (
        <p className="text-[11px] font-medium text-destructive">
          {errorMessage}
        </p>
      )}
    </div>
  );
}
