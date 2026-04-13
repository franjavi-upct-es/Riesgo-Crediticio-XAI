// dashboard/src/components/prediction/PredictionForm.tsx

import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Dices, Loader2, Send } from "lucide-react";
import Card from "@/components/ui/Card";
import { Button } from "@/components/ui/button";
import FieldRenderer from "@/components/prediction/FieldRenderer";
import { useRandomSample } from "@/hooks/useApi";
import {
  buildDefaultValues,
  buildPredictionSchema,
  type PredictionFormValues,
} from "@/lib/buildPredictionSchema";
import type { DatasetSchemaResponse, PredictionPayload } from "@/types/api";

interface Props {
  schema: DatasetSchemaResponse;
  datasetId: string;
  isSubmitting: boolean;
  submitError?: string | null;
  onSubmit: (payload: PredictionPayload) => void;
}

export default function PredictionForm({
  schema,
  datasetId,
  isSubmitting,
  submitError,
  onSubmit,
}: Props) {
  const validator = buildPredictionSchema(schema);

  const {
    control,
    handleSubmit,
    reset,
    formState: { errors, isValid },
  } = useForm<PredictionFormValues>({
    resolver: zodResolver(validator),
    defaultValues: buildDefaultValues(schema),
    mode: "onBlur",
  });

  // Reset when the dataset (and therefore the schema) changes.
  useEffect(() => {
    reset(buildDefaultValues(schema));
  }, [schema, reset]);

  const randomMutation = useRandomSample(datasetId);

  async function handleRandomize() {
    const resp = await randomMutation.mutateAsync();
    // The API may return numeric strings or numbers depending on the source;
    // normalise here so RHF holds correctly typed values.
    const normalised: PredictionFormValues = {};
    for (const feature of schema.features) {
      const raw = resp.sample[feature.name];
      normalised[feature.name] =
        feature.type === "numerical" ? Number(raw) : String(raw);
    }
    reset(normalised);
  }

  return (
    <Card className="xl:col-span-2" title="Applicant data">
      <form
        onSubmit={handleSubmit((values) => onSubmit(values as PredictionPayload))}
        noValidate
      >
        <div className="grid gap-4 sm:grid-cols-2">
          {schema.features.map((feature) => (
            <FieldRenderer
              key={feature.name}
              feature={feature}
              control={control}
              errors={errors}
            />
          ))}
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <Button
            type="submit"
            disabled={isSubmitting || (!isValid && Object.keys(errors).length > 0)}
            className="min-w-[120px]"
          >
            {isSubmitting ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <Send size={15} />
            )}
            {isSubmitting ? "Predicting…" : "Predict"}
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={handleRandomize}
            disabled={randomMutation.isPending}
          >
            <Dices size={15} />
            {randomMutation.isPending ? "Sampling…" : "Random"}
          </Button>
        </div>

        {submitError && (
          <p className="mt-3 text-sm font-medium text-destructive">
            {submitError}
          </p>
        )}
      </form>
    </Card>
  );
}
