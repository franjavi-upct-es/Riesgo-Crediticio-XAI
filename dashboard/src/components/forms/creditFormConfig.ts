// dashboard/src/components/forms/creditFormConfig.ts

export interface FieldDef {
  name: string;
  label: string;
  type: "text" | "number" | "select";
  options?: string[];
  defaultValue: string | number;
  min?: number;
  max?: number;
}

export const CREDIT_FIELDS: FieldDef[] = [
  {
    name: "checking_status",
    label: "Checking account status",
    type: "select",
    options: ["<0", "0<=X<200", ">=200", "no_checking"],
    defaultValue: "no_checking",
  },
  {
    name: "duration",
    label: "Duration (months)",
    type: "number",
    defaultValue: 12,
    min: 1,
    max: 120,
  },
  {
    name: "credit_history",
    label: "Credit history",
    type: "select",
    options: [
      "no credits/all paid",
      "all paid",
      "existing paid",
      "delayed previously",
      "critical/other",
    ],
    defaultValue: "existing paid",
  },
  {
    name: "purpose",
    label: "Purpose",
    type: "select",
    options: [
      "new car",
      "used car",
      "furniture/equipment",
      "radio/television",
      "domestic appliance",
      "repairs",
      "education",
      "retraining",
      "business",
      "other",
    ],
    defaultValue: "radio/television",
  },
  {
    name: "credit_amount",
    label: "Credit amount",
    type: "number",
    defaultValue: 5000,
    min: 1,
    max: 100000000,
  },
  {
    name: "savings_status",
    label: "Savings status",
    type: "select",
    options: ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no_savings"],
    defaultValue: "no_savings",
  },
  {
    name: "employment",
    label: "Employment",
    type: "select",
    options: ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    defaultValue: "4<=X<7",
  },
  {
    name: "installment_commitment",
    label: "Installment rate (1-4)",
    type: "number",
    defaultValue: 3,
    min: 1,
    max: 4,
  },
  {
    name: "personal_status",
    label: "Personal status",
    type: "select",
    options: [
      "male div/sep",
      "female div/dep/mar",
      "male single",
      "male mar/wid",
    ],
    defaultValue: "male single",
  },
  {
    name: "other_parties",
    label: "Other parties",
    type: "select",
    options: ["none", "co-applicant", "guarantor"],
    defaultValue: "none",
  },
  {
    name: "residence_since",
    label: "Residence since (1-4)",
    type: "number",
    defaultValue: 4,
    min: 1,
    max: 4,
  },
  {
    name: "property_magnitude",
    label: "Property type",
    type: "select",
    options: ["real estate", "life insurance", "car", "no known property"],
    defaultValue: "real estate",
  },
  {
    name: "age",
    label: "Age",
    type: "number",
    defaultValue: 35,
    min: 18,
    max: 120,
  },
  {
    name: "other_payment_plans",
    label: "Other payment plans",
    type: "select",
    options: ["bank", "stores", "none"],
    defaultValue: "none",
  },
  {
    name: "housing",
    label: "Housing",
    type: "select",
    options: ["rent", "own", "for free"],
    defaultValue: "own",
  },
  {
    name: "existing_credits",
    label: "Existing credits (1-10)",
    type: "number",
    defaultValue: 1,
    min: 1,
    max: 10,
  },
  {
    name: "job",
    label: "Job",
    type: "select",
    options: [
      "unemp/unskilled non res",
      "unskilled resident",
      "skilled",
      "high qualif/self emp/mgmt",
    ],
    defaultValue: "skilled",
  },
  {
    name: "num_dependents",
    label: "Dependents (1-10)",
    type: "number",
    defaultValue: 1,
    min: 1,
    max: 10,
  },
  {
    name: "own_telephone",
    label: "Telephone",
    type: "select",
    options: ["none", "yes"],
    defaultValue: "yes",
  },
  {
    name: "foreign_worker",
    label: "Foreign worker",
    type: "select",
    options: ["yes", "no"],
    defaultValue: "no",
  },
];

export function buildDefaults(): Record<string, string | number> {
  const defaults: Record<string, string | number> = {};
  for (const f of CREDIT_FIELDS) {
    defaults[f.name] = f.defaultValue;
  }
  return defaults;
}

export function randomize(): Record<string, string | number> {
  const result: Record<string, string | number> = {};
  for (const f of CREDIT_FIELDS) {
    if (f.type === "select" && f.options) {
      result[f.name] = f.options[Math.floor(Math.random() * f.options.length)];
    } else if (f.type === "number") {
      const min = f.min ?? 1;
      const max = f.max ?? 100;
      result[f.name] = Math.floor(Math.random() * (max - min + 1)) + min;
    } else {
      result[f.name] = f.defaultValue;
    }
  }
  return result;
}
