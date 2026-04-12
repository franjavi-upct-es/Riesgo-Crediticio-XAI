// dashboard/src/hooks/useDatasetContext.ts

import { createContext, useContext } from "react";

interface DatasetContextValue {
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string) => void;
}

export const DatasetContext = createContext<DatasetContextValue>({
  activeDatasetId: null,
  setActiveDatasetId: () => {},
});

export function useActiveDataset() {
  return useContext(DatasetContext);
}
