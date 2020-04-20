

export interface IGlobalSeries {
    unsortedAggregateY: number[];
    // feature x row, given how lookup is done
    unsortedIndividualY?: number[][];
    unsortedFeatureValues?: number[];
    name: string;
    index: number;
}