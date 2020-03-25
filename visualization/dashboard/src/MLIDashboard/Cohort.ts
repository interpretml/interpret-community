import { IFilter, FilterMethods } from "./Interfaces/IFilter";
import { JointDataset } from "./JointDataset";

export class Cohort {
    public static CohortKey: "Cohort";
    private static _cohortIndex: number = 0;

    private readonly cohortIndex: number;
    private mutateCount: number = 0;
    private _filteredData: Array<{[key: string]: number}>;
    private _cachedAverageImportance: number[];
    private currentSortKey: string | undefined;
    private currentSortReversed: boolean = false;
    constructor(public name: string, private jointDataset: JointDataset, public filters: IFilter[] = []) {
        this.cohortIndex = Cohort._cohortIndex;
        Cohort._cohortIndex += 1;
        this.applyFilters();
    }

    public updateFilter(filter: IFilter, index?: number): void {
        if (index === undefined) {
            index = this.filters.length;
        } 

        this.filters[index] = filter;
        this.applyFilters();
    }

    // An id to track if a change requireing rerender has occured.
    public getCohortID(): number {
        return this.cohortIndex;
    }

    public deleteFilter(index: number): void {
        this.filters.splice(index, 1);
        this.applyFilters();
    }

    public getRow(index: number): {[key: string]: number} {
        return {...this.jointDataset.dataDict[index]}
    }

    public sort(columnName: string = JointDataset.IndexLabel, reverse?: boolean): void {
        if (this.currentSortKey !== columnName) {
            this._filteredData.sort((a, b) => {
                return a[columnName] - b[columnName];
            });
            this.currentSortKey = columnName;
            this.currentSortReversed = false;
        }
        if (this.currentSortReversed !== reverse) {
            this._filteredData.reverse();
        }
    }

    // whether to apply bins is a decision made at the ui control level,
    // should not mutate the true dataset. Instead, bin props are preserved
    // and applied when requested.
    // Bin object stores array of upper bounds for each bin, return the index
    // if the bin of the value;
    public unwrap(key: string, applyBin?: boolean): any[] {
        if (applyBin && this.jointDataset.metaDict[key].isCategorical === false) {
            let binVector = this.jointDataset.binDict[key];
            if (binVector === undefined) {
                this.jointDataset.addBin(key);
                binVector = this.jointDataset.binDict[key];
            }
            return this._filteredData.map(row => {
                const rowValue = row[key];
                return binVector.findIndex(upperLimit => upperLimit >= rowValue );
            });
        }
        return this._filteredData.map(row => row[key]);
    }

    public calculateAverageImportance(): number[] {
        if (this._cachedAverageImportance) {
            return this._cachedAverageImportance;
        }
        var dict = this._filteredData;
        var result = new Array(this.jointDataset.localExplanationFeatureCount).fill(0);
        dict.forEach(row => {
            for (let i=0; i < this.jointDataset.localExplanationFeatureCount; i++) {
                result[i] += Math.abs(row[JointDataset.ReducedLocalImportanceRoot + i.toString()]);
            }
        });
        this._cachedAverageImportance = result.map(val => val / dict.length);
        return this._cachedAverageImportance;
    }

    private applyFilters(): void {
        this._cachedAverageImportance = undefined;
        this.mutateCount += 1;
        this._filteredData = this.jointDataset.dataDict.filter(row => 
            this.filters.every(filter => {
                const rowVal = row[filter.column];
                switch(filter.method){
                    case FilterMethods.equal:
                        return rowVal === filter.arg;
                    case FilterMethods.greaterThan:
                        return rowVal > filter.arg;
                    case FilterMethods.lessThan:
                        return rowVal < filter.arg;
                    case FilterMethods.includes:
                        return (filter.arg as number[]).includes(rowVal);
                }
            })
        );
    }
}