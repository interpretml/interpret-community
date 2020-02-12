import { IExplanationModelMetadata, ModelTypes } from "./IExplanationContext";
import { INumericRange, ICategoricalRange, RangeTypes } from "mlchartlib";
import { localization } from "../Localization/localization";
import { IFilter, FilterMethods } from "./Interfaces/IFilter";
import _ from "lodash";

export interface IJointDatasetArgs {
    dataset?: any[][];
    predictedY?: number[];
    trueY?: number[];
    metadata: IExplanationModelMetadata;
}

export enum ColumnCategories {
    outcome = 'outcome',
    dataset = 'dataset',
    explanation = 'explanation'
}

// The object that will store user-facing strings and associated metadata
// It stores the categorical labels for any numeric bins
export interface IJointMeta {
    label: string;
    abbridgedLabel: string;
    isCategorical: boolean;
    sortedCategoricalValues?: string[];
    featureRange?: INumericRange;
    category: ColumnCategories
}

// this is the single source for data, it should hold all raw data and be how data for presentation is
// accessed. It shall apply filters to the raw table and persist the filtered table for presenting to 
// charts and dashboards. It shall sort indexed rows by a selected column. 
// Filtering will create a copy of the underlying dataset and sorting should be in place on this copy.
// projection should create a copy of values.
// 
export class JointDataset {
    public static readonly IndexLabel = "Index";
    public static readonly DataLabelTemplate = "Data[{0}]";
    public static readonly PredictedYLabel = "PredictedY";
    public static readonly TrueYLabel = "TrueY";
    public static readonly DitherLabel = "Dither";

    public hasDataset: boolean = false;
    public hasPredictedY: boolean = false;
    public hasTrueY: boolean = false;

    private _dataDict: Array<{[key: string]: any}>;
    private _filteredData: Array<{[key: string]: any}>;
    private _binDict: {[key: string]: number[]} = {};
    public metaDict: {[key: string]: IJointMeta} = {};

    constructor(args: IJointDatasetArgs) {
        if (args.dataset) {
            this.initializeDataDictIfNeeded(args.dataset);
            args.dataset.forEach((row, index) => {
                row.forEach((val, colIndex) => {
                    const key = JointDataset.DataLabelTemplate.replace("{0}", colIndex.toString());
                    // store the index for categorical values rather than the actual value. Makes dataset uniform numeric and enables dithering
                    if (args.metadata.featureIsCategorical[colIndex]) {
                        const sortedUnique = (args.metadata.featureRanges[colIndex] as ICategoricalRange).uniqueValues.concat().sort();
                        this._dataDict[index][key] = sortedUnique.indexOf(val);
                        this.metaDict[key] = {
                            label: args.metadata.featureNames[colIndex],
                            abbridgedLabel: args.metadata.featureNamesAbridged[colIndex],
                            isCategorical: true,
                            sortedCategoricalValues: sortedUnique,
                            category: ColumnCategories.dataset
                        }
                    } else {
                        this._dataDict[index][key] = val;
                        this.metaDict[key] = {
                            label: args.metadata.featureNames[colIndex],
                            abbridgedLabel: args.metadata.featureNamesAbridged[colIndex],
                            isCategorical: false,
                            featureRange: args.metadata.featureRanges[colIndex] as INumericRange,
                            category: ColumnCategories.dataset
                        }
                    }
                });
            });
            this.hasDataset = true;
        }
        if (args.predictedY) {
            this.initializeDataDictIfNeeded(args.predictedY);
            args.predictedY.forEach((val, index) => {
                this._dataDict[index][JointDataset.PredictedYLabel] = val;
            });
            this.metaDict[JointDataset.PredictedYLabel] = {
                label: localization.ExplanationScatter.predictedY,
                abbridgedLabel: localization.ExplanationScatter.predictedY,
                isCategorical: args.metadata.modelType !== ModelTypes.regression,
                sortedCategoricalValues: args.metadata.modelType !== ModelTypes.regression ? args.metadata.classNames : undefined,
                category: ColumnCategories.outcome
                
            };
            if (args.metadata.modelType === ModelTypes.regression) {
                this.metaDict[JointDataset.PredictedYLabel].featureRange = {
                    min: Math.min(...args.predictedY),
                    max: Math.max(...args.predictedY),
                    rangeType: RangeTypes.numeric
                }
            }
            this.hasPredictedY = true;
        }
        if (args.trueY) {
            this.initializeDataDictIfNeeded(args.trueY);
            args.trueY.forEach((val, index) => {
                this._dataDict[index][JointDataset.TrueYLabel] = val;
            });
            this.hasTrueY = true;
        }
        this.applyFilters();
    }

    public applyFilters(filters: IFilter[] = []): void {
        this._filteredData = this._dataDict.filter(row => 
            filters.every(filter => {
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

    public sort(columnName: string = JointDataset.IndexLabel, reverse?: boolean): void {
        this._filteredData.sort((a, b) => {
            return a[columnName] - b[columnName];
        });
        if (reverse) {
            this._filteredData.reverse();
        }
    }

    public addBin(key: string, binCount?: number): void {
        const meta = this.metaDict[key];
        // use data-dict for undefined binCount (building default bin)
        // use filtered data for user provided binCount
        if (binCount === undefined) {
            if (meta.featureRange.rangeType === RangeTypes.integer) {
                const uniqueValues = _.uniq(this._dataDict.map(row => row[key]));
                binCount = Math.min(5, uniqueValues.length);
            }
            if (binCount === undefined) {
                binCount = 5;
            }
        }
        let delta = meta.featureRange.max - meta.featureRange.min;
        if (delta === 0 || binCount === 0) {
            this._binDict[key] = [meta.featureRange.max];
            meta.sortedCategoricalValues = [`${meta.featureRange.min} - ${meta.featureRange.max}`];
            return;
        }
        // make uniform bins in these cases
        if (meta.featureRange.rangeType === RangeTypes.numeric || delta < (binCount - 1)) {
            const binDelta = delta / binCount;
            const array = new Array(binCount).fill(0).map((unused, index) => {
                return index !== binCount - 1 ?
                    meta.featureRange.min + (binDelta * (1+ index)) :
                    meta.featureRange.max;
            });
            let prevMax = meta.featureRange.min;
            const labelArray = array.map((num) => {
                const label = `${prevMax.toLocaleString(undefined, {maximumSignificantDigits: 3})} - ${num.toLocaleString(undefined, {maximumSignificantDigits: 3})}`;
                prevMax = num;
                return label;
            });
            this._binDict[key] = array;
            meta.sortedCategoricalValues = labelArray;
            return;
        }
        // handle integer case, increment delta since we include the ends as discrete values
        const intDelta = delta / binCount;
        const array = new Array(binCount).fill(0).map((unused, index) => {
            if (index === binCount - 1) {
                return meta.featureRange.max;
            }
            return Math.ceil( meta.featureRange.min - 1 + intDelta * (index + 1));
        });
        let previousVal = meta.featureRange.min;
        const labelArray = array.map((num) => {
            const label = previousVal === num ?
            previousVal.toLocaleString(undefined, {maximumSignificantDigits: 3}) :
                `${previousVal.toLocaleString(undefined, {maximumSignificantDigits: 3})} - ${num.toLocaleString(undefined, {maximumSignificantDigits: 3})}`
            previousVal = num + 1;
            return label;
        });
        this._binDict[key] = array;
        meta.sortedCategoricalValues = labelArray;
    } 

    // whether to apply bins is a decision made at the ui control level,
    // should not mutate the true dataset. Instead, bin props are preserved
    // and applied when requested.
    // Bin object stores array of upper bounds for each bin, return the index
    // if the bin of the value;
    public unwrap(key: string, applyBin?: boolean): any[] {
        if (applyBin && this.metaDict[key].isCategorical === false) {
            let binVector = this._binDict[key];
            if (binVector === undefined) {
                this.addBin(key);
                binVector = this._binDict[key];
            }
            return this._filteredData.map(row => {
                const rowValue = row[key];
                return binVector.findIndex(upperLimit => upperLimit >= rowValue );
            });
        }
        return this._filteredData.map(row => row[key]);
    }

    private initializeDataDictIfNeeded(arr: any[]): void {
        if (arr === undefined) {
            return;
        }
        if (this._dataDict !== undefined) {
            if (this._dataDict.length !== arr.length) {
                throw new Error("Differing length inputs. Ensure data matches explanations and predictions.")
            }
            return;
        }
        this._dataDict = Array.from({length: arr.length} as any).map((unused, index) => {
            const dict = {};
            dict[JointDataset.IndexLabel] = index;
            dict[JointDataset.DitherLabel] = 0.2 * Math.random() - 0.1;
            return dict;
        });
        this.metaDict[JointDataset.IndexLabel] = {
            label: localization.ExplanationScatter.index,
            abbridgedLabel: localization.ExplanationScatter.index,
            isCategorical: false,
            featureRange: {
                rangeType: RangeTypes.integer,
                min: 0,
                max: arr.length - 1
            },
            category: ColumnCategories.outcome
        };
    }
}