import { IExplanationModelMetadata, ModelTypes } from "./IExplanationContext";
import { INumericRange, ICategoricalRange } from "mlchartlib";
import { localization } from "../Localization/localization";
import { IFilter, FilterMethods } from "./Interfaces/IFilter";

export interface IJointDatasetArgs {
    dataset?: any[][];
    predictedY?: number[];
    trueY?: number[];
    metadata: IExplanationModelMetadata;
}

export interface IJointMeta {
    label: string;
    abbridgedLabel: string;
    isCategorical: boolean;
    sortedCategoricalValues?: string[];
    featureRange?: INumericRange;
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
                            sortedCategoricalValues: sortedUnique
                        }
                    } else {
                        this._dataDict[index][key] = val;
                        this.metaDict[key] = {
                            label: args.metadata.featureNames[colIndex],
                            abbridgedLabel: args.metadata.featureNamesAbridged[colIndex],
                            isCategorical: false,
                            featureRange: args.metadata.featureRanges[colIndex] as INumericRange
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
                sortedCategoricalValues: args.metadata.modelType !== ModelTypes.regression ? args.metadata.classNames : undefined
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

    public unwrap(key: string): any[] {
        return this._dataDict.map(row => row[key]);
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
            isCategorical: false
        };
    }
}