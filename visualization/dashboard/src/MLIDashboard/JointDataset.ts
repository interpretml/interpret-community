import { IExplanationModelMetadata, ModelTypes } from "./IExplanationContext";
import { INumericRange, ICategoricalRange } from "mlchartlib";
import { localization } from "../Localization/localization";

export interface IJointDatasetArgs {
    dataset?: any[][];
    predictedY?: number[];
    trueY?: number[];
    metadata: IExplanationModelMetadata;
    dataDict?: Array<{[key: string]: number}>;
}

export interface IJointMeta {
    label: string;
    abbridgedLabel: string;
    isCategorical: boolean;
    sortedCategoricalValues?: string[];
    featureRange?: INumericRange;
}

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
    public metaDict: {[key: string]: IJointMeta} = {};
    private _dataset: any[][];

    constructor(args: IJointDatasetArgs) {
        if (args.dataDict) {
            this._dataDict = args.dataDict;
            // build the data by applying the filters, unpacking index, and Array.filtering the Data getter
            this._dataset = args.dataset;
        } else {
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