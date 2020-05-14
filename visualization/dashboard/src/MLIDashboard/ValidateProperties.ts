import { IExplanationDashboardProps } from "./Interfaces";
import { IExplanationModelMetadata } from "./IExplanationContext";
import { localization } from "../Localization/localization";

export class ValidateProperties {
    private readonly classLength: number;
    private readonly featureLength: number;
    public readonly errorStrings: string[] = [];
    private rowLength: number;
    constructor(private props: IExplanationDashboardProps, modelMetadata: IExplanationModelMetadata) {
        this.classLength = modelMetadata.classNames.length;
        this.featureLength = modelMetadata.featureNames.length;
        this.validateProps();
    }

    // Mutates the passed in props arg, removing any properties that are incompatible.
    private validateProps(): void {
        if (this.props.trueY) {
            this.rowLength = this.props.trueY.length;
        }
        if (this.props.predictedY) {
            this.validatePredictedY();
        }
        if (this.props.probabilityY) {
            this.validatePredictProba();
        }
        if (this.props.testData) {
            this.validateTestData();
        }
        if (this.props.precomputedExplanations && this.props.precomputedExplanations.localFeatureImportance && this.props.precomputedExplanations.localFeatureImportance.scores) {
            this.validateLocalExplanation();
        }
    }

    private static validatePredictedProbabilities(props: IExplanationDashboardProps, modelMetadata: IExplanationModelMetadata, classLength: number, rowLength?: number): string | undefined {
        if (props.probabilityY) {
            if (!Array.isArray(props.probabilityY)) {
                props.probabilityY = undefined;
                if (rowLength !== undefined) {
                    rowLength = 0;
                }
                return localization.formatString(localization.ValidationErrors.predictedProbNotArray, rowLength, classLength);
            }
            const length = props.probabilityY.length;
            if (rowLength === undefined) {
                rowLength = length;
            }
            if (length !== rowLength) {
                return `Inconsistent dimensions. Predicted y has length [${length}], expected [${rowLength}]`;
            }
            if (length === 0) {
                return "Predicted probability input not a non-empty array";
            }
            const cLength = props.probabilityY[0].length;
            if (cLength !== classLength) {
                return `Inconsistent dimensions. Predicted probability has dimensions [${length} x ${cLength}], expected [${rowLength} x ${classLength}]`;
            }
            if (!props.probabilityY.every(row => row.length === classLength)) {
                return `Inconsistent dimensions. Predicted probability has rows of varying length`;
            }
        }
    }

    private validatePredictedY(): void {
        const length = this.props.predictedY.length;
        if (this.rowLength === undefined) {
            this.rowLength = length;
        }
        if (length !== this.rowLength) {
            this.props.predictedY = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.predictedY, length, this.rowLength));
        }
    }

    private validatePredictProba(): void {
        if (!Array.isArray(this.props.probabilityY)) {
            this.props.probabilityY = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.notArray, localization.ValidationErrors.predictedProbability, `${this.rowLength || 0} x ${this.classLength}`));
            return;
        }
        const length = this.props.probabilityY.length;
        if (this.rowLength === undefined) {
            this.rowLength = length;
        }
        if (length !== this.rowLength) {
            this.props.probabilityY = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.predictedProbability, length, this.rowLength))
            return;
        }
        if (length === 0) {
            this.props.probabilityY = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.notNonEmpty, localization.ValidationErrors.predictedProbability))
            return;
        }
        const cLength = this.props.probabilityY[0].length;
        if (cLength !== this.classLength) {
            this.props.probabilityY = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.predictedProbability, `[${length} x ${cLength}]`, `[${this.rowLength} x ${this.classLength}]`));
            return;
        }
        if (!this.props.probabilityY.every(row => row.length === this.classLength)) {
            this.props.probabilityY = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.varyingLength, localization.ValidationErrors.predictedProbability));
            return;
        } 
    }

    private validateTestData(): void {
        if (!Array.isArray(this.props.testData)) {
            this.props.testData = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.notArray, localization.ValidationErrors.evalData, `${this.rowLength || 0} x ${this.classLength}`));
            return;
        }
        const length = this.props.testData.length;
        if (this.rowLength === undefined) {
            this.rowLength = length;
        }
        if (length !== this.rowLength) {
            this.props.testData = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.evalData, length, this.rowLength))
            return;
        }
        if (length === 0) {
            this.props.testData = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.notNonEmpty, localization.ValidationErrors.evalData))
            return;
        }
        const fLength = this.props.testData[0].length;
        if (fLength !== this.featureLength) {
            this.props.testData = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.evalData, `[${length} x ${fLength}]`, `[${this.rowLength} x ${this.featureLength}]`));
            return;
        }
        if (!this.props.testData.every(row => row.length === this.featureLength)) {
            this.props.testData = undefined;
            this.errorStrings.push(localization.formatString(localization.ValidationErrors.varyingLength, localization.ValidationErrors.evalData));
            return;
        }
    }

    private validateLocalExplanation(): void {
        const localExp = this.props.precomputedExplanations.localFeatureImportance.scores;
            if (!Array.isArray(localExp)) {
                this.props.precomputedExplanations.localFeatureImportance = undefined;
                this.errorStrings.push(localization.formatString(localization.ValidationErrors.notArray, localization.ValidationErrors.localFeatureImportance, `${this.rowLength || 0} x ${this.featureLength} x ${this.classLength}`));
                return;
            }
            // explanation will be 2d in case of regression models. 3 for classifier
            let expDim: number = 2;
            if ((localExp as number[][][]).every(dim1 => {
                return dim1.every(dim2 => Array.isArray(dim2));
            })) {
                expDim = 3;
            }
            if (expDim === 3) {
                const cLength = localExp.length;
                if (this.classLength !== cLength) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.localFeatureImportance, cLength, this.classLength))
                    return;
                }
                if (cLength === 0) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.notNonEmpty, localization.ValidationErrors.localFeatureImportance));
                    return;
                }
                const rLength = localExp[0].length;
                if (rLength === 0) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.notNonEmpty, localization.ValidationErrors.localFeatureImportance));
                    return;
                }
                if (this.rowLength === undefined) {
                    this.rowLength = rLength;
                }
                if (rLength !== this.rowLength) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.localFeatureImportance, `${cLength} x ${rLength}`, `${this.classLength} x ${this.rowLength}`));
                    return;
                }
                if (!localExp.every(classArray => classArray.length === this.rowLength)) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.varyingLength, localization.ValidationErrors.localFeatureImportance));
                    return;
                }
                const fLength = (localExp[0][0] as number[]).length;
                if (fLength !== this.featureLength) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.localFeatureImportance, `${cLength} x ${rLength} x ${fLength}`, `${this.classLength} x ${this.rowLength} x ${this.featureLength}`));
                    return;
                }
                if (!localExp.every(classArray => classArray.every(rowArray => rowArray.length === this.featureLength))) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.varyingLength, localization.ValidationErrors.localFeatureImportance));
                    return;
                }
            } else {
                const length = localExp.length;
                if (this.rowLength === undefined) {
                    this.rowLength = length;
                }
                if (length !== this.rowLength) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.localFeatureImportance, `${length}`, `${this.rowLength}`));
                    return;
                }
                if (length === 0) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.notNonEmpty, localization.ValidationErrors.localFeatureImportance));
                    return;
                }
                const fLength = (localExp[0] as number[]).length;
                if (fLength !== this.featureLength) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.inconsistentDimensions, localization.ValidationErrors.localFeatureImportance, `${length} x ${fLength}`, `${this.rowLength} x ${this.featureLength}`));
                    return;
                }
                if (!localExp.every(rowArray => rowArray.length === this.featureLength)) {
                    this.props.precomputedExplanations.localFeatureImportance = undefined;
                    this.errorStrings.push(localization.formatString(localization.ValidationErrors.varyingLength, localization.ValidationErrors.localFeatureImportance));
                    return;
                }
            }
    }
}