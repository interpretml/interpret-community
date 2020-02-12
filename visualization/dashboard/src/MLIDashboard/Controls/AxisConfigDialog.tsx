import React from "react";
import { JointDataset, ColumnCategories, IJointMeta } from "../JointDataset";
import { ISelectorConfig } from "./ChartWithControls";
import { Target, Callout } from "office-ui-fabric-react/lib/Callout";
import { Checkbox } from "office-ui-fabric-react/lib/Checkbox";
import { DetailsList, SelectionMode, Selection } from "office-ui-fabric-react/lib/DetailsList";
import { getTheme, mergeStyleSets } from "@uifabric/styling";
import { DefaultButton } from "office-ui-fabric-react/lib/Button";
import _ from "lodash";
import { SpinButton } from "office-ui-fabric-react/lib/SpinButton";
import { localization } from "../../Localization/localization";

const theme = getTheme();
const styles = mergeStyleSets({
    wrapper: {
        minHeight: "300px",
        width: "400px",
        display: "block"
    },
    leftHalf: {
        display: "inline-flex",
        width: "50%"
    },
    rightHalf: {
        display: "inline-flex",
        width: "50%"
    }
});

export interface IAxisConfigProps {
    jointDataset: JointDataset;
    orderedGroupTitles: ColumnCategories[];
    selectedColumn: ISelectorConfig;
    canBin: boolean;
    mustBin: boolean;
    canDither: boolean;
    onAccept: (newConfig: ISelectorConfig) => void;
    onCancel: () => void;
    target: Target;
}

export interface IAxisConfigState {
    selectedColumn: ISelectorConfig;
    items: {key: string, title: string}[][];
    binCount?: number;
}

export class AxisConfigDialog extends React.PureComponent<IAxisConfigProps, IAxisConfigState> {
    private _isInitialized = false;

    private _allItems = this.props.orderedGroupTitles.map(groupLabel => {
        return Object.entries(this.props.jointDataset.metaDict)
        .filter(([key, val]) => {return val.category === groupLabel})
        .map(([key, val]) => {
            return {key, title: val.abbridgedLabel}
        })
    })

    private _selection: Selection;
    private readonly MIN_HIST_COLS = 2;
    private readonly MAX_HIST_COLS = 40;


    constructor(props: IAxisConfigProps) {
        super(props);
        this.state = {
            selectedColumn: _.cloneDeep(this.props.selectedColumn),
            items: this._allItems,
            binCount: this._getBinCountForProperty(this.props.selectedColumn.property)
        };
        this._selection = new Selection({
            selectionMode: SelectionMode.single,
            onSelectionChanged: this._setSelection
          });
        this._selection.setItems(this._allItems.reduce((prev, cur) => {return prev.concat(cur)}, []));
        this._selection.setKeySelected(this.props.selectedColumn.property, true, false);
        this._isInitialized = true;
    }

    componentDidMount() {
        if (this._selection.getSelectedCount() === 0) {
            this._selection.setKeySelected(this.props.selectedColumn.property, true, false);
        }
    }

    public render(): React.ReactNode {
        const items = this.state.items.reduce((prev, cur) => {return prev.concat(cur)}, []);
        const selectedMeta = this.props.jointDataset.metaDict[this.state.selectedColumn.property];
        return (
            <Callout
                target={'#' + this.props.target}
                onDismiss={this.props.onCancel}
                setInitialFocus={true}
                hidden={false}
            >
                <div className={styles.wrapper}>
                    <div className={styles.leftHalf}>
                        <DetailsList
                            items={items}
                            // groups={this._groups}
                            // groupProps={{
                            // onRenderHeader: this._onRenderGroupHeader,
                            // onRenderFooter: this._onRenderGroupFooter
                            // }}
                            // getGroupHeight={this._getGroupHeight}
                            ariaLabelForSelectionColumn="Toggle selection"
                            ariaLabelForSelectAllCheckbox="Toggle selection for all items"
                            checkButtonAriaLabel="Row checkbox"
                            onRenderDetailsHeader={this._onRenderDetailsHeader}
                            selection={this._selection}
                            selectionPreservedOnEmptyClick={true}
                            //getKey={this.getKey}
                            setKey={"set"}
                            columns={[{key: 'col1', name: 'name', minWidth: 200, fieldName: 'title'}]}
                        />
                    </div>
                    <div className={styles.rightHalf}>
                        <div>Data summary</div>
                        {selectedMeta.isCategorical && (
                            <div>
                                <div>{`# of unique values: ${selectedMeta.sortedCategoricalValues.length}`}</div>
                                {this.props.canDither && (
                                    <Checkbox 
                                        label={"Should dither"}
                                        checked={this.state.selectedColumn.options.dither}
                                        onChange={this.ditherChecked}
                                    />
                                )}
                            </div>
                        )}
                        {selectedMeta.isCategorical === false && (
                            <div>
                                <div>{`min: ${selectedMeta.featureRange.min}`}</div>
                                <div>{`max: ${selectedMeta.featureRange.max}`}</div>
                                {this.state.binCount !== undefined && (
                                <SpinButton
                                    styles={{
                                        spinButtonWrapper: {maxWidth: "98px"},
                                        labelWrapper: { alignSelf: "center"},
                                        root: {
                                            display: "inline-flex",
                                            float: "right",
                                            selectors: {
                                                "> div": {
                                                    maxWidth: "108px"
                                                }
                                            }
                                        }
                                    }}
                                    label={localization.Filters.numericValue}
                                    min={this.MIN_HIST_COLS}
                                    max={this.MAX_HIST_COLS}
                                    value={this.state.binCount.toString()}
                                    onIncrement={this.setNumericValue.bind(this, 1, selectedMeta)}
                                    onDecrement={this.setNumericValue.bind(this, -1, selectedMeta)}
                                    onValidate={this.setNumericValue.bind(this, 0, selectedMeta)}
                                />
                                )}
                            </div>
                        )}
                    </div>
                </div>
                <DefaultButton 
                    text={"Accept"}
                    onClick={this.saveState}
                />
            </Callout>
        );
    }

    private readonly saveState = (): void => {
        if (this.state.binCount) {
            this.props.jointDataset.addBin(this.state.selectedColumn.property, this.state.binCount);
        }
        this.props.onAccept(this.state.selectedColumn);
    }

    private readonly _onRenderDetailsHeader = () => {
        return <div></div>
    }

    private readonly ditherChecked = (ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean): void => {
        this.setState({
            selectedColumn: {
                property: this._selection.getSelection()[0].key as string,
                options: {
                    dither: checked
                }
            }
        })
    }

    private readonly setNumericValue = (delta: number, column: IJointMeta, stringVal: string): string | void => {
        if (delta === 0) {
            const number = +stringVal;
            if (!Number.isInteger(number) || number > this.MAX_HIST_COLS || number < this.MIN_HIST_COLS) {
                return this.state.binCount.toString();
            }
            this.setState({binCount: number});
        } else {
            const prevVal = this.state.binCount as number;
            const newVal = prevVal + delta;
            if (newVal > this.MAX_HIST_COLS || newVal < this.MIN_HIST_COLS) {
                return prevVal.toString();
            }
            this.setState({binCount: newVal});
        }
    }

    private readonly _setSelection = (): void => {
        if (!this._isInitialized) {
            return;
        }
        const property = this._selection.getSelection()[0].key as string;
        const binCount = this._getBinCountForProperty(property);
        this.setState({selectedColumn: {
            property,
            options: {
                dither: this.props.canDither
            }
        }, binCount});
    }
    
    private _getBinCountForProperty(key: string): number | undefined {
        const selectedMeta = this.props.jointDataset.metaDict[key];
        let binCount = undefined;
        if (this.props.canBin && selectedMeta.isCategorical === false) {
            binCount = selectedMeta.sortedCategoricalValues !== undefined ?
                selectedMeta.sortedCategoricalValues.length : 5;
        }
        return binCount;
    }
}